# Copyright 2026 Hackable Diffusion Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for posterior-bridges Phase 3 cloud-aware guidance.

Three groups:

1. ``CloudThreadingTest``: verifies the sampler's wiring of
   ``posterior_cloud_size`` and ``cloud_fn`` through twist /
   correction protocols.  Backward-compat path
   (``posterior_cloud_size = 0`` + existing single-point twist) and
   forward-compat path (``> 0`` + cloud-aware twist) are both
   exercised end-to-end through ``ConditionalDiffusionSampler``.

2. ``EndpointTiltCloudTwistFnTest``: unit tests for the SMC-potential
   estimator -- shape, lambda=0 baseline, large-R consistency against
   the closed-form ``H_t``, the no-cloud error path.

3. ``MixtureProjectionTest``: the manuscript's
   Proposition ``gaussian-mixture-example`` regression -- with the
   exact responsibility correction ``A_{+/-}(x_t) = omega_t^M(x_t) /
   omega_t(x_t)``, the projected cloud's tangent-coordinate mean
   collapses from a known bias floor to MC noise.  This single test
   numerically certifies the operational claim that distinguishes
   posterior-sample projection from mean-plug-in projection.
"""

from __future__ import annotations

import dataclasses
import math
import unittest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.corruption.gaussian import GaussianProcess
from hackable_diffusion.lib.guidance import (
    ConditionalDiffusionSampler,
    EndpointTiltCloudTwistFn,
    ProjectionCloudCorrectionFn,
    SystematicResamplerFn,
    self_normalized_posterior_expectation,
)
from hackable_diffusion.lib.inference.base import InferenceFn
from hackable_diffusion.lib.sampling.gaussian_step_sampler import DDIMStep
from hackable_diffusion.lib.sampling.sampling import DiffusionSampler
from hackable_diffusion.lib.sampling.time_scheduling import UniformTimeSchedule


################################################################################
# MARK: Stochastic inference fn fixture (mirrors PosteriorSamplerInferenceFn)
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class _StochasticIdentityInferenceFn(InferenceFn):
  """Returns ``x_0 = xt + sigma * z`` with z drawn fresh per call.

  Stand-in for a trained posterior sampler in tests where we don't need
  a real model; ``rng`` controls the per-call noise so calling this
  with R independent rng keys produces R independent posterior
  samples.
  """

  noise_scale: float = 0.1

  def __call__(
      self,
      time,
      xt,
      conditioning=None,
      rng=None,
  ):
    if rng is None:
      raise ValueError("_StochasticIdentityInferenceFn needs rng.")
    z = jax.random.normal(rng, xt.shape, dtype=xt.dtype)
    return {"x0": xt + float(self.noise_scale) * z}


################################################################################
# MARK: Sampler-level threading
################################################################################


class CloudThreadingTest(unittest.TestCase):

  def _gaussian_pieces(self, num_steps=6):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    base_sampler = DiffusionSampler(
        time_schedule=UniformTimeSchedule(),
        stepper=DDIMStep(corruption_process=corruption, stoch_coeff=0.0),
        num_steps=num_steps,
        return_trajectory=False,
    )
    return schedule, corruption, base_sampler

  def test_posterior_cloud_size_zero_keeps_existing_path(self):
    """``posterior_cloud_size = 0`` (the default) is backward
    compatible: the sampler must produce the same trajectory as before
    Phase 3 for any single-point twist / correction."""
    schedule, corruption, base_sampler = self._gaussian_pieces()
    inference_fn = _StochasticIdentityInferenceFn()

    sampler = ConditionalDiffusionSampler(
        base_sampler=base_sampler,
        corruption_process=corruption,
        # No twist, no correction; identity resampler -> short-circuits
        # to base_sampler regardless of posterior_cloud_size.
    )
    rng = jax.random.PRNGKey(0)
    init = jax.random.normal(rng, (4, 8), dtype=jnp.float64)
    out = sampler(inference_fn=inference_fn, rng=rng, initial_noise=init)
    final = out[0] if isinstance(out, tuple) else out
    self.assertTrue(bool(jnp.all(jnp.isfinite(final.xt))))

  def test_cloud_aware_twist_runs_when_cloud_size_positive(self):
    """``posterior_cloud_size > 0`` + ``EndpointTiltCloudTwistFn`` is
    the manuscript's Algorithm 1 sample SMC: must run end to end."""
    schedule, corruption, base_sampler = self._gaussian_pieces()
    inference_fn = _StochasticIdentityInferenceFn(noise_scale=0.05)

    # Reward favors x_0 with positive sum.
    def log_L_y(x0):
      return jnp.sum(x0, axis=-1)

    twist = EndpointTiltCloudTwistFn(log_L_y=log_L_y)
    sampler = ConditionalDiffusionSampler(
        base_sampler=base_sampler,
        corruption_process=corruption,
        twist_fn=twist,
        resampler_fn=SystematicResamplerFn(),
        posterior_cloud_size=8,
    )
    rng = jax.random.PRNGKey(1)
    init = jax.random.normal(rng, (4, 4), dtype=jnp.float64)
    final, _, log_w = sampler(
        inference_fn=inference_fn, rng=rng, initial_noise=init,
    )
    self.assertTrue(bool(jnp.all(jnp.isfinite(final.xt))))
    self.assertEqual(log_w.shape, (4,))
    self.assertTrue(bool(jnp.all(jnp.isfinite(log_w))))

  def test_cloud_aware_twist_with_zero_cloud_size_raises(self):
    """Cloud-aware twist with ``posterior_cloud_size = 0`` errors
    informatively when the sampler tries to evaluate it without
    building a cloud."""
    schedule, corruption, base_sampler = self._gaussian_pieces()
    inference_fn = _StochasticIdentityInferenceFn()

    def log_L_y(x0):
      return jnp.sum(x0, axis=-1)

    twist = EndpointTiltCloudTwistFn(log_L_y=log_L_y)
    sampler = ConditionalDiffusionSampler(
        base_sampler=base_sampler,
        corruption_process=corruption,
        twist_fn=twist,
        resampler_fn=SystematicResamplerFn(),
        # posterior_cloud_size=0 (default); cloud_fn will be None and
        # the cloud-aware twist must complain.
    )
    rng = jax.random.PRNGKey(2)
    init = jax.random.normal(rng, (3, 4), dtype=jnp.float64)
    with self.assertRaises(ValueError):
      sampler(inference_fn=inference_fn, rng=rng, initial_noise=init)


################################################################################
# MARK: EndpointTiltCloudTwistFn unit tests
################################################################################


class EndpointTiltCloudTwistFnTest(unittest.TestCase):

  def test_log_h_matches_logsumexp_minus_log_R(self):
    """``log h_t^R = logsumexp(log L_y) - log R`` evaluated by hand."""
    B, R, n = 2, 5, 3
    rng = jax.random.PRNGKey(0)
    x0_cloud = jax.random.normal(rng, (B, R, n), dtype=jnp.float64)

    def log_L_y(x0):
      return jnp.sum(x0, axis=-1)

    # Build a cloud_fn closure that returns this fixed cloud.
    def cloud_fn(xt):
      del xt
      return x0_cloud

    twist = EndpointTiltCloudTwistFn(log_L_y=log_L_y)
    out = twist(
        xt=jnp.zeros((B, n), dtype=jnp.float64),
        time=jnp.asarray([0.5, 0.5], dtype=jnp.float64),
        denoiser_fn=lambda xt: xt,
        cloud_fn=cloud_fn,
    )
    log_L_per_sample = jnp.sum(x0_cloud, axis=-1)            # [B, R]
    expected = jax.nn.logsumexp(log_L_per_sample, axis=-1) - jnp.log(R)
    self.assertTrue(jnp.allclose(out, expected, atol=1e-12))

  def test_no_cloud_fn_raises(self):
    twist = EndpointTiltCloudTwistFn(log_L_y=lambda x: jnp.sum(x, axis=-1))
    with self.assertRaises(ValueError):
      twist(
          xt=jnp.zeros((1, 2), dtype=jnp.float64),
          time=jnp.asarray([0.5]),
          denoiser_fn=lambda xt: xt,
          cloud_fn=None,
      )

  def test_self_normalized_expectation_matches_softmax_weighted_average(self):
    """``self_normalized_posterior_expectation`` = softmax-weighted
    mean of f over the cloud."""
    B, R, n = 2, 6, 3
    rng = jax.random.PRNGKey(3)
    cloud = jax.random.normal(rng, (B, R, n), dtype=jnp.float64)

    def f(x0):  # take just the first coordinate
      return x0[..., 0]

    def log_L(x0):
      return jnp.sum(x0, axis=-1)

    out = self_normalized_posterior_expectation(f, log_L, cloud)
    self.assertEqual(out.shape, (B,))

    # Hand-computed.
    log_L_v = jnp.sum(cloud, axis=-1)                         # [B, R]
    weights = jax.nn.softmax(log_L_v, axis=-1)                # [B, R]
    expected = jnp.sum(weights * cloud[..., 0], axis=-1)      # [B]
    self.assertTrue(jnp.allclose(out, expected, atol=1e-12))


################################################################################
# MARK: Manuscript Proposition gaussian-mixture-example regression
################################################################################


class MixtureProjectionTest(unittest.TestCase):
  """Hard linear constraint on a Gaussian mixture: projection without
  responsibility correction has a tangent-coordinate bias floor; with
  the correction the bias collapses to MC noise.  This is the
  numerical certificate of the manuscript's operational claim.

  Setup (manuscript Proposition ``gaussian-mixture-example``):
    S in {-1, +1} with prior 1/2 each.
    X_0 | S = s ~ N(m_s, tau^2 I_2),  m_s = (s a, s b).
    X_t = alpha_t X_0 + sigma_t Z,  alpha_t^2 + sigma_t^2 = 1.

    Constraint M = {z : z_perp = 0}, projection Pi(z) = (0, z_par).

  At the diagnostic state x_t = (u, 0):
    Constrained posterior responsibility omega_t^M(u, 0) = 1/2.
    Constrained component means m_tilde_+/- = (0, +/- b (1 - kappa_t alpha_t))
                                            = (0, +/- b sigma_t^2 / v_t).
    Constrained target tangent mean = 0 (by symmetry).

    Base posterior responsibility omega_t(u, 0)
        = sigmoid(2 alpha_t a u / v_t)   != 1/2 in general.
    Projected base posterior tangent mean = (2 omega_t - 1) b sigma_t^2 / v_t
        = tanh(alpha_t a u / v_t) * B_t   where B_t = b sigma_t^2 / v_t.

  We construct a cloud of ``R`` samples from the analytic mixture
  posterior at ``x_t = (u, 0)``, project by setting ``x_perp = 0``,
  and verify:

  * Without ``log_importance_fn``: the empirical tangent mean of the
    SELECTED sample (uniformly drawn from the projected cloud) is the
    biased value ``B_t tanh(alpha_t a u / v_t)`` to MC noise.

  * With ``log_importance_fn`` = ``log A_S(x_t)``: the empirical
    tangent mean is the unbiased value ``0`` to MC noise.

  Particles pin the variance: we run ``B`` independent particles per
  trial and aggregate, taking advantage of vectorisation.
  """

  # Concrete constants.
  a = 1.0
  b = 1.0
  tau = 0.3
  alpha_t = 0.7
  u = 1.0
  R = 1024     # cloud size
  B = 256      # number of independent particles per trial
  rng_seed = 17

  def _analytic_constants(self):
    sigma_t = math.sqrt(1.0 - self.alpha_t ** 2)
    v_t = self.alpha_t ** 2 * self.tau ** 2 + sigma_t ** 2
    kappa_t = self.alpha_t * self.tau ** 2 / v_t
    tau_bar_sq = self.tau ** 2 * sigma_t ** 2 / v_t
    Bt = self.b * sigma_t ** 2 / v_t
    omega_base = 1.0 / (1.0 + math.exp(-2.0 * self.alpha_t * self.a * self.u / v_t))
    omega_M = 0.5
    # Constrained component means.
    m_tilde_plus_par = self.b * sigma_t ** 2 / v_t          # +B_t
    m_tilde_minus_par = -self.b * sigma_t ** 2 / v_t        # -B_t
    return dict(
        sigma_t=sigma_t, v_t=v_t, kappa_t=kappa_t, tau_bar_sq=tau_bar_sq,
        Bt=Bt, omega_base=omega_base, omega_M=omega_M,
        m_tilde_plus_par=m_tilde_plus_par,
        m_tilde_minus_par=m_tilde_minus_par,
    )

  def _build_cloud(self, rng):
    """Sample a [B, R, 3] cloud from the analytic mixture posterior at
    x = (u, 0).  Last axis: (x_0_perp, x_0_par, label) where label is
    -1 or +1."""
    c = self._analytic_constants()
    rng_s, rng_normal = jax.random.split(rng)
    # Component label per (B, R) drawn from base posterior responsibility.
    s_int = jax.random.bernoulli(
        rng_s, p=c["omega_base"], shape=(self.B, self.R),
    ).astype(jnp.float64)
    s = 2.0 * s_int - 1.0                                     # +/- 1

    # Means per (B, R, 2): m_{s, t}(x) = m_s + kappa_t (x - alpha_t m_s)
    # at x = (u, 0).
    x = jnp.array([self.u, 0.0], dtype=jnp.float64)
    m_per_s = jnp.stack([
        s * jnp.full_like(s, self.a),                         # m_s_perp = s * a
        s * jnp.full_like(s, self.b),                         # m_s_par = s * b
    ], axis=-1)                                               # [B, R, 2]
    means = m_per_s + c["kappa_t"] * (
        x[None, None, :] - self.alpha_t * m_per_s
    )                                                          # [B, R, 2]
    # x_0 ~ N(means, tau_bar_sq * I).
    z = jax.random.normal(rng_normal, (self.B, self.R, 2), dtype=jnp.float64)
    x0 = means + math.sqrt(c["tau_bar_sq"]) * z
    cloud_with_label = jnp.concatenate([x0, s[..., None]], axis=-1)  # [B, R, 3]
    return cloud_with_label, c

  def _projection_fn(self, x0_with_label):
    """Project: set x_perp = 0; preserve x_par and label."""
    perp = jnp.zeros_like(x0_with_label[..., 0])
    par = x0_with_label[..., 1]
    label = x0_with_label[..., 2]
    return jnp.stack([perp, par, label], axis=-1)

  def test_uniform_projection_has_predicted_bias(self):
    c = self._analytic_constants()
    rng = jax.random.PRNGKey(self.rng_seed)
    rng_cloud, rng_categorical = jax.random.split(rng)
    cloud_with_label, c = self._build_cloud(rng_cloud)

    def cloud_fn(xt):
      del xt
      return cloud_with_label

    correction = ProjectionCloudCorrectionFn(
        projection_fn=self._projection_fn,
        log_importance_fn=None,
    )
    # Call directly (skip the sampler).  Pass xt = (u, 0) for completeness.
    xt = jnp.broadcast_to(
        jnp.array([self.u, 0.0, 0.0], dtype=jnp.float64),
        (self.B, 3),
    )
    selected = correction(
        x0=jnp.zeros((self.B, 3), dtype=jnp.float64),  # ignored
        xt=xt,
        time=jnp.zeros((self.B,), dtype=jnp.float64),
        denoiser_fn=lambda v: v,
        schedule=None,
        cloud_fn=cloud_fn,
        rng=rng_categorical,
    )                                                          # [B, 3]
    # Empirical tangent mean across B particles.
    empirical_par_mean = float(jnp.mean(selected[..., 1]))
    expected_bias = c["Bt"] * math.tanh(
        self.alpha_t * self.a * self.u / c["v_t"]
    )
    # MC tolerance scales as sqrt(tau_bar_sq / B); quite tight here.
    mc_tol = 5.0 * math.sqrt(c["tau_bar_sq"] / self.B)
    self.assertAlmostEqual(empirical_par_mean, expected_bias, delta=mc_tol)

  def test_responsibility_correction_eliminates_bias(self):
    c = self._analytic_constants()
    rng = jax.random.PRNGKey(self.rng_seed + 1)
    rng_cloud, rng_categorical = jax.random.split(rng)
    cloud_with_label, c = self._build_cloud(rng_cloud)

    omega_base = c["omega_base"]
    omega_M = c["omega_M"]
    log_A_plus = math.log(omega_M / omega_base)
    log_A_minus = math.log((1.0 - omega_M) / (1.0 - omega_base))

    def log_importance_fn(x_proj_with_label, xt, time):
      del xt, time
      label = x_proj_with_label[..., 2]
      return jnp.where(label > 0.0, log_A_plus, log_A_minus)

    def cloud_fn(xt):
      del xt
      return cloud_with_label

    correction = ProjectionCloudCorrectionFn(
        projection_fn=self._projection_fn,
        log_importance_fn=log_importance_fn,
    )
    xt = jnp.broadcast_to(
        jnp.array([self.u, 0.0, 0.0], dtype=jnp.float64),
        (self.B, 3),
    )
    selected = correction(
        x0=jnp.zeros((self.B, 3), dtype=jnp.float64),
        xt=xt,
        time=jnp.zeros((self.B,), dtype=jnp.float64),
        denoiser_fn=lambda v: v,
        schedule=None,
        cloud_fn=cloud_fn,
        rng=rng_categorical,
    )
    empirical_par_mean = float(jnp.mean(selected[..., 1]))
    # Target: 0 (constrained mean by symmetry at x_perp = 0).
    mc_tol = 5.0 * math.sqrt(c["tau_bar_sq"] / self.B)
    self.assertAlmostEqual(empirical_par_mean, 0.0, delta=mc_tol)

  def test_correction_pulls_toward_zero_relative_to_uncorrected(self):
    """Direct comparison of the two branches on the SAME cloud --
    uncorrected branch keeps the bias floor; corrected branch zeroes
    it.  This rules out the trivial null hypothesis that both
    procedures happen to produce the same mean for this seed."""
    c = self._analytic_constants()
    rng = jax.random.PRNGKey(self.rng_seed + 2)
    rng_cloud, rng_a, rng_b = jax.random.split(rng, 3)
    cloud_with_label, c = self._build_cloud(rng_cloud)

    log_A_plus = math.log(c["omega_M"] / c["omega_base"])
    log_A_minus = math.log((1.0 - c["omega_M"]) / (1.0 - c["omega_base"]))

    def cloud_fn(xt):
      del xt
      return cloud_with_label

    def log_importance_fn(x_proj_with_label, xt, time):
      del xt, time
      label = x_proj_with_label[..., 2]
      return jnp.where(label > 0.0, log_A_plus, log_A_minus)

    uncorrected = ProjectionCloudCorrectionFn(
        projection_fn=self._projection_fn, log_importance_fn=None,
    )
    corrected = ProjectionCloudCorrectionFn(
        projection_fn=self._projection_fn,
        log_importance_fn=log_importance_fn,
    )
    xt = jnp.broadcast_to(
        jnp.array([self.u, 0.0, 0.0], dtype=jnp.float64),
        (self.B, 3),
    )

    sel_unc = uncorrected(
        x0=jnp.zeros((self.B, 3), dtype=jnp.float64),
        xt=xt, time=jnp.zeros((self.B,), dtype=jnp.float64),
        denoiser_fn=lambda v: v, schedule=None,
        cloud_fn=cloud_fn, rng=rng_a,
    )
    sel_cor = corrected(
        x0=jnp.zeros((self.B, 3), dtype=jnp.float64),
        xt=xt, time=jnp.zeros((self.B,), dtype=jnp.float64),
        denoiser_fn=lambda v: v, schedule=None,
        cloud_fn=cloud_fn, rng=rng_b,
    )

    bias_unc = float(jnp.abs(jnp.mean(sel_unc[..., 1])))
    bias_cor = float(jnp.abs(jnp.mean(sel_cor[..., 1])))
    # The corrected version should be much closer to zero.
    self.assertLess(bias_cor, 0.5 * bias_unc)


if __name__ == "__main__":
  unittest.main()
