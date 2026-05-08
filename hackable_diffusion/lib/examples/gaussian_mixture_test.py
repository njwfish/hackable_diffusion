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

"""End-to-end integration tests on a Gaussian mixture target.

These tests compose the analytic :class:`GaussianMixtureBridge` with
the framework's cloud-aware machinery
(:class:`EndpointTiltCloudTwistFn`,
:func:`make_posterior_cloud_fn`,
:class:`ConditionalDiffusionSampler`) and verify the framework's
output matches closed-form analytic references.  The "ground truth"
inference fn is :func:`posterior_sampler_inference_fn`, which draws
*exact* posterior samples from the analytic mixture; this isolates
framework wiring from model error.

Three groups:

1. ``PotentialEstimatorTest``: the manuscript's central operational
   distinction -- mean-plug-in ``L_y(E[X_0 | x_t])`` vs posterior-MC
   ``(1/R) sum L_y(x_0^r)`` -- on a controlled GMM where the analytic
   ``H_t(x_t)`` is closed-form.  Verifies the cloud estimator
   converges to the analytic value as ``R`` grows; verifies the
   mean-plug-in is *biased* (does not converge to the same value).

2. ``SMCConcentrationTest``: drives ``ConditionalDiffusionSampler``
   with the analytic posterior sampler + ``EndpointTiltCloudTwistFn``
   over a multi-step grid.  Verifies the final-particle distribution
   concentrates on the mode of the tilted posterior ``p_0^y`` (the
   mode the reward favours), not on the unbiased base mean.

3. ``SelfNormalizedExpectationTest``: verifies that the
   self-normalised endpoint MC estimator matches the analytic tilted-
   posterior expectation for a smooth functional.
"""

from __future__ import annotations

import math
import unittest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.corruption.gaussian import GaussianProcess
from hackable_diffusion.lib.examples import (
    alpha_from_schedule,
    GaussianMixture,
    GaussianMixtureBridge,
    posterior_sampler_inference_fn,
)
from hackable_diffusion.lib.guidance import (
    ConditionalDiffusionSampler,
    EndpointTiltCloudTwistFn,
    SystematicResamplerFn,
    make_posterior_cloud_fn,
    self_normalized_posterior_expectation,
)
from hackable_diffusion.lib.sampling.gaussian_step_sampler import DDIMStep
from hackable_diffusion.lib.sampling.sampling import DiffusionSampler
from hackable_diffusion.lib.sampling.time_scheduling import UniformTimeSchedule


def _two_mode_mixture(separation: float = 1.5):
  """Two-component 2D GMM with modes at +/- (separation, 0)."""
  return GaussianMixture(
      weights=jnp.array([0.5, 0.5], dtype=jnp.float64),
      means=jnp.array(
          [[separation, 0.0], [-separation, 0.0]], dtype=jnp.float64,
      ),
      component_var=0.05,
  )


################################################################################
# MARK: Posterior-MC potential estimator vs mean plug-in
################################################################################


class PotentialEstimatorTest(unittest.TestCase):
  """The manuscript's central operational claim: posterior-MC for
  ``H_t(x_t) = E[L_y(X_0) | x_t]`` is consistent; mean-plug-in
  ``L_y(E[X_0 | x_t])`` is biased away from ``H_t`` whenever the
  posterior is non-trivially spread or multimodal.

  We verify on a 2D two-mode GMM with a Gaussian tilt and a noisy
  ``x_t`` that lies between the modes, where the posterior is
  bimodal.
  """

  def _setup(self):
    mix = _two_mode_mixture(separation=1.5)
    schedule = schedules.CosineSchedule()
    bridge = GaussianMixtureBridge(
        mixture=mix, alpha_fn=alpha_from_schedule(schedule),
    )
    return mix, bridge, schedule

  def test_cloud_potential_converges_to_analytic_H_t(self):
    """As ``R -> infty``, ``log [(1/R) sum L_y(x_0^r)]`` -> ``log H_t``."""
    mix, bridge, schedule = self._setup()
    inference_fn = posterior_sampler_inference_fn(bridge)

    # Diagnostic state: noisy point pulled toward +mode at moderate t.
    t = jnp.asarray(0.4, dtype=jnp.float64)
    xt = jnp.asarray([[0.6, 0.0]], dtype=jnp.float64)          # B=1
    y = jnp.asarray([1.5, 0.0], dtype=jnp.float64)             # tilt toward +mode
    sigma_y = 0.6

    log_H_analytic = float(
        bridge.gaussian_tilt_log_potential(xt, t, y, sigma_y=sigma_y)[0]
    )

    def log_L_y(x0):
      diff = x0 - y[None, :]
      return -0.5 * jnp.sum(diff ** 2, axis=-1) / sigma_y ** 2

    # Increasing R: the cloud estimator should approach analytic.
    errs = []
    for R, seed in [(8, 0), (64, 1), (512, 2), (4096, 3)]:
      rng = jax.random.PRNGKey(seed)
      cloud_fn = make_posterior_cloud_fn(
          inference_fn, GaussianProcess(schedule=schedule),
          time=t, rng=rng, population_size=R,
      )
      cloud = cloud_fn(xt)                                     # [B=1, R, 2]
      log_L = jax.vmap(log_L_y, in_axes=1, out_axes=1)(cloud)  # [1, R]
      log_h_R = float(jax.nn.logsumexp(log_L, axis=-1)[0]) - math.log(R)
      errs.append(abs(log_h_R - log_H_analytic))

    # Errors should decrease (not strictly monotone with finite R, but
    # large enough sequence shows convergence).  Compare smallest two
    # to largest two: small-R average error >> large-R average error.
    avg_small = (errs[0] + errs[1]) / 2.0
    avg_large = (errs[2] + errs[3]) / 2.0
    self.assertLess(avg_large, avg_small)
    # Sanity: largest R should be within a few * 1/sqrt(R) of analytic.
    self.assertLess(errs[-1], 0.1)

  def test_mean_plug_in_is_biased_for_bimodal_posterior(self):
    """Mean-plug-in ``log L_y(E[X_0 | x_t])`` should be far from
    analytic ``log H_t`` when the posterior is bimodal.  We construct
    such a state and check the gap is significant."""
    mix, bridge, schedule = self._setup()
    t = jnp.asarray(0.5, dtype=jnp.float64)
    xt = jnp.asarray([[0.0, 0.0]], dtype=jnp.float64)           # midpoint
    y = jnp.asarray([1.5, 0.0], dtype=jnp.float64)
    sigma_y = 0.4

    log_H_analytic = float(
        bridge.gaussian_tilt_log_potential(xt, t, y, sigma_y=sigma_y)[0]
    )
    posterior_mean = bridge.posterior_mean(xt, t)              # [1, 2]
    diff = posterior_mean[0] - y
    log_L_at_mean = float(-0.5 * jnp.sum(diff ** 2) / sigma_y ** 2)
    # Mean-plug-in approximates H_t by L_y(posterior_mean).
    bias = abs(log_L_at_mean - log_H_analytic)
    # By the cumulant expansion (manuscript position note), the bias
    # for a Gaussian L_y is at least the posterior covariance term
    # 0.5 * tr(C_post / sigma_y^2).  For our setup that's > 1 nat;
    # at least require >> 0.1 nat to certify the bias is real.
    self.assertGreater(bias, 0.5)


################################################################################
# MARK: SMC sampler concentrates on the tilted-posterior mode
################################################################################


class SMCConcentrationTest(unittest.TestCase):
  """Runs ``ConditionalDiffusionSampler`` with the analytic posterior
  sampler as ``inference_fn``, ``EndpointTiltCloudTwistFn``, and
  ``SystematicResamplerFn``.  The final particles must concentrate
  near the mode of the tilted posterior, not the unbiased base mean.

  Setup: symmetric two-mode GMM at ``+/- (a, 0)``.  Tilt strongly
  toward ``+a``; with enough particles and steps the SMC sampler must
  produce samples whose mean has positive ``x``-coordinate (matching
  the tilt).
  """

  def test_smc_pulls_samples_toward_tilted_mode(self):
    a = 1.5
    mix = _two_mode_mixture(separation=a)
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    bridge = GaussianMixtureBridge(
        mixture=mix, alpha_fn=alpha_from_schedule(schedule),
    )
    inference_fn = posterior_sampler_inference_fn(bridge)

    # Strong tilt toward +mode.
    y = jnp.asarray([a, 0.0], dtype=jnp.float64)
    sigma_y = 0.3
    def log_L_y(x0):
      diff = x0 - y[None, :]
      return -0.5 * jnp.sum(diff ** 2, axis=-1) / sigma_y ** 2

    twist = EndpointTiltCloudTwistFn(log_L_y=log_L_y)
    base_sampler = DiffusionSampler(
        time_schedule=UniformTimeSchedule(),
        stepper=DDIMStep(corruption_process=corruption, stoch_coeff=0.5),
        num_steps=20,
        return_trajectory=False,
    )
    sampler = ConditionalDiffusionSampler(
        base_sampler=base_sampler,
        corruption_process=corruption,
        twist_fn=twist,
        resampler_fn=SystematicResamplerFn(),
        posterior_cloud_size=8,
    )

    P = 256                                                    # particles
    rng = jax.random.PRNGKey(0)
    init = jax.random.normal(rng, (P, 2), dtype=jnp.float64)
    final, _, log_w = sampler(
        inference_fn=inference_fn, rng=rng, initial_noise=init,
    )
    samples = np.asarray(final.xt)
    # Mean of the empirical (uniformly weighted -- last step resampled
    # so weights are uniform across the cloud) distribution should
    # have positive x.
    mean_x = float(samples[:, 0].mean())
    # The tilted target's mode is close to (+a, 0); the unbiased base
    # mean is (0, 0).  Weak claim: empirical mean has the right sign
    # and is at least halfway to the tilted mode.
    self.assertGreater(mean_x, a / 2.0)


################################################################################
# MARK: Self-normalised endpoint expectation
################################################################################


class SelfNormalizedExpectationTest(unittest.TestCase):
  """``self_normalized_posterior_expectation`` should converge to the
  analytic tilted-posterior expectation as ``R`` grows.

  We test on a smooth functional ``f(x_0) = x_0[0]`` (the first
  coordinate) and a Gaussian tilt.  The closed-form tilted-posterior
  mean of ``x_0[0]`` is reweighting the unconstrained component means
  by the tilted weights -- compute it analytically from the bridge.
  """

  def _setup(self):
    mix = _two_mode_mixture(separation=1.5)
    schedule = schedules.CosineSchedule()
    bridge = GaussianMixtureBridge(
        mixture=mix, alpha_fn=alpha_from_schedule(schedule),
    )
    return mix, bridge, schedule

  def _analytic_tilted_first_coord_mean(
      self, bridge, x_t, t, y, sigma_y,
  ):
    """E[x_0[0] | x_t, Y=y] = sum_k tilted_r_k(x_t) m_{k, t}(x_t)[0]
    where tilted_r_k(x_t) propto r_k(x_t) * E[exp(-||X_0 - y||^2 /
    (2 sigma_y^2)) | X_t, S=k] * sigma_y^D / (tau_bar^2 + sigma_y^2)^(D/2).
    """
    r = bridge.posterior_responsibilities(x_t, t)              # [B, K]
    component_means = bridge.posterior_component_means(x_t, t) # [B, K, D]
    tau_bar_sq = bridge.posterior_component_var(t)
    sigma_y_sq = float(sigma_y) ** 2
    denom = tau_bar_sq + sigma_y_sq
    diff = component_means - y[None, None, :]                  # [B, K, D]
    sq = jnp.sum(diff ** 2, axis=-1)                           # [B, K]
    log_per_k = -sq / (2.0 * denom)                            # [B, K]
    log_tilt = jnp.log(jnp.clip(r, 1e-30, None)) + log_per_k
    tilted_r = jax.nn.softmax(log_tilt, axis=-1)               # [B, K]
    # E[x_0[0] under tilted component k] = m_{k, t}(x_t)[0] (Gaussian
    # mean is unchanged under a Gaussian tilt; only the weighting
    # changes when we look at the mixture).
    return jnp.sum(tilted_r * component_means[..., 0], axis=-1)  # [B]

  def test_estimator_matches_analytic_tilted_mean(self):
    mix, bridge, schedule = self._setup()
    inference_fn = posterior_sampler_inference_fn(bridge)
    t = jnp.asarray(0.4, dtype=jnp.float64)
    xt = jnp.asarray([[0.3, 0.0]], dtype=jnp.float64)
    y = jnp.asarray([1.5, 0.0], dtype=jnp.float64)
    sigma_y = 0.5

    analytic = float(
        self._analytic_tilted_first_coord_mean(bridge, xt, t, y, sigma_y)[0]
    )

    def f(x0):
      return x0[..., 0]

    def log_L(x0):
      diff = x0 - y[None, :]
      return -0.5 * jnp.sum(diff ** 2, axis=-1) / sigma_y ** 2

    rng = jax.random.PRNGKey(0)
    R = 4096
    cloud_fn = make_posterior_cloud_fn(
        inference_fn, GaussianProcess(schedule=schedule),
        time=t, rng=rng, population_size=R,
    )
    cloud = cloud_fn(xt)                                       # [1, R, 2]
    out = float(self_normalized_posterior_expectation(f, log_L, cloud)[0])

    self.assertAlmostEqual(out, analytic, delta=0.1)


if __name__ == "__main__":
  unittest.main()
