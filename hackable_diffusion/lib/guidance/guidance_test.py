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

"""Pure unit tests for ``lib/guidance`` primitives.

Every test here depends only on hackable_diffusion itself (plus a tiny
``_MeanForwardFn`` test fixture) -- no external project code.  Exercises
protocols, resamplers, corrections, twists, the DDIM proposal-ratio
formula, the registry, and the ``ConditionalDiffusionSampler``
backward-compat path.
"""

from __future__ import annotations

import dataclasses
import unittest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.corruption.gaussian import GaussianProcess
from hackable_diffusion.lib.sampling.gaussian_step_sampler import (
    AdjustedDDIMStep,
    DDIMStep,
    HeunStep,
    SdeStep,
    VelocityStep,
)
from hackable_diffusion.lib.inference.guidance import ScalarGuidanceFn

from hackable_diffusion.lib.guidance.adapters import CFGCorrectionFn
from hackable_diffusion.lib.sampling.sampling import DiffusionSampler
from hackable_diffusion.lib.sampling.time_scheduling import UniformTimeSchedule

from hackable_diffusion.lib.guidance.corrections import (
    GradientCorrectionFn,
    IteratedCorrectionFn,
    PiGDMCorrectionFn,
    dps_prefactor,
    miyasawa_prefactor,
)
from hackable_diffusion.lib.guidance.forward_ops import (
    ComposeForwardFn,
    ConvForwardFn,
    InpaintingForwardFn,
    LinearForwardFn,
    SubsampleForwardFn,
)
from hackable_diffusion.lib.guidance.linalg import (
    batch_inner,
    batched_cg,
    linear_adjoint,
)
from hackable_diffusion.lib.guidance.posterior_covariance import (
    FixedPriorPosteriorCovarianceFn,
    IsotropicPosteriorCovarianceFn,
    TweediePosteriorCovarianceFn,
    miyasawa_scale,
)
from hackable_diffusion.lib.guidance.proposal_ratio import (
    ddim_proposal_log_ratio,
    proposal_log_ratio,
    register_proposal_ratio,
    sde_proposal_log_ratio,
    velocity_proposal_log_ratio,
)
from hackable_diffusion.lib.guidance.resamplers import (
    ESSThresholdedResamplerFn,
    MultinomialResamplerFn,
    NoResamplerFn,
    SystematicResamplerFn,
    normalised_weights,
)
from hackable_diffusion.lib.guidance.sampler import ConditionalDiffusionSampler
from hackable_diffusion.lib.guidance.twists import (
    ClassifierTwistFn,
    EnergyTwistFn,
    GaussianLikelihoodTwistFn,
)
from hackable_diffusion.lib.guidance.utils import (
    accepts_rng_kwarg,
    call_inference_fn,
    scalar_alpha_sigma,
)


################################################################################
# Shared fixtures
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class _MeanForwardFn:
  """Minimal ``ForwardFn``: mean over the last axis, keepdims=True."""

  def forward(self, x: jax.Array) -> jax.Array:
    return jnp.mean(x, axis=-1, keepdims=True)


def _identity_x0_inference_fn(x0_fixed: jax.Array):
  """Return a deterministic inference fn that always emits ``x0_fixed``."""

  def fn(xt, time, conditioning=None):
    del xt, time, conditioning
    return {"x0": x0_fixed}

  return fn


def _gaussian_pieces(eta: float = 0.0, num_steps: int = 8):
  """Return a (schedule, corruption, base_sampler) triple for tests."""
  schedule = schedules.CosineSchedule()
  corruption = GaussianProcess(schedule=schedule)
  stepper = DDIMStep(corruption_process=corruption, stoch_coeff=eta)
  base_sampler = DiffusionSampler(
      time_schedule=UniformTimeSchedule(),
      stepper=stepper,
      num_steps=num_steps,
      return_trajectory=False,
  )
  return schedule, corruption, base_sampler


################################################################################
# Utils
################################################################################


class UtilsTest(unittest.TestCase):

  def test_normalised_weights_sums_to_one(self):
    log_w = jnp.log(jnp.asarray([1.0, 2.0, 3.0, 4.0]))
    weights, _ = normalised_weights(log_w)
    self.assertAlmostEqual(float(jnp.sum(weights)), 1.0, places=12)

  def test_normalised_weights_handles_large_log_values(self):
    # Without shift this would overflow; normalised_weights must shift.
    log_w = jnp.asarray([1e6, 1e6 + 1.0])
    weights, log_mean = normalised_weights(log_w)
    self.assertTrue(bool(jnp.all(jnp.isfinite(weights))))
    self.assertTrue(bool(jnp.isfinite(log_mean)))

  def test_accepts_rng_kwarg_true_for_explicit_rng(self):
    def f(xt, time, rng=None): return None
    self.assertTrue(accepts_rng_kwarg(f))

  def test_accepts_rng_kwarg_true_for_var_kwargs(self):
    def f(xt, time, **kwargs): return None
    self.assertTrue(accepts_rng_kwarg(f))

  def test_accepts_rng_kwarg_false_when_no_rng(self):
    def f(xt, time, conditioning=None): return None
    self.assertFalse(accepts_rng_kwarg(f))

  def test_call_inference_fn_passes_rng_when_accepted(self):
    captured = {}
    def f(xt, time, conditioning=None, rng=None):
      captured["rng"] = rng
      return {"x0": xt}
    call_inference_fn(
        f, xt=jnp.zeros((1, 4)), time=jnp.asarray([0.5]),
        rng=jax.random.PRNGKey(0),
    )
    self.assertIsNotNone(captured["rng"])

  def test_call_inference_fn_skips_rng_when_not_accepted(self):
    def f(xt, time, conditioning=None):
      return {"x0": xt}
    out = call_inference_fn(
        f, xt=jnp.zeros((1, 4)), time=jnp.asarray([0.5]),
        rng=jax.random.PRNGKey(0),
    )
    self.assertIn("x0", out)

  def test_scalar_alpha_sigma_returns_scalars(self):
    schedule = schedules.CosineSchedule()
    alpha, sigma = scalar_alpha_sigma(schedule, jnp.asarray([0.5]))
    self.assertEqual(alpha.shape, ())
    self.assertEqual(sigma.shape, ())


################################################################################
# Resamplers
################################################################################


class ResamplerTest(unittest.TestCase):

  def test_no_resampler_is_identity(self):
    particles = jnp.arange(5, dtype=jnp.float64).reshape(5, 1)
    log_w = jnp.zeros(5)
    p_out, w_out = NoResamplerFn()(
        particles, log_w, rng=jax.random.PRNGKey(0),
    )
    self.assertTrue(bool(jnp.all(p_out == particles)))
    self.assertTrue(bool(jnp.all(w_out == log_w)))

  def test_systematic_resampler_preserves_count_and_equalises_weights(self):
    k = 32
    particles = jnp.arange(k, dtype=jnp.float64).reshape(k, 1)
    log_w = jnp.zeros(k)
    p_out, w_out = SystematicResamplerFn()(
        particles, log_w, rng=jax.random.PRNGKey(0),
    )
    self.assertEqual(p_out.shape, particles.shape)
    self.assertTrue(jnp.allclose(w_out, w_out[0]))

  def test_multinomial_resampler_concentrates_on_heavy_particle(self):
    k = 1024
    particles = jnp.arange(k, dtype=jnp.float64).reshape(k, 1)
    log_w = jnp.full((k,), -1e3)
    heavy = 500
    log_w = log_w.at[heavy].set(0.0)
    p_out, _ = MultinomialResamplerFn()(
        particles, log_w, rng=jax.random.PRNGKey(1),
    )
    frac_heavy = float(jnp.mean(p_out[:, 0] == heavy))
    self.assertGreater(frac_heavy, 0.95)

  def test_ess_thresholded_triggers_below_threshold(self):
    k = 128
    particles = jnp.arange(k, dtype=jnp.float64).reshape(k, 1)
    # One heavy particle => normalised ESS ≈ 1/k << 0.5.
    log_w = jnp.full((k,), -1e3).at[0].set(0.0)
    resampler = ESSThresholdedResamplerFn(
        base=SystematicResamplerFn(), threshold=0.5,
    )
    p_out, _ = resampler(particles, log_w, rng=jax.random.PRNGKey(0))
    self.assertTrue(bool(jnp.all(p_out == particles[:1])))

  def test_ess_thresholded_skips_when_above_threshold(self):
    k = 64
    particles = jnp.arange(k, dtype=jnp.float64).reshape(k, 1)
    log_w = jnp.zeros(k)  # uniform => normalised ESS = 1.0
    resampler = ESSThresholdedResamplerFn(
        base=SystematicResamplerFn(), threshold=0.5,
    )
    p_out, w_out = resampler(particles, log_w, rng=jax.random.PRNGKey(0))
    self.assertTrue(bool(jnp.all(p_out == particles)))
    self.assertTrue(bool(jnp.all(w_out == log_w)))


################################################################################
# Prefactors
################################################################################


class PrefactorTest(unittest.TestCase):

  def test_miyasawa_is_sigma_squared_over_alpha(self):
    alpha = jnp.asarray(0.5)
    sigma = jnp.asarray(2.0)
    p = miyasawa_prefactor(alpha=alpha, sigma=sigma, xt=None, x0=None)
    self.assertAlmostEqual(float(p), 4.0 / 0.5, places=12)

  def test_dps_is_inverse_residual_norm(self):
    alpha = jnp.asarray(1.0)
    sigma = jnp.asarray(1.0)
    # residual = x0 - xt/alpha = x0. Per-row norms: [5, 1].
    xt = jnp.zeros((2, 4), dtype=jnp.float64)
    x0 = jnp.asarray(
        [[3.0, 4.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=jnp.float64,
    )
    p = dps_prefactor(alpha=alpha, sigma=sigma, xt=xt, x0=x0)
    self.assertTrue(jnp.allclose(
        p.squeeze(-1), jnp.asarray([1 / 5.0, 1.0]), atol=1e-6,
    ))


################################################################################
# Gaussian twist
################################################################################


class GaussianTwistTest(unittest.TestCase):

  def test_log_psi_is_zero_at_matching_observation(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    fwd = _MeanForwardFn()
    batch, n = 3, 8
    x0 = jax.random.normal(jax.random.PRNGKey(0), (batch, n), dtype=jnp.float64)
    y = fwd.forward(x0)

    twist = GaussianLikelihoodTwistFn(
        observation=y, forward_fn=fwd, observation_noise=1.0,
    )
    log_psi = twist(
        xt=jnp.zeros_like(x0),
        time=jnp.asarray([0.5], dtype=jnp.float64),
        inference_fn=_identity_x0_inference_fn(x0),
        schedule=schedule, corruption_process=corruption,
    )
    self.assertTrue(jnp.allclose(log_psi, jnp.zeros(batch), atol=1e-12))

  def test_log_psi_decreases_with_residual(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    fwd = _MeanForwardFn()
    batch, n = 2, 4
    x0 = jax.random.normal(jax.random.PRNGKey(1), (batch, n), dtype=jnp.float64)
    y_true = fwd.forward(x0)
    y_off = y_true + 1.0
    infer = _identity_x0_inference_fn(x0)

    t_on = GaussianLikelihoodTwistFn(
        observation=y_true, forward_fn=fwd, observation_noise=1.0,
    )
    t_off = GaussianLikelihoodTwistFn(
        observation=y_off, forward_fn=fwd, observation_noise=1.0,
    )
    t = jnp.asarray([0.5], dtype=jnp.float64)
    kwargs = dict(
        xt=jnp.zeros_like(x0), time=t, inference_fn=infer,
        schedule=schedule, corruption_process=corruption,
    )
    self.assertTrue(bool(jnp.all(t_on(**kwargs) > t_off(**kwargs))))


################################################################################
# Corrections
################################################################################


class GradientCorrectionTest(unittest.TestCase):
  """A Gaussian twist + Miyasawa prefactor should shift x0 toward y."""

  def test_delta_x0_reduces_residual(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    fwd = _MeanForwardFn()
    batch, n = 1, 4
    y = jnp.asarray([[1.0]], dtype=jnp.float64)  # target observation

    # Denoiser emits a fixed x0 that depends on xt (scalar mean = 0).
    def inference_fn(xt, time, conditioning=None):
      del time, conditioning
      return {"x0": xt}

    twist = GaussianLikelihoodTwistFn(
        observation=y, forward_fn=fwd, observation_noise=1.0,
    )
    correction = GradientCorrectionFn(
        twist=twist, strength=1.0, prefactor_fn=miyasawa_prefactor,
    )

    xt = jnp.zeros((batch, n), dtype=jnp.float64)
    time = jnp.asarray([0.5], dtype=jnp.float64)
    outputs = {"x0": jnp.zeros_like(xt)}
    corrected = correction(
        outputs, xt, time,
        schedule=schedule, corruption_process=corruption,
        inference_fn=inference_fn,
    )
    x0_corrected = corruption.convert_predictions(corrected, xt, time)["x0"]
    # Moved toward y: the per-site mean should have increased.
    self.assertGreater(
        float(jnp.mean(x0_corrected)), float(jnp.mean(xt)),
    )


class IteratedCorrectionTest(unittest.TestCase):
  """IteratedCorrectionFn must invoke its base exactly ``num_iters`` times."""

  def test_base_called_num_iters_times(self):
    call_count = {"n": 0}

    @dataclasses.dataclass(kw_only=True, frozen=True)
    class _CountingCorrection:
      def __call__(self, outputs, xt, time, *, schedule, corruption_process,
                   conditioning=None, rng=None):
        call_count["n"] += 1
        # Identity: no shift so xt stays put.
        return outputs

    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)

    def inference_fn(xt, time, conditioning=None):
      del conditioning
      return {"x0": xt * 0.0}

    iterated = IteratedCorrectionFn(base=_CountingCorrection(), num_iters=4)
    xt = jnp.zeros((1, 4), dtype=jnp.float64)
    time = jnp.asarray([0.5], dtype=jnp.float64)
    iterated(
        {"x0": jnp.zeros_like(xt)}, xt, time,
        schedule=schedule, corruption_process=corruption,
        inference_fn=inference_fn,
    )
    self.assertEqual(call_count["n"], 4)


################################################################################
# DDIM proposal ratio + registry
################################################################################


class DDIMProposalRatioTest(unittest.TestCase):

  def test_deterministic_eta_returns_zero(self):
    _, corruption, _ = _gaussian_pieces(eta=0.0)
    stepper = DDIMStep(corruption_process=corruption, stoch_coeff=0.0)
    xt = jnp.ones((3, 8), dtype=jnp.float64)
    ratio = ddim_proposal_log_ratio(
        stepper=stepper, corruption_process=corruption,
        outputs_uncorrected={"x0": jnp.zeros_like(xt)},
        outputs_corrected={"x0": jnp.ones_like(xt)},
        xt_prev=xt, xt_next=xt + 0.1,
        time_prev=jnp.asarray([0.5]),
        time_next=jnp.asarray([0.4]),
    )
    self.assertTrue(bool(jnp.all(ratio == 0.0)))

  def test_identity_correction_returns_zero(self):
    _, corruption, _ = _gaussian_pieces(eta=0.5)
    stepper = DDIMStep(corruption_process=corruption, stoch_coeff=0.5)
    xt = jnp.ones((2, 4), dtype=jnp.float64)
    x0 = jnp.zeros_like(xt)
    ratio = ddim_proposal_log_ratio(
        stepper=stepper, corruption_process=corruption,
        outputs_uncorrected={"x0": x0},
        outputs_corrected={"x0": x0},
        xt_prev=xt, xt_next=xt + 0.1,
        time_prev=jnp.asarray([0.5]),
        time_next=jnp.asarray([0.4]),
    )
    self.assertTrue(jnp.allclose(ratio, jnp.zeros(2), atol=1e-12))

  def test_stochastic_ratio_is_finite_and_nonzero(self):
    _, corruption, _ = _gaussian_pieces(eta=0.5)
    stepper = DDIMStep(corruption_process=corruption, stoch_coeff=0.5)
    xt = jnp.ones((2, 4), dtype=jnp.float64)
    ratio = ddim_proposal_log_ratio(
        stepper=stepper, corruption_process=corruption,
        outputs_uncorrected={"x0": jnp.zeros_like(xt)},
        outputs_corrected={"x0": jnp.full_like(xt, 0.3)},
        xt_prev=xt, xt_next=jnp.full_like(xt, 0.2),
        time_prev=jnp.asarray([0.5]),
        time_next=jnp.asarray([0.4]),
    )
    self.assertTrue(bool(jnp.all(jnp.isfinite(ratio))))
    self.assertTrue(bool(jnp.all(ratio != 0.0)))


class RegistryTest(unittest.TestCase):

  def test_identity_short_circuits_before_dispatch(self):
    # Any fake stepper works because correction_identity=True short-circuits.
    class _FakeStepper: pass
    xt = jnp.zeros((2, 4), dtype=jnp.float64)
    ratio = proposal_log_ratio(
        stepper=_FakeStepper(), corruption_process=None,
        outputs_uncorrected={}, outputs_corrected={},
        xt_prev=xt, xt_next=xt,
        time_prev=jnp.asarray([0.5]), time_next=jnp.asarray([0.4]),
        correction_identity=True,
    )
    self.assertTrue(jnp.allclose(ratio, jnp.zeros(2), atol=1e-12))

  def test_unknown_stepper_raises(self):
    class _Unknown: pass
    with self.assertRaises(NotImplementedError):
      proposal_log_ratio(
          stepper=_Unknown(), corruption_process=None,
          outputs_uncorrected={}, outputs_corrected={},
          xt_prev=jnp.zeros((1, 2)), xt_next=jnp.zeros((1, 2)),
          time_prev=jnp.zeros((1,)), time_next=jnp.zeros((1,)),
          correction_identity=False,
      )

  def test_register_dispatches_to_new_stepper(self):
    class _Mock: pass
    called = {"yes": False}

    def fake_ratio(*, stepper, corruption_process, outputs_uncorrected,
                   outputs_corrected, xt_prev, xt_next, time_prev, time_next):
      called["yes"] = True
      return jnp.zeros(xt_prev.shape[0])

    register_proposal_ratio(_Mock, fake_ratio)
    proposal_log_ratio(
        stepper=_Mock(), corruption_process=None,
        outputs_uncorrected={}, outputs_corrected={},
        xt_prev=jnp.zeros((2, 1)), xt_next=jnp.zeros((2, 1)),
        time_prev=jnp.zeros((1,)), time_next=jnp.zeros((1,)),
        correction_identity=False,
    )
    self.assertTrue(called["yes"])


################################################################################
# Sampler backward-compat
################################################################################


class SamplerBackwardCompatTest(unittest.TestCase):
  """``ConditionalDiffusionSampler`` with no correction / twist / resampler
  must delegate to the base sampler bit-for-bit."""

  def test_identity_config_delegates_to_base(self):
    schedule, corruption, base_sampler = _gaussian_pieces(eta=0.0, num_steps=8)
    cond = ConditionalDiffusionSampler(
        base_sampler=base_sampler, corruption_process=corruption,
    )

    @jax.jit
    def inference_fn(xt, time, conditioning=None):
      del conditioning
      t = jnp.atleast_1d(time).reshape(-1)[0:1]
      alpha = schedule.alpha(t).reshape(())
      sigma = schedule.sigma(t).reshape(())
      gain = alpha / (alpha ** 2 + sigma ** 2)
      return {"x0": gain * xt}

    rng = jax.random.PRNGKey(0)
    init = jax.random.normal(rng, (4, 8), dtype=jnp.float64)
    base_out = base_sampler(inference_fn=inference_fn, rng=rng, initial_noise=init)
    cond_out = cond(inference_fn=inference_fn, rng=rng, initial_noise=init)
    final_base = base_out[0] if isinstance(base_out, tuple) else base_out
    final_cond = cond_out[0] if isinstance(cond_out, tuple) else cond_out
    self.assertTrue(jnp.allclose(final_base.xt, final_cond.xt, atol=1e-12))


################################################################################
# Linear-algebra helpers
################################################################################


class LinAlgTest(unittest.TestCase):

  def test_batch_inner_matches_manual(self):
    x = jnp.arange(12.0).reshape(3, 4)
    y = jnp.ones((3, 4))
    out = batch_inner(x, y)
    self.assertTrue(jnp.allclose(
        out, jnp.asarray([6.0, 22.0, 38.0]), atol=1e-12,
    ))

  def test_batched_cg_solves_diagonal_system(self):
    # M = diag(d_i) per particle (d_i vary across batch).  Residual = d * x_true
    # so the solution must return x_true.
    d = jnp.asarray([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]])
    x_true = jnp.asarray([[1.0, 2.0, 3.0], [4.0, -1.0, 2.0]])
    residual = d * x_true
    def matvec(p):
      return d * p
    x = batched_cg(matvec, residual, max_iter=20, tol=1e-10)
    self.assertTrue(jnp.allclose(x, x_true, atol=1e-8))

  def test_batched_cg_dense_random_psd(self):
    rng = jax.random.PRNGKey(0)
    B, n = 3, 6
    # Each particle has its own random PSD matrix M_i.
    A = jax.random.normal(rng, (B, n, n), dtype=jnp.float64)
    M = jnp.einsum("bij,bkj->bik", A, A) + jnp.eye(n, dtype=jnp.float64)[None]
    x_true = jax.random.normal(jax.random.PRNGKey(1), (B, n), dtype=jnp.float64)
    residual = jnp.einsum("bij,bj->bi", M, x_true)
    def matvec(p):
      return jnp.einsum("bij,bj->bi", M, p)
    x = batched_cg(matvec, residual, max_iter=50, tol=1e-10)
    self.assertTrue(jnp.allclose(x, x_true, atol=1e-6))

  def test_linear_adjoint_matches_matrix_transpose(self):
    # forward = linear map x -> x @ W.T, adjoint should be w -> w @ W.
    rng = jax.random.PRNGKey(0)
    n, m = 8, 3
    W = jax.random.normal(rng, (m, n), dtype=jnp.float64)

    @dataclasses.dataclass(kw_only=True, frozen=True)
    class _LinFwd:
      def forward(self, x):
        return x @ W.T

    x = jax.random.normal(jax.random.PRNGKey(1), (4, n), dtype=jnp.float64)
    adj = linear_adjoint(_LinFwd(), x)
    w = jax.random.normal(jax.random.PRNGKey(2), (4, m), dtype=jnp.float64)
    out = adj(w)
    expected = w @ W  # (4, m) @ (m, n) = (4, n)
    self.assertTrue(jnp.allclose(out, expected, atol=1e-12))


################################################################################
# PosteriorCovarianceFn variants
################################################################################


class IsotropicPosteriorCovarianceTest(unittest.TestCase):

  def test_default_scale_is_miyasawa(self):
    schedule = schedules.CosineSchedule()
    op = IsotropicPosteriorCovarianceFn()
    v = jnp.ones((2, 4), dtype=jnp.float64)
    t = jnp.asarray([0.3], dtype=jnp.float64)
    out = op(v, xt=v, time=t, schedule=schedule)
    alpha, sigma = scalar_alpha_sigma(schedule, t)
    self.assertTrue(jnp.allclose(
        out, (sigma ** 2 / alpha) * v, atol=1e-12,
    ))

  def test_custom_scale_fn(self):
    schedule = schedules.CosineSchedule()
    op = IsotropicPosteriorCovarianceFn(
        scale_fn=lambda alpha, sigma: jnp.asarray(3.0, dtype=alpha.dtype),
    )
    v = jnp.ones((2, 4), dtype=jnp.float64)
    out = op(v, xt=v, time=jnp.asarray([0.5]), schedule=schedule)
    self.assertTrue(jnp.allclose(out, 3.0 * v, atol=1e-12))


class FixedPriorPosteriorCovarianceTest(unittest.TestCase):

  def test_dense_matrix_application(self):
    schedule = schedules.CosineSchedule()
    C = jnp.asarray([[2.0, 0.5], [0.5, 3.0]], dtype=jnp.float64)
    op = FixedPriorPosteriorCovarianceFn(prior_covariance=C)
    v = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64)
    t = jnp.asarray([0.5], dtype=jnp.float64)
    out = op(v, xt=v, time=t, schedule=schedule)
    alpha, sigma = scalar_alpha_sigma(schedule, t)
    expected = (sigma ** 2 / alpha) * (v @ C.T)
    self.assertTrue(jnp.allclose(out, expected, atol=1e-12))

  def test_apply_fn_matches_dense(self):
    schedule = schedules.CosineSchedule()
    C = jnp.asarray([[2.0, 0.5], [0.5, 3.0]], dtype=jnp.float64)
    op_dense = FixedPriorPosteriorCovarianceFn(prior_covariance=C)
    op_apply = FixedPriorPosteriorCovarianceFn(
        apply_fn=lambda v_flat: v_flat @ C.T,
    )
    v = jnp.asarray([[1.0, -1.0]], dtype=jnp.float64)
    t = jnp.asarray([0.5], dtype=jnp.float64)
    a = op_dense(v, xt=v, time=t, schedule=schedule)
    b = op_apply(v, xt=v, time=t, schedule=schedule)
    self.assertTrue(jnp.allclose(a, b, atol=1e-12))

  def test_requires_exactly_one_operator(self):
    with self.assertRaises(ValueError):
      FixedPriorPosteriorCovarianceFn()
    with self.assertRaises(ValueError):
      FixedPriorPosteriorCovarianceFn(
          prior_covariance=jnp.eye(2), apply_fn=lambda v: v,
      )


class TweediePosteriorCovarianceTest(unittest.TestCase):

  def test_linear_denoiser_recovers_scaled_gain(self):
    # If denoiser_fn(xt) = G xt for a fixed matrix G, then JVP(denoiser, xt, v)
    # = G v and the operator returns (sigma^2/alpha) * G v.
    schedule = schedules.CosineSchedule()
    rng = jax.random.PRNGKey(0)
    n = 4
    G = jax.random.normal(rng, (n, n), dtype=jnp.float64)
    def denoiser(x):
      return x @ G.T

    op = TweediePosteriorCovarianceFn()
    v = jnp.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float64)
    t = jnp.asarray([0.4], dtype=jnp.float64)
    out = op(v, xt=v, time=t, schedule=schedule, denoiser_fn=denoiser)
    alpha, sigma = scalar_alpha_sigma(schedule, t)
    expected = (sigma ** 2 / alpha) * (v @ G.T)
    self.assertTrue(jnp.allclose(out, expected, atol=1e-10))

  def test_missing_denoiser_raises(self):
    schedule = schedules.CosineSchedule()
    op = TweediePosteriorCovarianceFn()
    v = jnp.ones((1, 2))
    with self.assertRaises(ValueError):
      op(v, xt=v, time=jnp.asarray([0.5]), schedule=schedule)


################################################################################
# PiGDM correction
################################################################################


class PiGDMCorrectionTest(unittest.TestCase):
  """End-to-end Kalman update sanity checks on a linear-Gaussian setup."""

  def _setup(self, *, n=4, batch=2):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    fwd = _MeanForwardFn()  # maps (B, n) -> (B, 1)
    # Arbitrary fixed x0 and target observation.
    x0 = jax.random.normal(jax.random.PRNGKey(0), (batch, n), dtype=jnp.float64)
    y_target = fwd.forward(x0) + 1.0  # shift target by 1 to create a residual
    return schedule, corruption, fwd, x0, y_target

  def test_isotropic_pigdm_reduces_residual(self):
    schedule, corruption, fwd, x0, y = self._setup()

    def inference_fn(xt, time, conditioning=None):
      del xt, time, conditioning
      return {"x0": x0}

    correction = PiGDMCorrectionFn(
        observation=y, forward_fn=fwd,
        posterior_covariance_fn=IsotropicPosteriorCovarianceFn(),
        observation_noise=0.1,
    )
    xt = jnp.zeros_like(x0)
    time = jnp.asarray([0.5], dtype=jnp.float64)
    corrected = correction(
        {"x0": x0}, xt, time,
        schedule=schedule, corruption_process=corruption,
        inference_fn=inference_fn,
    )
    x0_new = corruption.convert_predictions(corrected, xt, time)["x0"]
    res_old = jnp.abs(fwd.forward(x0) - y)
    res_new = jnp.abs(fwd.forward(x0_new) - y)
    self.assertTrue(bool(jnp.all(res_new < res_old)))

  def test_tweedie_pigdm_runs_and_reduces_residual(self):
    schedule, corruption, fwd, x0, y = self._setup()

    # Non-trivial denoiser: xhat_0 = tanh(xt).  Jacobian = diag(1 - tanh^2).
    def inference_fn(xt, time, conditioning=None):
      del time, conditioning
      return {"x0": jnp.tanh(xt)}

    correction = PiGDMCorrectionFn(
        observation=y, forward_fn=fwd,
        posterior_covariance_fn=TweediePosteriorCovarianceFn(),
        observation_noise=0.1, cg_max_iter=25, cg_tol=1e-8,
    )
    xt = jnp.full_like(x0, 0.3)  # tanh(xt) is non-identity
    time = jnp.asarray([0.5], dtype=jnp.float64)
    # The denoiser's output at this xt is what x0 is set to.
    x0_at_xt = jnp.tanh(xt)
    corrected = correction(
        {"x0": x0_at_xt}, xt, time,
        schedule=schedule, corruption_process=corruption,
        inference_fn=inference_fn,
    )
    x0_new = corruption.convert_predictions(corrected, xt, time)["x0"]
    res_old = jnp.abs(fwd.forward(x0_at_xt) - y)
    res_new = jnp.abs(fwd.forward(x0_new) - y)
    self.assertTrue(bool(jnp.all(res_new < res_old)))

  def test_zero_residual_gives_zero_correction(self):
    # If observation matches A x0 already, Kalman update is zero.
    schedule, corruption, fwd, x0, _ = self._setup()
    y_match = fwd.forward(x0)  # consistent observation

    def inference_fn(xt, time, conditioning=None):
      del xt, time, conditioning
      return {"x0": x0}

    correction = PiGDMCorrectionFn(
        observation=y_match, forward_fn=fwd,
        posterior_covariance_fn=IsotropicPosteriorCovarianceFn(),
        observation_noise=0.1,
    )
    xt = jnp.zeros_like(x0)
    time = jnp.asarray([0.5], dtype=jnp.float64)
    corrected = correction(
        {"x0": x0}, xt, time,
        schedule=schedule, corruption_process=corruption,
        inference_fn=inference_fn,
    )
    x0_new = corruption.convert_predictions(corrected, xt, time)["x0"]
    self.assertTrue(jnp.allclose(x0_new, x0, atol=1e-10))


################################################################################
# Forward operators
################################################################################


class ForwardOpsTest(unittest.TestCase):

  def test_linear_dense_matches_matmul(self):
    W = jnp.asarray([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]], dtype=jnp.float64)
    fwd = LinearForwardFn(matrix=W)
    x = jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float64)
    out = fwd.forward(x)
    self.assertTrue(jnp.allclose(out, x @ W.T, atol=1e-12))

  def test_linear_apply_fn_matches_matrix(self):
    W = jnp.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=jnp.float64)
    fwd_dense = LinearForwardFn(matrix=W)
    fwd_apply = LinearForwardFn(apply_fn=lambda x: x @ W.T)
    x = jnp.asarray([[1.0, 1.0]], dtype=jnp.float64)
    self.assertTrue(jnp.allclose(fwd_dense.forward(x), fwd_apply.forward(x)))

  def test_linear_requires_exactly_one_operator(self):
    with self.assertRaises(ValueError):
      LinearForwardFn()
    with self.assertRaises(ValueError):
      LinearForwardFn(matrix=jnp.eye(2), apply_fn=lambda x: x)

  def test_subsample_selects_indices(self):
    fwd = SubsampleForwardFn(indices=jnp.asarray([0, 2, 4]))
    x = jnp.arange(12.0).reshape(2, 6)
    out = fwd.forward(x)
    self.assertEqual(out.shape, (2, 3))
    self.assertTrue(jnp.allclose(out, x[:, [0, 2, 4]]))

  def test_inpainting_masks_entries(self):
    mask = jnp.asarray([1.0, 0.0, 1.0, 0.0], dtype=jnp.float64)
    fwd = InpaintingForwardFn(mask=mask)
    x = jnp.asarray([[10.0, 20.0, 30.0, 40.0]], dtype=jnp.float64)
    out = fwd.forward(x)
    self.assertTrue(jnp.allclose(out, jnp.asarray([[10.0, 0.0, 30.0, 0.0]])))

  def test_conv_identity_kernel_is_identity(self):
    # 1-D NCL conv with a length-1 kernel = 1.0 is the identity.
    kernel = jnp.ones((1, 1, 1), dtype=jnp.float64)  # (out_ch, in_ch, len)
    fwd = ConvForwardFn(kernel=kernel, stride=(1,), padding="VALID")
    x = jnp.arange(10.0).reshape(1, 1, 10)
    out = fwd.forward(x)
    self.assertTrue(jnp.allclose(out, x, atol=1e-12))

  def test_compose_is_function_composition(self):
    W1 = jnp.asarray([[1.0, 2.0], [0.0, 1.0]], dtype=jnp.float64)
    W2 = jnp.asarray([[0.5, 0.0], [0.0, 0.5]], dtype=jnp.float64)
    fwd = ComposeForwardFn(
        first=LinearForwardFn(matrix=W1),
        second=LinearForwardFn(matrix=W2),
    )
    x = jnp.asarray([[1.0, 1.0]], dtype=jnp.float64)
    self.assertTrue(jnp.allclose(fwd.forward(x), (x @ W1.T) @ W2.T))


################################################################################
# Classifier / Energy twists
################################################################################


class ClassifierEnergyTwistTest(unittest.TestCase):

  def test_classifier_twist_returns_log_prob_of_denoiser_x0(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    batch, n = 2, 4
    x0 = jax.random.normal(jax.random.PRNGKey(0), (batch, n), dtype=jnp.float64)

    # log_prob_fn rewards matching a fixed target.
    target = jnp.zeros_like(x0)
    def log_prob(x):
      return -0.5 * jnp.sum((x - target) ** 2, axis=-1)

    twist = ClassifierTwistFn(log_prob_fn=log_prob)
    log_psi = twist(
        xt=jnp.zeros_like(x0),
        time=jnp.asarray([0.5], dtype=jnp.float64),
        inference_fn=_identity_x0_inference_fn(x0),
        schedule=schedule, corruption_process=corruption,
    )
    expected = log_prob(x0)
    self.assertTrue(jnp.allclose(log_psi, expected, atol=1e-12))

  def test_energy_twist_is_negative_energy_over_temperature(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    batch, n = 3, 5
    x0 = jax.random.normal(jax.random.PRNGKey(1), (batch, n), dtype=jnp.float64)

    def energy(x):
      return jnp.sum(x ** 2, axis=-1)

    twist = EnergyTwistFn(energy_fn=energy, temperature=2.0)
    log_psi = twist(
        xt=jnp.zeros_like(x0),
        time=jnp.asarray([0.5], dtype=jnp.float64),
        inference_fn=_identity_x0_inference_fn(x0),
        schedule=schedule, corruption_process=corruption,
    )
    expected = -energy(x0) / 2.0
    self.assertTrue(jnp.allclose(log_psi, expected, atol=1e-12))


################################################################################
# SDE proposal ratio
################################################################################


class SdeProposalRatioTest(unittest.TestCase):

  def test_deterministic_churn_returns_zero(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    stepper = SdeStep(corruption_process=corruption, churn=0.0)
    xt = jnp.ones((3, 8), dtype=jnp.float64)
    ratio = sde_proposal_log_ratio(
        stepper=stepper, corruption_process=corruption,
        outputs_uncorrected={"x0": jnp.zeros_like(xt)},
        outputs_corrected={"x0": jnp.ones_like(xt)},
        xt_prev=xt, xt_next=xt + 0.1,
        time_prev=jnp.asarray([0.5]),
        time_next=jnp.asarray([0.4]),
    )
    self.assertTrue(bool(jnp.all(ratio == 0.0)))

  def test_stochastic_churn_finite_and_nonzero(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    stepper = SdeStep(corruption_process=corruption, churn=1.0)
    xt = jnp.ones((2, 4), dtype=jnp.float64)
    ratio = sde_proposal_log_ratio(
        stepper=stepper, corruption_process=corruption,
        outputs_uncorrected={"x0": jnp.zeros_like(xt)},
        outputs_corrected={"x0": jnp.full_like(xt, 0.3)},
        xt_prev=xt, xt_next=jnp.full_like(xt, 0.2),
        time_prev=jnp.asarray([0.5]),
        time_next=jnp.asarray([0.4]),
    )
    self.assertTrue(bool(jnp.all(jnp.isfinite(ratio))))
    self.assertTrue(bool(jnp.all(ratio != 0.0)))

  def test_registry_dispatches_sde_step(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    stepper = SdeStep(corruption_process=corruption, churn=0.5)
    xt = jnp.ones((2, 4), dtype=jnp.float64)
    ratio = proposal_log_ratio(
        stepper=stepper, corruption_process=corruption,
        outputs_uncorrected={"x0": jnp.zeros_like(xt)},
        outputs_corrected={"x0": jnp.full_like(xt, 0.2)},
        xt_prev=xt, xt_next=jnp.full_like(xt, 0.5),
        time_prev=jnp.asarray([0.5]),
        time_next=jnp.asarray([0.4]),
        correction_identity=False,
    )
    self.assertTrue(bool(jnp.all(jnp.isfinite(ratio))))


################################################################################
# Additional stepper ratios: AdjustedDDIM / Velocity / Heun
################################################################################


class DeterministicStepperRatioTest(unittest.TestCase):
  """AdjustedDDIMStep and HeunStep are deterministic; ratio must be 0."""

  def test_adjusted_ddim_ratio_is_zero_via_registry(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    stepper = AdjustedDDIMStep(corruption_process=corruption)
    xt = jnp.ones((3, 8), dtype=jnp.float64)
    ratio = proposal_log_ratio(
        stepper=stepper, corruption_process=corruption,
        outputs_uncorrected={"x0": jnp.zeros_like(xt)},
        outputs_corrected={"x0": jnp.ones_like(xt)},
        xt_prev=xt, xt_next=xt + 0.1,
        time_prev=jnp.asarray([0.5]), time_next=jnp.asarray([0.4]),
        correction_identity=False,
    )
    self.assertTrue(bool(jnp.all(ratio == 0.0)))


class VelocityProposalRatioTest(unittest.TestCase):

  def test_deterministic_epsilon_returns_zero(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    stepper = VelocityStep(corruption_process=corruption, epsilon=0.0)
    xt = jnp.ones((3, 8), dtype=jnp.float64)
    ratio = velocity_proposal_log_ratio(
        stepper=stepper, corruption_process=corruption,
        outputs_uncorrected={"x0": jnp.zeros_like(xt)},
        outputs_corrected={"x0": jnp.ones_like(xt)},
        xt_prev=xt, xt_next=xt + 0.1,
        time_prev=jnp.asarray([0.5]), time_next=jnp.asarray([0.4]),
    )
    self.assertTrue(bool(jnp.all(ratio == 0.0)))

  def test_stochastic_epsilon_finite_and_nonzero(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    stepper = VelocityStep(corruption_process=corruption, epsilon=0.5)
    xt = jnp.ones((2, 4), dtype=jnp.float64)
    ratio = velocity_proposal_log_ratio(
        stepper=stepper, corruption_process=corruption,
        outputs_uncorrected={"x0": jnp.zeros_like(xt)},
        outputs_corrected={"x0": jnp.full_like(xt, 0.3)},
        xt_prev=xt, xt_next=jnp.full_like(xt, 0.2),
        time_prev=jnp.asarray([0.5]), time_next=jnp.asarray([0.4]),
    )
    self.assertTrue(bool(jnp.all(jnp.isfinite(ratio))))
    self.assertTrue(bool(jnp.all(ratio != 0.0)))


################################################################################
# Classifier-free guidance correction adapter
################################################################################


class CFGCorrectionTest(unittest.TestCase):
  """CFGCorrectionFn blends conditional + unconditional outputs."""

  def test_scalar_cfg_reproduces_linear_blend(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    batch, n = 2, 4
    x0_cond = jax.random.normal(
        jax.random.PRNGKey(0), (batch, n), dtype=jnp.float64,
    )
    x0_uncond = jax.random.normal(
        jax.random.PRNGKey(1), (batch, n), dtype=jnp.float64,
    )

    # Conditional branch: we hand it ``outputs`` directly.
    # Unconditional branch: called through CFGCorrectionFn.
    def uncond_fn(xt, time, conditioning=None):
      del xt, time, conditioning
      return {"x0": x0_uncond}

    w = 1.5
    correction = CFGCorrectionFn(
        unconditional_inference_fn=uncond_fn,
        guidance_fn=ScalarGuidanceFn(guidance=w),
    )

    xt = jnp.zeros_like(x0_cond)
    time = jnp.asarray([0.5], dtype=jnp.float64)
    out = correction(
        {"x0": x0_cond}, xt, time,
        schedule=schedule, corruption_process=corruption,
        conditioning=None,
    )
    # ScalarGuidanceFn blends as ``cond * (1+w) - uncond * w`` per tree leaf.
    expected = x0_cond * (1.0 + w) - x0_uncond * w
    self.assertTrue(jnp.allclose(out["x0"], expected, atol=1e-12))


if __name__ == "__main__":
  unittest.main()
