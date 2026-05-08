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

from hackable_diffusion.lib.sampling.sampling import DiffusionSampler
from hackable_diffusion.lib.sampling.time_scheduling import UniformTimeSchedule

from hackable_diffusion.lib.guidance.corrections import (
    GradientCorrectionFn,
    IteratedCorrectionFn,
    KalmanCorrectionFn,
    dps_prefactor,
    miyasawa_prefactor,
)
from hackable_diffusion.lib.guidance.denoisers import (
    LinearBlendDenoiserFn,
    make_cfg_inference_fn,
    make_denoiser_fn,
    make_posterior_cloud_fn,
)
from hackable_diffusion.lib.guidance.forward_ops import (
    ComposeForwardFn,
    ConvForwardFn,
    InpaintingForwardFn,
    LinearForwardFn,
    SubsampleForwardFn,
)
from hackable_diffusion.lib.guidance.gaussian_conditioning import (
    PosteriorPredictiveGaussianTwistFn,
    PseudoInverseKalmanCorrectionFn,
)
from hackable_diffusion.lib.guidance.linalg import (
    batch_inner,
    batched_cg,
    batched_minres,
    linear_adjoint,
)
from hackable_diffusion.lib.guidance.posterior_covariance import (
    FixedPriorPosteriorCovarianceFn,
    IsotropicPosteriorCovarianceFn,
    LowRankTweediePosteriorCovarianceFn,
    PCAPosteriorCovarianceFn,
    TweediePosteriorCovarianceFn,
    miyasawa_scale,
)
from hackable_diffusion.lib.guidance.proposal_ratio import proposal_log_ratio
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


@dataclasses.dataclass(kw_only=True, frozen=True)
class _FixedPosteriorCovarianceFn:
  covariance: jax.Array

  def __call__(self, *, xt, time, schedule, denoiser_fn=None):
    del time, schedule, denoiser_fn
    cov = self.covariance.astype(xt.dtype)

    def matvec(v: jax.Array) -> jax.Array:
      v_flat = v.reshape(v.shape[0], -1)
      return (v_flat @ cov.T).reshape(v.shape)

    return matvec


def _identity_x0_inference_fn(x0_fixed: jax.Array):
  """Return a deterministic inference fn that always emits ``x0_fixed``."""

  def fn(xt, time, conditioning=None):
    del xt, time, conditioning
    return {"x0": x0_fixed}

  return fn


def _denoiser_from_x0(x0_fixed: jax.Array, corruption, time):
  """Test fixture: build a DenoiserFn that returns a fixed x0."""
  return make_denoiser_fn(
      _identity_x0_inference_fn(x0_fixed), corruption, time=time,
  )


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
    time = jnp.asarray([0.5], dtype=jnp.float64)

    twist = GaussianLikelihoodTwistFn(
        observation=y, forward_fn=fwd, observation_noise=1.0,
    )
    log_psi = twist(
        jnp.zeros_like(x0), time,
        denoiser_fn=_denoiser_from_x0(x0, corruption, time),
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
    time = jnp.asarray([0.5], dtype=jnp.float64)
    denoiser_fn = _denoiser_from_x0(x0, corruption, time)

    t_on = GaussianLikelihoodTwistFn(
        observation=y_true, forward_fn=fwd, observation_noise=1.0,
    )
    t_off = GaussianLikelihoodTwistFn(
        observation=y_off, forward_fn=fwd, observation_noise=1.0,
    )
    xt = jnp.zeros_like(x0)
    self.assertTrue(bool(jnp.all(
        t_on(xt, time, denoiser_fn=denoiser_fn)
        > t_off(xt, time, denoiser_fn=denoiser_fn),
    )))


class PosteriorPredictiveGaussianTwistTest(unittest.TestCase):

  def test_singular_log_density_uses_predictive_covariance(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    # Duplicate observation rows make A C A^T singular but consistent.
    A = jnp.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64)
    fwd = LinearForwardFn(matrix=A)
    cov = jnp.diag(jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64))
    x0 = jnp.asarray([[0.25, -1.0, 2.0]], dtype=jnp.float64)
    y = jnp.asarray([[1.0, 1.0]], dtype=jnp.float64)
    time = jnp.asarray([0.5], dtype=jnp.float64)

    twist = PosteriorPredictiveGaussianTwistFn(
        observation=y,
        forward_fn=fwd,
        posterior_covariance_fn=_FixedPosteriorCovarianceFn(covariance=cov),
        schedule=schedule,
        observation_noise=0.0,
        pinv_rtol=1e-12,
        pinv_atol=1e-12,
    )
    logp = twist(
        jnp.zeros_like(x0),
        time,
        denoiser_fn=_denoiser_from_x0(x0, corruption, time),
    )

    residual = 0.75
    expected = -0.5 * (
        residual ** 2 + jnp.log(2.0) + jnp.log(2.0 * jnp.pi)
    )
    self.assertTrue(jnp.allclose(logp, expected[None], atol=1e-10))


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

    # Denoiser returns xt itself -- differentiable through the gradient step.
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
    denoiser_fn = make_denoiser_fn(inference_fn, corruption, time=time)
    x0_corrected = correction(
        denoiser_fn(xt), xt, time,
        denoiser_fn=denoiser_fn, schedule=schedule,
    )
    # Moved toward y: the per-site mean should have increased.
    self.assertGreater(float(jnp.mean(x0_corrected)), float(jnp.mean(xt)))


class IteratedCorrectionTest(unittest.TestCase):
  """IteratedCorrectionFn must invoke its base exactly ``num_iters`` times."""

  def test_base_called_num_iters_times(self):
    call_count = {"n": 0}

    @dataclasses.dataclass(kw_only=True, frozen=True)
    class _CountingCorrection:
      def __call__(self, x0, xt, time, *, denoiser_fn, schedule):
        del xt, time, denoiser_fn, schedule
        call_count["n"] += 1
        return x0  # identity: no shift

    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)

    def inference_fn(xt, time, conditioning=None):
      del conditioning
      return {"x0": xt * 0.0}

    iterated = IteratedCorrectionFn(base=_CountingCorrection(), num_iters=4)
    xt = jnp.zeros((1, 4), dtype=jnp.float64)
    time = jnp.asarray([0.5], dtype=jnp.float64)
    denoiser_fn = make_denoiser_fn(inference_fn, corruption, time=time)
    iterated(
        denoiser_fn(xt), xt, time,
        denoiser_fn=denoiser_fn, schedule=schedule,
    )
    self.assertEqual(call_count["n"], 4)


################################################################################
# DDIM proposal ratio + registry
################################################################################


def _proposal_log_ratio_for(stepper, corruption_process, x0_unc, x0_cor,
                            xt_prev, xt_next, time_prev, time_next):
  """Test helper: invoke the polymorphic dispatcher with x0 dicts."""
  return proposal_log_ratio(
      stepper=stepper,
      outputs_uncorrected={"x0": x0_unc},
      outputs_corrected={"x0": x0_cor},
      xt_prev=xt_prev, xt_next=xt_next,
      time_prev=time_prev, time_next=time_next,
      correction_identity=False,
  )


class DDIMProposalRatioTest(unittest.TestCase):
  """Polymorphic ratio via ``DDIMStep.kernel``."""

  def test_deterministic_eta_returns_zero(self):
    _, corruption, _ = _gaussian_pieces(eta=0.0)
    stepper = DDIMStep(corruption_process=corruption, stoch_coeff=0.0)
    xt = jnp.ones((3, 8), dtype=jnp.float64)
    ratio = _proposal_log_ratio_for(
        stepper, corruption,
        jnp.zeros_like(xt), jnp.ones_like(xt),
        xt, xt + 0.1,
        jnp.asarray([0.5]), jnp.asarray([0.4]),
    )
    self.assertTrue(bool(jnp.all(ratio == 0.0)))

  def test_identity_correction_returns_zero(self):
    _, corruption, _ = _gaussian_pieces(eta=0.5)
    stepper = DDIMStep(corruption_process=corruption, stoch_coeff=0.5)
    xt = jnp.ones((2, 4), dtype=jnp.float64)
    x0 = jnp.zeros_like(xt)
    ratio = proposal_log_ratio(
        stepper=stepper,
        outputs_uncorrected={"x0": x0},
        outputs_corrected={"x0": x0},
        xt_prev=xt, xt_next=xt + 0.1,
        time_prev=jnp.asarray([0.5]), time_next=jnp.asarray([0.4]),
        correction_identity=True,
    )
    self.assertTrue(jnp.allclose(ratio, jnp.zeros(2), atol=1e-12))

  def test_stochastic_ratio_is_finite_and_nonzero(self):
    _, corruption, _ = _gaussian_pieces(eta=0.5)
    stepper = DDIMStep(corruption_process=corruption, stoch_coeff=0.5)
    xt = jnp.ones((2, 4), dtype=jnp.float64)
    ratio = _proposal_log_ratio_for(
        stepper, corruption,
        jnp.zeros_like(xt), jnp.full_like(xt, 0.3),
        xt, jnp.full_like(xt, 0.2),
        jnp.asarray([0.5]), jnp.asarray([0.4]),
    )
    self.assertTrue(bool(jnp.all(jnp.isfinite(ratio))))
    self.assertTrue(bool(jnp.all(ratio != 0.0)))


class PolymorphicDispatchTest(unittest.TestCase):
  """``proposal_log_ratio`` only asks for ``stepper.kernel``."""

  def test_identity_correction_short_circuits(self):
    # Any fake stepper works because correction_identity=True skips the kernel.
    class _FakeStepper: pass
    xt = jnp.zeros((2, 4), dtype=jnp.float64)
    ratio = proposal_log_ratio(
        stepper=_FakeStepper(),
        outputs_uncorrected={}, outputs_corrected={},
        xt_prev=xt, xt_next=xt,
        time_prev=jnp.asarray([0.5]), time_next=jnp.asarray([0.4]),
        correction_identity=True,
    )
    self.assertTrue(jnp.allclose(ratio, jnp.zeros(2), atol=1e-12))

  def test_stepper_without_kernel_raises(self):
    class _Unkernelled: pass
    with self.assertRaises(AttributeError):
      proposal_log_ratio(
          stepper=_Unkernelled(),
          outputs_uncorrected={}, outputs_corrected={},
          xt_prev=jnp.zeros((1, 2)), xt_next=jnp.zeros((1, 2)),
          time_prev=jnp.zeros((1,)), time_next=jnp.zeros((1,)),
          correction_identity=False,
      )


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

  def test_batched_minres_solves_indefinite_system(self):
    # M has both positive and negative eigenvalues; CG would struggle.
    M = jnp.asarray([
        [[2.0, 0.5, 0.0, 0.1],
         [0.5, -1.0, 0.3, 0.0],
         [0.0, 0.3, 1.5, 0.2],
         [0.1, 0.0, 0.2, 0.8]],
        [[1.0, 0.0, 0.2, 0.0],
         [0.0, 3.0, 0.1, 0.0],
         [0.2, 0.1, -0.5, 0.0],
         [0.0, 0.0, 0.0, 2.0]],
    ], dtype=jnp.float64)
    rhs = jnp.asarray(
        [[1.0, 2.0, -1.0, 0.5], [0.0, 1.0, 0.0, 0.5]],
        dtype=jnp.float64,
    )
    def matvec(v):
      return jnp.einsum('bij,bj->bi', M, v)
    z = batched_minres(matvec, rhs, max_iter=200, tol=1e-12)
    recon = jnp.einsum('bij,bj->bi', M, z)
    self.assertTrue(jnp.allclose(recon, rhs, atol=1e-9))

  def test_batched_minres_matches_cg_on_psd(self):
    # On a PSD matrix MINRES and CG should both converge to the same solution.
    rng = jax.random.PRNGKey(0)
    B, n = 2, 5
    A = jax.random.normal(rng, (B, n, n), dtype=jnp.float64)
    M = jnp.einsum('bij,bkj->bik', A, A) + 0.5 * jnp.eye(n)[None]
    x_true = jax.random.normal(jax.random.PRNGKey(1), (B, n), dtype=jnp.float64)
    rhs = jnp.einsum('bij,bj->bi', M, x_true)
    def matvec(v):
      return jnp.einsum('bij,bj->bi', M, v)
    x_cg = batched_cg(matvec, rhs, max_iter=200, tol=1e-12)
    x_minres = batched_minres(matvec, rhs, max_iter=200, tol=1e-12)
    self.assertTrue(jnp.allclose(x_cg, x_true, atol=1e-8))
    self.assertTrue(jnp.allclose(x_minres, x_true, atol=1e-8))

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
    out = op(xt=v, time=t, schedule=schedule)(v)
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
    out = op(xt=v, time=jnp.asarray([0.5]), schedule=schedule)(v)
    self.assertTrue(jnp.allclose(out, 3.0 * v, atol=1e-12))


class FixedPriorPosteriorCovarianceTest(unittest.TestCase):

  def test_dense_matrix_application(self):
    schedule = schedules.CosineSchedule()
    C = jnp.asarray([[2.0, 0.5], [0.5, 3.0]], dtype=jnp.float64)
    op = FixedPriorPosteriorCovarianceFn(prior_covariance=C)
    v = jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64)
    t = jnp.asarray([0.5], dtype=jnp.float64)
    out = op(xt=v, time=t, schedule=schedule)(v)
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
    a = op_dense(xt=v, time=t, schedule=schedule)(v)
    b = op_apply(xt=v, time=t, schedule=schedule)(v)
    self.assertTrue(jnp.allclose(a, b, atol=1e-12))

  def test_requires_exactly_one_operator(self):
    with self.assertRaises(ValueError):
      FixedPriorPosteriorCovarianceFn()
    with self.assertRaises(ValueError):
      FixedPriorPosteriorCovarianceFn(
          prior_covariance=jnp.eye(2), apply_fn=lambda v: v,
      )


class TweediePosteriorCovarianceTest(unittest.TestCase):

  def test_linear_denoiser_no_symmetrize_recovers_scaled_gain(self):
    # If denoiser_fn(xt) = G xt for a fixed matrix G, then JVP(denoiser, xt, v)
    # = G v and the operator returns (sigma^2/alpha) * G v.
    schedule = schedules.CosineSchedule()
    rng = jax.random.PRNGKey(0)
    n = 4
    G = jax.random.normal(rng, (n, n), dtype=jnp.float64)
    def denoiser(x):
      return x @ G.T

    op = TweediePosteriorCovarianceFn(symmetrize=False)
    v = jnp.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float64)
    t = jnp.asarray([0.4], dtype=jnp.float64)
    out = op(xt=v, time=t, schedule=schedule, denoiser_fn=denoiser)(v)
    alpha, sigma = scalar_alpha_sigma(schedule, t)
    expected = (sigma ** 2 / alpha) * (v @ G.T)
    self.assertTrue(jnp.allclose(out, expected, atol=1e-10))

  def test_linear_denoiser_symmetrize_uses_half_J_plus_JT(self):
    # With symmetrize=True (default), Cov v = (sigma^2/alpha) * (J + J^T)/2 v
    # = (sigma^2/alpha) * (G + G^T)/2 v.  Independent of orientation of v
    # under (G + G^T) symmetric.
    schedule = schedules.CosineSchedule()
    rng = jax.random.PRNGKey(0)
    n = 4
    G = jax.random.normal(rng, (n, n), dtype=jnp.float64)
    G_sym = 0.5 * (G + G.T)
    def denoiser(x):
      return x @ G.T

    op = TweediePosteriorCovarianceFn()  # symmetrize default True
    v = jnp.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=jnp.float64)
    t = jnp.asarray([0.4], dtype=jnp.float64)
    out = op(xt=v, time=t, schedule=schedule, denoiser_fn=denoiser)(v)
    alpha, sigma = scalar_alpha_sigma(schedule, t)
    expected = (sigma ** 2 / alpha) * (v @ G_sym.T)
    self.assertTrue(jnp.allclose(out, expected, atol=1e-10))

  def test_missing_denoiser_raises(self):
    schedule = schedules.CosineSchedule()
    op = TweediePosteriorCovarianceFn()
    v = jnp.ones((1, 2))
    with self.assertRaises(ValueError):
      op(xt=v, time=jnp.asarray([0.5]), schedule=schedule)


class PCAPosteriorCovarianceTest(unittest.TestCase):

  def test_full_rank_recovers_fixed_prior(self):
    # At rank k = d with unit singular values, U U^T = I (if U is any
    # orthonormal basis).  Op matches IsotropicPosteriorCovarianceFn.
    schedule = schedules.CosineSchedule()
    rng = jax.random.PRNGKey(0)
    d = 4
    u, _ = jnp.linalg.qr(jax.random.normal(rng, (d, d), dtype=jnp.float64))
    pca = PCAPosteriorCovarianceFn(u_factor=u)
    iso = IsotropicPosteriorCovarianceFn()
    v = jax.random.normal(jax.random.PRNGKey(1), (2, d), dtype=jnp.float64)
    t = jnp.asarray([0.3], dtype=jnp.float64)
    a = pca(xt=v, time=t, schedule=schedule)(v)
    b = iso(xt=v, time=t, schedule=schedule)(v)
    self.assertTrue(jnp.allclose(a, b, atol=1e-10))

  def test_from_covariance_matches_fixed_prior_at_full_rank(self):
    # Top-d eigendecomposition of C reconstructs C exactly; PCA op
    # should match FixedPriorPosteriorCovarianceFn.
    schedule = schedules.CosineSchedule()
    rng = jax.random.PRNGKey(0)
    d = 5
    a_mat = jax.random.normal(rng, (d, d), dtype=jnp.float64)
    c = a_mat @ a_mat.T + 0.1 * jnp.eye(d, dtype=jnp.float64)
    pca = PCAPosteriorCovarianceFn.from_covariance(c, num_components=d)
    fixed = FixedPriorPosteriorCovarianceFn(prior_covariance=c)
    v = jax.random.normal(jax.random.PRNGKey(2), (3, d), dtype=jnp.float64)
    t = jnp.asarray([0.4], dtype=jnp.float64)
    a = pca(xt=v, time=t, schedule=schedule)(v)
    b = fixed(xt=v, time=t, schedule=schedule)(v)
    self.assertTrue(jnp.allclose(a, b, atol=1e-9))

  def test_truncated_rank_captures_dominant_directions(self):
    # Construct a rank-2 covariance; truncation at k=2 should recover it
    # exactly, while k=1 should miss the smaller mode.
    schedule = schedules.CosineSchedule()
    d = 6
    u_true = jnp.linalg.qr(
        jax.random.normal(jax.random.PRNGKey(0), (d, 2), dtype=jnp.float64),
    )[0]
    lambdas = jnp.asarray([5.0, 1.0], dtype=jnp.float64)
    c = u_true @ jnp.diag(lambdas) @ u_true.T
    pca2 = PCAPosteriorCovarianceFn.from_covariance(c, num_components=2)
    pca1 = PCAPosteriorCovarianceFn.from_covariance(c, num_components=1)
    fixed = FixedPriorPosteriorCovarianceFn(prior_covariance=c)
    v = jax.random.normal(jax.random.PRNGKey(3), (1, d), dtype=jnp.float64)
    t = jnp.asarray([0.5], dtype=jnp.float64)
    full = fixed(xt=v, time=t, schedule=schedule)(v)
    at_2 = pca2(xt=v, time=t, schedule=schedule)(v)
    at_1 = pca1(xt=v, time=t, schedule=schedule)(v)
    self.assertTrue(jnp.allclose(at_2, full, atol=1e-9))
    # k=1 drops the smaller-eigenvalue mode and therefore differs.
    self.assertFalse(jnp.allclose(at_1, full, atol=1e-2))

  def test_image_shape_preserved(self):
    # Shape-agnostic: the factor is flattened against non-batch axes, so
    # the op works on (B, H, W, C) images unchanged.
    schedule = schedules.CosineSchedule()
    rng = jax.random.PRNGKey(0)
    h, w, c = 4, 4, 3
    d = h * w * c
    u, _ = jnp.linalg.qr(
        jax.random.normal(rng, (d, 5), dtype=jnp.float64),
    )
    pca = PCAPosteriorCovarianceFn(
        u_factor=u,
        singular_values=jnp.ones((5,), dtype=jnp.float64),
    )
    v = jax.random.normal(
        jax.random.PRNGKey(1), (2, h, w, c), dtype=jnp.float64,
    )
    out = pca(xt=v, time=jnp.asarray([0.3]), schedule=schedule)(v)
    self.assertEqual(out.shape, v.shape)

  def test_regulariser_adds_isotropic_component(self):
    schedule = schedules.CosineSchedule()
    rng = jax.random.PRNGKey(0)
    d = 4
    u, _ = jnp.linalg.qr(
        jax.random.normal(rng, (d, 2), dtype=jnp.float64),
    )
    pca = PCAPosteriorCovarianceFn(u_factor=u, regulariser=0.5)
    pca_no_reg = PCAPosteriorCovarianceFn(u_factor=u, regulariser=0.0)
    v = jnp.ones((1, d), dtype=jnp.float64)
    t = jnp.asarray([0.4], dtype=jnp.float64)
    out_reg = pca(xt=v, time=t, schedule=schedule)(v)
    out_no = pca_no_reg(xt=v, time=t, schedule=schedule)(v)
    alpha, sigma = scalar_alpha_sigma(schedule, t)
    expected_diff = (sigma ** 2 / alpha) * 0.5 * v
    self.assertTrue(jnp.allclose(out_reg - out_no, expected_diff, atol=1e-12))


class LowRankTweediePosteriorCovarianceTest(unittest.TestCase):

  def test_linear_psd_denoiser_recovers_scaled_jacobian(self):
    # With a symmetric PSD denoiser Jacobian, full-rank low-rank Tweedie
    # (symmetrize + project_psd defaults) reproduces full Tweedie's
    # output to high accuracy.
    schedule = schedules.CosineSchedule()
    rng = jax.random.PRNGKey(0)
    d = 6
    g0 = jax.random.normal(rng, (d, d), dtype=jnp.float64)
    g_psd = g0 @ g0.T  # symmetric PSD
    def denoiser(x):
      return x @ g_psd.T

    v = jax.random.normal(
        jax.random.PRNGKey(1), (1, d), dtype=jnp.float64,
    )
    t = jnp.asarray([0.4], dtype=jnp.float64)

    full = TweediePosteriorCovarianceFn()(
        xt=v, time=t, schedule=schedule, denoiser_fn=denoiser,
    )(v)
    lowrank = LowRankTweediePosteriorCovarianceFn(
        num_components=d, oversample=2, num_power_iters=0,
    )(xt=v, time=t, schedule=schedule, denoiser_fn=denoiser)(v)
    self.assertTrue(jnp.allclose(full, lowrank, atol=1e-6))

  def test_psd_projection_clips_negative_eigenvalues(self):
    # Construct a denoiser whose symmetrized Jacobian has a known negative
    # eigenvalue.  With project_psd=True (default) the operator should
    # zero out that direction; with project_psd=False it shouldn't.
    schedule = schedules.CosineSchedule()
    d = 4
    # G_sym = diag([2, -1, 1, 0.5]) -- one negative eigenvalue.
    g_sym = jnp.diag(jnp.asarray([2.0, -1.0, 1.0, 0.5], dtype=jnp.float64))
    def denoiser(x):
      return x @ g_sym  # symmetric -> J = J^T = g_sym

    # Probe along the negative-eigenvalue direction (e_2).
    v = jnp.asarray([[0.0, 1.0, 0.0, 0.0]], dtype=jnp.float64)
    t = jnp.asarray([0.5], dtype=jnp.float64)
    alpha, sigma = scalar_alpha_sigma(schedule, t)
    scale = float(sigma ** 2 / alpha)

    no_proj = LowRankTweediePosteriorCovarianceFn(
        num_components=d, oversample=2, num_power_iters=0,
        project_psd=False,
    )(xt=v, time=t, schedule=schedule, denoiser_fn=denoiser)(v)
    with_proj = LowRankTweediePosteriorCovarianceFn(
        num_components=d, oversample=2, num_power_iters=0,
        project_psd=True,
    )(xt=v, time=t, schedule=schedule, denoiser_fn=denoiser)(v)
    # Without projection: -scale * v.  With projection: 0 * v.
    self.assertTrue(jnp.allclose(no_proj, -scale * v, atol=1e-6))
    self.assertTrue(jnp.allclose(with_proj, jnp.zeros_like(v), atol=1e-6))

  def test_truncated_rank_approximates_not_equals(self):
    # For a full-rank Jacobian truncated to k < d, the sketch differs
    # from the full Tweedie.  Just check shape + finiteness.
    schedule = schedules.CosineSchedule()
    rng = jax.random.PRNGKey(0)
    d = 8
    g = jax.random.normal(rng, (d, d), dtype=jnp.float64)
    def denoiser(x):
      return x @ g.T

    v = jnp.ones((1, d), dtype=jnp.float64)
    t = jnp.asarray([0.5], dtype=jnp.float64)
    op = LowRankTweediePosteriorCovarianceFn(
        num_components=3, oversample=2, num_power_iters=1,
    )
    out = op(xt=v, time=t, schedule=schedule, denoiser_fn=denoiser)(v)
    self.assertEqual(out.shape, v.shape)
    self.assertTrue(jnp.all(jnp.isfinite(out)))

  def test_missing_denoiser_raises(self):
    schedule = schedules.CosineSchedule()
    op = LowRankTweediePosteriorCovarianceFn(num_components=2)
    v = jnp.ones((1, 4))
    with self.assertRaises(ValueError):
      op(xt=v, time=jnp.asarray([0.5]), schedule=schedule)

  def test_image_shape_preserved(self):
    schedule = schedules.CosineSchedule()
    h, w, c = 4, 4, 3
    rng = jax.random.PRNGKey(0)
    g = jax.random.normal(rng, (h * w * c, h * w * c), dtype=jnp.float64)
    def denoiser(x):
      flat = x.reshape(x.shape[0], -1) @ g.T
      return flat.reshape(x.shape)

    v = jax.random.normal(
        jax.random.PRNGKey(1), (2, h, w, c), dtype=jnp.float64,
    )
    op = LowRankTweediePosteriorCovarianceFn(num_components=4)
    out = op(xt=v, time=jnp.asarray([0.3]),
             schedule=schedule, denoiser_fn=denoiser)(v)
    self.assertEqual(out.shape, v.shape)
    self.assertTrue(jnp.all(jnp.isfinite(out)))


################################################################################
# Kalman correction (Pi-GDM family)
################################################################################


class KalmanCorrectionTest(unittest.TestCase):
  """End-to-end Kalman update sanity checks on a linear-Gaussian setup."""

  def _setup(self, *, n=4, batch=2):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    fwd = _MeanForwardFn()  # maps (B, n) -> (B, 1)
    x0 = jax.random.normal(jax.random.PRNGKey(0), (batch, n), dtype=jnp.float64)
    y_target = fwd.forward(x0) + 1.0  # shift target by 1 to create a residual
    return schedule, corruption, fwd, x0, y_target

  def test_isotropic_reduces_residual(self):
    schedule, corruption, fwd, x0, y = self._setup()

    def inference_fn(xt, time, conditioning=None):
      del xt, time, conditioning
      return {"x0": x0}

    correction = KalmanCorrectionFn(
        observation=y, forward_fn=fwd,
        posterior_covariance_fn=IsotropicPosteriorCovarianceFn(),
        observation_noise=0.1,
    )
    xt = jnp.zeros_like(x0)
    time = jnp.asarray([0.5], dtype=jnp.float64)
    denoiser_fn = make_denoiser_fn(inference_fn, corruption, time=time)
    x0_new = correction(
        x0, xt, time, denoiser_fn=denoiser_fn, schedule=schedule,
    )
    res_old = jnp.abs(fwd.forward(x0) - y)
    res_new = jnp.abs(fwd.forward(x0_new) - y)
    self.assertTrue(bool(jnp.all(res_new < res_old)))

  def test_tweedie_runs_and_reduces_residual(self):
    schedule, corruption, fwd, x0, y = self._setup()
    del x0  # overridden below

    # Non-trivial denoiser: xhat_0 = tanh(xt).  Jacobian = diag(1 - tanh^2).
    def inference_fn(xt, time, conditioning=None):
      del time, conditioning
      return {"x0": jnp.tanh(xt)}

    correction = KalmanCorrectionFn(
        observation=y, forward_fn=fwd,
        posterior_covariance_fn=TweediePosteriorCovarianceFn(),
        observation_noise=0.1, cg_max_iter=25, cg_tol=1e-8,
    )
    xt = jnp.full((y.shape[0], 4), 0.3, dtype=jnp.float64)
    time = jnp.asarray([0.5], dtype=jnp.float64)
    denoiser_fn = make_denoiser_fn(inference_fn, corruption, time=time)
    x0_at_xt = denoiser_fn(xt)
    x0_new = correction(
        x0_at_xt, xt, time, denoiser_fn=denoiser_fn, schedule=schedule,
    )
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

    correction = KalmanCorrectionFn(
        observation=y_match, forward_fn=fwd,
        posterior_covariance_fn=IsotropicPosteriorCovarianceFn(),
        observation_noise=0.1,
    )
    xt = jnp.zeros_like(x0)
    time = jnp.asarray([0.5], dtype=jnp.float64)
    denoiser_fn = make_denoiser_fn(inference_fn, corruption, time=time)
    x0_new = correction(
        x0, xt, time, denoiser_fn=denoiser_fn, schedule=schedule,
    )
    self.assertTrue(jnp.allclose(x0_new, x0, atol=1e-10))

  def test_pca_cov_reduces_residual(self):
    # PCA-based posterior covariance drops into KalmanCorrectionFn
    # and still reduces the measurement residual.
    schedule, corruption, fwd, x0, y = self._setup()
    n = x0.shape[1]
    # Use a random orthonormal U with some prescribed singular values
    # -- simulates a PCA factor extracted from training data.
    u, _ = jnp.linalg.qr(
        jax.random.normal(jax.random.PRNGKey(7), (n, n), dtype=jnp.float64),
    )
    svals = jnp.asarray([1.5, 1.0, 0.5, 0.2], dtype=jnp.float64)

    def inference_fn(xt, time, conditioning=None):
      del xt, time, conditioning
      return {"x0": x0}

    correction = KalmanCorrectionFn(
        observation=y, forward_fn=fwd,
        posterior_covariance_fn=PCAPosteriorCovarianceFn(
            u_factor=u, singular_values=svals, regulariser=1e-3,
        ),
        observation_noise=0.1,
    )
    xt = jnp.zeros_like(x0)
    time = jnp.asarray([0.5], dtype=jnp.float64)
    denoiser_fn = make_denoiser_fn(inference_fn, corruption, time=time)
    x0_new = correction(
        x0, xt, time, denoiser_fn=denoiser_fn, schedule=schedule,
    )
    res_old = jnp.abs(fwd.forward(x0) - y)
    res_new = jnp.abs(fwd.forward(x0_new) - y)
    self.assertTrue(bool(jnp.all(res_new < res_old)))

  def test_full_tweedie_with_minres_reduces_residual(self):
    # Indefinite Tweedie covariance + MINRES solver: full Tweedie should
    # work even when symmetrize alone leaves negative eigenvalues.  Use
    # a non-PSD denoiser whose Jacobian's symmetrisation is indefinite.
    schedule, corruption, fwd, x0, y = self._setup()
    n = x0.shape[1]
    del x0

    # Linear "denoiser" with a non-symmetric, non-PSD Jacobian.
    rng = jax.random.PRNGKey(7)
    G = jax.random.normal(rng, (n, n), dtype=jnp.float64) * 0.5
    G_sym = 0.5 * (G + G.T)
    # Confirm symmetrised G has at least one negative eigenvalue.
    eigs = jnp.linalg.eigvalsh(G_sym)
    self.assertTrue(bool(jnp.any(eigs < 0.0)))

    def inference_fn(xt, time, conditioning=None):
      del time, conditioning
      return {"x0": xt @ G.T}

    correction = KalmanCorrectionFn(
        observation=y, forward_fn=fwd,
        posterior_covariance_fn=TweediePosteriorCovarianceFn(symmetrize=True),
        observation_noise=0.5, cg_max_iter=80, cg_tol=1e-10,
        solver='minres',
    )
    xt = jnp.full((y.shape[0], n), 0.3, dtype=jnp.float64)
    time = jnp.asarray([0.5], dtype=jnp.float64)
    denoiser_fn = make_denoiser_fn(inference_fn, corruption, time=time)
    x0_at_xt = denoiser_fn(xt)
    x0_new = correction(
        x0_at_xt, xt, time, denoiser_fn=denoiser_fn, schedule=schedule,
    )
    res_old = jnp.abs(fwd.forward(x0_at_xt) - y)
    res_new = jnp.abs(fwd.forward(x0_new) - y)
    self.assertTrue(bool(jnp.all(res_new < res_old)))

  def test_full_tweedie_default_solver_auto_selects_minres(self):
    # Same indefinite-Jacobian setup as the explicit-MINRES test, but
    # leaving ``solver`` unset.  KalmanCorrectionFn should auto-pick
    # MINRES because TweediePosteriorCovarianceFn doesn't enforce PSD,
    # and the residual should still drop.  Regression guard against
    # silently falling back to CG on indefinite operators.
    schedule, corruption, fwd, x0, y = self._setup()
    n = x0.shape[1]
    del x0

    rng = jax.random.PRNGKey(7)
    G = jax.random.normal(rng, (n, n), dtype=jnp.float64) * 0.5
    G_sym = 0.5 * (G + G.T)
    self.assertTrue(bool(jnp.any(jnp.linalg.eigvalsh(G_sym) < 0.0)))

    def inference_fn(xt, time, conditioning=None):
      del time, conditioning
      return {"x0": xt @ G.T}

    correction = KalmanCorrectionFn(
        observation=y, forward_fn=fwd,
        posterior_covariance_fn=TweediePosteriorCovarianceFn(symmetrize=True),
        observation_noise=0.5, cg_max_iter=80, cg_tol=1e-10,
        # solver intentionally omitted -- exercise the auto-select path.
    )
    self.assertIsNone(correction.solver)
    xt = jnp.full((y.shape[0], n), 0.3, dtype=jnp.float64)
    time = jnp.asarray([0.5], dtype=jnp.float64)
    denoiser_fn = make_denoiser_fn(inference_fn, corruption, time=time)
    x0_at_xt = denoiser_fn(xt)
    x0_new = correction(
        x0_at_xt, xt, time, denoiser_fn=denoiser_fn, schedule=schedule,
    )
    res_old = jnp.abs(fwd.forward(x0_at_xt) - y)
    res_new = jnp.abs(fwd.forward(x0_new) - y)
    self.assertTrue(bool(jnp.all(res_new < res_old)))

  def test_low_rank_tweedie_reduces_residual(self):
    # Same correction as test_tweedie_runs_and_reduces_residual, but
    # using the randomized-SVD sketch.  Rank set to n for full-rank
    # agreement on this tiny test.
    schedule, corruption, fwd, x0, y = self._setup()
    n = x0.shape[1]
    del x0

    def inference_fn(xt, time, conditioning=None):
      del time, conditioning
      return {"x0": jnp.tanh(xt)}

    correction = KalmanCorrectionFn(
        observation=y, forward_fn=fwd,
        posterior_covariance_fn=LowRankTweediePosteriorCovarianceFn(
            num_components=n, oversample=2, num_power_iters=1,
        ),
        observation_noise=0.1, cg_max_iter=25, cg_tol=1e-8,
    )
    xt = jnp.full((y.shape[0], n), 0.3, dtype=jnp.float64)
    time = jnp.asarray([0.5], dtype=jnp.float64)
    denoiser_fn = make_denoiser_fn(inference_fn, corruption, time=time)
    x0_at_xt = denoiser_fn(xt)
    x0_new = correction(
        x0_at_xt, xt, time, denoiser_fn=denoiser_fn, schedule=schedule,
    )
    res_old = jnp.abs(fwd.forward(x0_at_xt) - y)
    res_new = jnp.abs(fwd.forward(x0_new) - y)
    self.assertTrue(bool(jnp.all(res_new < res_old)))


class PseudoInverseKalmanCorrectionTest(unittest.TestCase):

  def test_hard_singular_update_matches_row_space_conditional_mean(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    A = jnp.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64)
    fwd = LinearForwardFn(matrix=A)
    cov = jnp.diag(jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64))
    x0 = jnp.asarray([[0.25, -1.0, 2.0]], dtype=jnp.float64)
    y = jnp.asarray([[1.0, 1.0]], dtype=jnp.float64)
    time = jnp.asarray([0.5], dtype=jnp.float64)

    correction = PseudoInverseKalmanCorrectionFn(
        observation=y,
        forward_fn=fwd,
        posterior_covariance_fn=_FixedPosteriorCovarianceFn(covariance=cov),
        observation_noise=0.0,
        pinv_rtol=1e-12,
        pinv_atol=1e-12,
    )
    x0_new = correction(
        x0,
        jnp.zeros_like(x0),
        time,
        denoiser_fn=_denoiser_from_x0(x0, corruption, time),
        schedule=schedule,
    )

    expected = jnp.asarray([[1.0, -1.0, 2.0]], dtype=jnp.float64)
    self.assertTrue(jnp.allclose(x0_new, expected, atol=1e-10))
    self.assertTrue(jnp.allclose(fwd.forward(x0_new), y, atol=1e-10))


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

    target = jnp.zeros_like(x0)
    def log_prob(x):
      return -0.5 * jnp.sum((x - target) ** 2, axis=-1)

    twist = ClassifierTwistFn(log_prob_fn=log_prob)
    time = jnp.asarray([0.5], dtype=jnp.float64)
    log_psi = twist(
        jnp.zeros_like(x0), time,
        denoiser_fn=_denoiser_from_x0(x0, corruption, time),
    )
    self.assertTrue(jnp.allclose(log_psi, log_prob(x0), atol=1e-12))

  def test_energy_twist_is_negative_energy_over_temperature(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    batch, n = 3, 5
    x0 = jax.random.normal(jax.random.PRNGKey(1), (batch, n), dtype=jnp.float64)

    def energy(x):
      return jnp.sum(x ** 2, axis=-1)

    twist = EnergyTwistFn(energy_fn=energy, temperature=2.0)
    time = jnp.asarray([0.5], dtype=jnp.float64)
    log_psi = twist(
        jnp.zeros_like(x0), time,
        denoiser_fn=_denoiser_from_x0(x0, corruption, time),
    )
    self.assertTrue(jnp.allclose(log_psi, -energy(x0) / 2.0, atol=1e-12))


################################################################################
# SDE proposal ratio
################################################################################


class SdeProposalRatioTest(unittest.TestCase):

  def test_deterministic_churn_returns_zero(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    stepper = SdeStep(corruption_process=corruption, churn=0.0)
    xt = jnp.ones((3, 8), dtype=jnp.float64)
    ratio = _proposal_log_ratio_for(
        stepper, corruption,
        jnp.zeros_like(xt), jnp.ones_like(xt),
        xt, xt + 0.1,
        jnp.asarray([0.5]), jnp.asarray([0.4]),
    )
    self.assertTrue(bool(jnp.all(ratio == 0.0)))

  def test_stochastic_churn_finite_and_nonzero(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    stepper = SdeStep(corruption_process=corruption, churn=1.0)
    xt = jnp.ones((2, 4), dtype=jnp.float64)
    ratio = _proposal_log_ratio_for(
        stepper, corruption,
        jnp.zeros_like(xt), jnp.full_like(xt, 0.3),
        xt, jnp.full_like(xt, 0.2),
        jnp.asarray([0.5]), jnp.asarray([0.4]),
    )
    self.assertTrue(bool(jnp.all(jnp.isfinite(ratio))))
    self.assertTrue(bool(jnp.all(ratio != 0.0)))


################################################################################
# Additional stepper ratios: AdjustedDDIM / Velocity / Heun
################################################################################


class DeterministicStepperRatioTest(unittest.TestCase):
  """AdjustedDDIMStep and HeunStep are deterministic; ratio must be 0."""

  def test_adjusted_ddim_ratio_is_zero(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    stepper = AdjustedDDIMStep(corruption_process=corruption)
    xt = jnp.ones((3, 8), dtype=jnp.float64)
    ratio = _proposal_log_ratio_for(
        stepper, corruption,
        jnp.zeros_like(xt), jnp.ones_like(xt),
        xt, xt + 0.1,
        jnp.asarray([0.5]), jnp.asarray([0.4]),
    )
    self.assertTrue(bool(jnp.all(ratio == 0.0)))


class VelocityProposalRatioTest(unittest.TestCase):

  def test_deterministic_epsilon_returns_zero(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    stepper = VelocityStep(corruption_process=corruption, epsilon=0.0)
    xt = jnp.ones((3, 8), dtype=jnp.float64)
    ratio = _proposal_log_ratio_for(
        stepper, corruption,
        jnp.zeros_like(xt), jnp.ones_like(xt),
        xt, xt + 0.1,
        jnp.asarray([0.5]), jnp.asarray([0.4]),
    )
    self.assertTrue(bool(jnp.all(ratio == 0.0)))

  def test_stochastic_epsilon_finite_and_nonzero(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    stepper = VelocityStep(corruption_process=corruption, epsilon=0.5)
    xt = jnp.ones((2, 4), dtype=jnp.float64)
    ratio = _proposal_log_ratio_for(
        stepper, corruption,
        jnp.zeros_like(xt), jnp.full_like(xt, 0.3),
        xt, jnp.full_like(xt, 0.2),
        jnp.asarray([0.5]), jnp.asarray([0.4]),
    )
    self.assertTrue(bool(jnp.all(jnp.isfinite(ratio))))
    self.assertTrue(bool(jnp.all(ratio != 0.0)))


################################################################################
# Classifier-free guidance correction adapter
################################################################################


class CFGCompositionTest(unittest.TestCase):
  """CFG is denoiser composition -- either via ``make_cfg_inference_fn``
  at the inference-level, or via :class:`LinearBlendDenoiserFn` at the
  denoiser-level.  Both express the same scalar-blend formula."""

  def test_make_cfg_inference_fn_reproduces_linear_blend(self):
    batch, n = 2, 4
    x0_cond = jax.random.normal(
        jax.random.PRNGKey(0), (batch, n), dtype=jnp.float64,
    )
    x0_uncond = jax.random.normal(
        jax.random.PRNGKey(1), (batch, n), dtype=jnp.float64,
    )

    def cond_fn(xt, time, conditioning=None):
      del xt, time, conditioning
      return {"x0": x0_cond}

    def uncond_fn(xt, time, conditioning=None):
      del xt, time, conditioning
      return {"x0": x0_uncond}

    w = 1.5
    blended = make_cfg_inference_fn(cond_fn, uncond_fn, ScalarGuidanceFn(guidance=w))
    out = blended(xt=jnp.zeros((batch, n), dtype=jnp.float64),
                  time=jnp.asarray([0.5], dtype=jnp.float64),
                  conditioning=None)
    expected = x0_cond * (1.0 + w) - x0_uncond * w
    self.assertTrue(jnp.allclose(out["x0"], expected, atol=1e-12))

  def test_linear_blend_denoiser_scalar_cfg(self):
    batch, n = 2, 4
    x0_cond = jax.random.normal(
        jax.random.PRNGKey(0), (batch, n), dtype=jnp.float64,
    )
    x0_uncond = jax.random.normal(
        jax.random.PRNGKey(1), (batch, n), dtype=jnp.float64,
    )
    cond_denoiser = lambda xt: x0_cond
    uncond_denoiser = lambda xt: x0_uncond
    w = 1.5
    blended = LinearBlendDenoiserFn(
        denoisers=(cond_denoiser, uncond_denoiser),
        weights=(1.0 + w, -w),
    )
    out = blended(jnp.zeros((batch, n)))
    expected = x0_cond * (1.0 + w) - x0_uncond * w
    self.assertTrue(jnp.allclose(out, expected, atol=1e-12))


################################################################################
# MARK: Posterior cloud (R-sample wrapper around any inference fn)
################################################################################


class PosteriorCloudFnTest(unittest.TestCase):
  """``make_posterior_cloud_fn`` should:

    1. Emit ``[B, R, *x0_shape]`` for any ``R >= 1``.
    2. For a stochastic inference fn, the ``R`` slices along the
       population axis are independent draws.
    3. For a deterministic inference fn, the ``R`` slices are bit-equal
       (the explicit mean-plug-in baseline).
    4. The ``R=1`` slice ``cloud[:, 0]`` is bit-identical to the same-rng
       :func:`make_denoiser_fn` output -- the cloud is a strict
       generalisation, not an alternative code path.
  """

  def test_shape_and_independence_for_stochastic_inference_fn(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)

    # Stochastic inference fn: x_0 = xt + 0.1 * Z, Z ~ N(0, I).
    def stoch_fn(xt, time, conditioning=None, rng=None):
      del time, conditioning
      if rng is None:
        raise ValueError("stoch_fn needs rng")
      return {"x0": xt + 0.1 * jax.random.normal(rng, xt.shape, dtype=xt.dtype)}

    batch, n, R = 4, 8, 6
    xt = jnp.ones((batch, n), dtype=jnp.float64)
    t = jnp.asarray([0.5], dtype=jnp.float64)
    rng = jax.random.PRNGKey(0)

    cloud_fn = make_posterior_cloud_fn(
        stoch_fn, corruption,
        time=t, rng=rng, population_size=R,
    )
    cloud = cloud_fn(xt)
    self.assertEqual(cloud.shape, (batch, R, n))

    # All R slices should differ (independent rng splits).
    for r in range(1, R):
      self.assertFalse(
          jnp.allclose(cloud[:, 0], cloud[:, r], atol=1e-8),
          f"cloud[:, 0] and cloud[:, {r}] are identical -- rng split broken.",
      )

  def test_deterministic_inference_fn_collapses_to_identical_copies(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)

    def det_fn(xt, time, conditioning=None):
      del time, conditioning
      return {"x0": xt}

    batch, n, R = 3, 5, 4
    xt = jax.random.normal(jax.random.PRNGKey(7), (batch, n), dtype=jnp.float64)
    t = jnp.asarray([0.5], dtype=jnp.float64)

    cloud = make_posterior_cloud_fn(
        det_fn, corruption,
        time=t, rng=jax.random.PRNGKey(0), population_size=R,
    )(xt)
    self.assertEqual(cloud.shape, (batch, R, n))
    # All R slices identical -- the mean-plug-in baseline.
    for r in range(1, R):
      self.assertTrue(jnp.allclose(cloud[:, 0], cloud[:, r], atol=1e-12))

  def test_R_eq_1_matches_make_denoiser_fn_with_first_split_key(self):
    """Cloud is a strict generalisation: cloud[:, 0] == denoiser(xt) when
    both use the same first split key from rng."""
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)

    def stoch_fn(xt, time, conditioning=None, rng=None):
      del time, conditioning
      if rng is None:
        raise ValueError("stoch_fn needs rng")
      return {"x0": xt + 0.05 * jax.random.normal(rng, xt.shape, dtype=xt.dtype)}

    batch, n = 2, 4
    xt = jnp.ones((batch, n), dtype=jnp.float64)
    t = jnp.asarray([0.5], dtype=jnp.float64)
    rng = jax.random.PRNGKey(11)

    # Cloud with R=1 must match make_denoiser_fn called with split(rng, 1)[0].
    cloud = make_posterior_cloud_fn(
        stoch_fn, corruption,
        time=t, rng=rng, population_size=1,
    )(xt)
    expected = make_denoiser_fn(
        stoch_fn, corruption,
        time=t, rng=jax.random.split(rng, 1)[0],
    )(xt)
    self.assertEqual(cloud.shape, (batch, 1, n))
    self.assertTrue(jnp.allclose(cloud[:, 0], expected, atol=1e-12))

  def test_population_size_zero_raises(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)

    def det_fn(xt, time, conditioning=None):
      del time, conditioning
      return {"x0": xt}

    with self.assertRaises(ValueError):
      make_posterior_cloud_fn(
          det_fn, corruption,
          time=jnp.asarray([0.5]), rng=jax.random.PRNGKey(0),
          population_size=0,
      )


if __name__ == "__main__":
  unittest.main()
