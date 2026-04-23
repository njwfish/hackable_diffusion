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
from hackable_diffusion.lib.sampling.gaussian_step_sampler import DDIMStep
from hackable_diffusion.lib.sampling.sampling import DiffusionSampler
from hackable_diffusion.lib.sampling.time_scheduling import UniformTimeSchedule

from hackable_diffusion.lib.guidance.corrections import (
    GradientCorrectionFn,
    IteratedCorrectionFn,
    dps_prefactor,
    miyasawa_prefactor,
)
from hackable_diffusion.lib.guidance.proposal_ratio import (
    ddim_proposal_log_ratio,
    proposal_log_ratio,
    register_proposal_ratio,
)
from hackable_diffusion.lib.guidance.resamplers import (
    ESSThresholdedResamplerFn,
    MultinomialResamplerFn,
    NoResamplerFn,
    SystematicResamplerFn,
    normalised_weights,
)
from hackable_diffusion.lib.guidance.sampler import ConditionalDiffusionSampler
from hackable_diffusion.lib.guidance.twists import GaussianLikelihoodTwistFn
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


if __name__ == "__main__":
  unittest.main()
