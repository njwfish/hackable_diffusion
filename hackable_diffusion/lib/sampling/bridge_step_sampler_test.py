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

"""Tests for the posterior-bridge sampler step ``BridgeStep``."""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import flax.linen as nn

from hackable_diffusion.lib import posterior
from hackable_diffusion.lib.corruption import base as corruption_base
from hackable_diffusion.lib.corruption import gaussian
from hackable_diffusion.lib.corruption import interpolants
from hackable_diffusion.lib.corruption import priors
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.corruption import targets
from hackable_diffusion.lib.guidance.sampler import ConditionalDiffusionSampler
from hackable_diffusion.lib.inference.posterior_sampler import (
    PosteriorSamplerInferenceFn,
)
from hackable_diffusion.lib.sampling import bridge_step_sampler
from hackable_diffusion.lib.sampling import sampling
from hackable_diffusion.lib.sampling import time_scheduling

from absl.testing import absltest


class BridgeStepSamplerTest(absltest.TestCase):

  def test_constant_posterior_lands_on_x0(self):
    # If the posterior always returns the same clean endpoint c, then the
    # final step lands on c (up to the time schedule's safety-epsilon
    # final time, where sigma is tiny but nonzero), for any trajectory --
    # the bridge collapses toward delta_c as s -> 0.
    process = gaussian.GaussianProcess(schedule=schedules.RFSchedule())
    c = jnp.asarray([[2.0, -1.0, 0.5, 4.0]], dtype=jnp.float64)

    def inference_fn(xt, conditioning, time, rng=None):
      del conditioning, time, rng
      return {"x0": jnp.broadcast_to(c, xt.shape)}

    sample_fn = sampling.DiffusionSampler(
        time_schedule=time_scheduling.UniformTimeSchedule(),
        stepper=bridge_step_sampler.BridgeStep(corruption_process=process),
        num_steps=8,
    )
    initial_noise = jax.random.normal(
        jax.random.PRNGKey(0), (1, 4), dtype=jnp.float64
    )
    last_step, _ = sample_fn(
        inference_fn=inference_fn,
        initial_noise=initial_noise,
        conditioning=None,
        rng=jax.random.PRNGKey(1),
    )
    np.testing.assert_allclose(last_step.xt, c, atol=1e-3)

  def test_reads_x0_from_prediction_not_score(self):
    # BridgeStep must consume the endpoint sample directly; it should not
    # depend on score/velocity. A prediction carrying a misleading score
    # alongside x0 must not change the result.
    process = gaussian.GaussianProcess(schedule=schedules.CosineSchedule())
    interp = process.interpolant
    x0 = jax.random.normal(jax.random.PRNGKey(0), (4, 3), dtype=jnp.float64)
    xt = jax.random.normal(jax.random.PRNGKey(1), (4, 3), dtype=jnp.float64)
    t = jnp.full((4,), 0.7, dtype=jnp.float64)
    s = jnp.full((4,), 0.4, dtype=jnp.float64)

    stepper = bridge_step_sampler.BridgeStep(corruption_process=process)
    extracted = stepper._x0({"x0": x0, "score": xt * 99.0}, xt, t)
    np.testing.assert_array_equal(extracted, x0)

    # And the resulting step matches the interpolant bridge_step on that x0.
    expected = interp.bridge_step(x0=x0, xt=xt, t=t, s=s)
    np.testing.assert_allclose(
        interp.bridge_step(x0=extracted, xt=xt, t=t, s=s), expected, atol=1e-12
    )


class BridgeStepKernelTest(absltest.TestCase):

  def test_deterministic_bridge_ratio_is_zero(self):
    # LinearInterpolant bridge has sigma_step = 0 => Dirac proposal =>
    # the SMC proposal log-ratio is identically zero regardless of the
    # corrected vs uncorrected endpoint.
    process = gaussian.GaussianProcess(schedule=schedules.RFSchedule())
    stepper = bridge_step_sampler.BridgeStep(corruption_process=process)
    xt_prev = jax.random.normal(jax.random.PRNGKey(0), (4, 3), dtype=jnp.float64)
    xt_next = jax.random.normal(jax.random.PRNGKey(1), (4, 3), dtype=jnp.float64)
    kernel = stepper.kernel(
        prediction_uncorrected={"x0": jnp.zeros((4, 3))},
        prediction_corrected={"x0": jnp.ones((4, 3))},
        xt=xt_prev,
        time_prev=jnp.full((4,), 0.7, dtype=jnp.float64),
        time_next=jnp.full((4,), 0.3, dtype=jnp.float64),
    )
    ratio = kernel.log_density_ratio(xt_prev, xt_next)
    np.testing.assert_allclose(ratio, jnp.zeros(4), atol=1e-12)

  def test_stochastic_bridge_ratio_matches_gaussian_quadratic(self):
    # StochasticInterpolant bridge is Gaussian with std sigma_step; the
    # proposal ratio is the standard quadratic form.
    si = interpolants.StochasticInterpolant(
        alpha=lambda t: 1.0 - t,
        beta=lambda t: t,
        gamma=interpolants.canonical_gamma,
    )
    process = corruption_base.InterpolantProcess(
        prior=priors.GaussianPrior(),
        interpolant=si,
        targets=targets.VelocityOnlyTargets(),
    )
    stepper = bridge_step_sampler.BridgeStep(corruption_process=process)
    t_val, s_val = 0.8, 0.3
    x0_unc = jnp.asarray([[0.0, 0.0, 0.0]], dtype=jnp.float64)
    x0_cor = jnp.asarray([[1.0, -1.0, 0.5]], dtype=jnp.float64)
    xt_prev = jnp.asarray([[0.2, 0.4, -0.1]], dtype=jnp.float64)
    xt_next = jnp.asarray([[0.3, 0.1, 0.0]], dtype=jnp.float64)

    kernel = stepper.kernel(
        prediction_uncorrected={"x0": x0_unc},
        prediction_corrected={"x0": x0_cor},
        xt=xt_prev,
        time_prev=jnp.asarray([t_val], dtype=jnp.float64),
        time_next=jnp.asarray([s_val], dtype=jnp.float64),
    )
    ratio = kernel.log_density_ratio(xt_prev, xt_next)

    # Hand-rolled reference using coeff_x0, coeff_xt, sigma_step (scalars).
    cx0, cxt, sig = si.bridge_coefficients(
        jnp.asarray([t_val]), jnp.asarray([s_val])
    )
    cx0, cxt, sig = cx0.reshape(()), cxt.reshape(()), sig.reshape(())
    mu_p = cx0 * x0_unc + cxt * xt_prev
    mu_q = cx0 * x0_cor + cxt * xt_prev
    sq_p = jnp.sum((xt_next - mu_p) ** 2)
    sq_q = jnp.sum((xt_next - mu_q) ** 2)
    expected = (sq_q - sq_p) / (2.0 * sig**2)
    np.testing.assert_allclose(ratio, jnp.reshape(expected, (1,)), rtol=1e-9)


class _ConstantShiftCorrection:
  """Trivial observation-free correction ``x0 -> x0 + shift`` for tests."""

  def __init__(self, shift):
    self.shift = shift

  def __call__(self, x0, xt, time, *, denoiser_fn, schedule,
               cloud_fn=None, rng=None):
    del xt, time, denoiser_fn, schedule, cloud_fn, rng
    return x0 + self.shift


class BridgeGuidanceCompositionTest(absltest.TestCase):

  def test_composes_with_conditional_sampler_and_correction(self):
    # The whole point of the bridge view: a posterior correction shifts
    # x0, then the *unchanged* bridge step applies. Exercises the SMC
    # loop, kernel() proposal ratio, and the x0 flow end-to-end.
    process = gaussian.GaussianProcess(schedule=schedules.RFSchedule())
    base = sampling.DiffusionSampler(
        time_schedule=time_scheduling.UniformTimeSchedule(),
        stepper=bridge_step_sampler.BridgeStep(corruption_process=process),
        num_steps=6,
    )
    sampler = ConditionalDiffusionSampler(
        base_sampler=base,
        corruption_process=process,
        correction_fn=_ConstantShiftCorrection(0.1),
    )

    def inference_fn(xt, conditioning, time, rng=None):
      del conditioning, time
      xi = jax.random.normal(rng, xt.shape, dtype=xt.dtype)
      return {"x0": xt + 0.5 * xi}  # stochastic posterior sample

    final_step, _, log_w = sampler(
        inference_fn=inference_fn,
        rng=jax.random.PRNGKey(0),
        initial_noise=jax.random.normal(
            jax.random.PRNGKey(1), (4, 3), dtype=jnp.float64
        ),
        conditioning=None,
    )
    self.assertEqual(final_step.xt.shape, (4, 3))
    self.assertEqual(log_w.shape, (4,))
    self.assertTrue(bool(jnp.all(jnp.isfinite(final_step.xt))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(log_w))))


class _TinyPosteriorNet(nn.Module):
  """Minimal shape-preserving distributional backbone.

  Consumes the channel-concatenated ``[x_t, xi]`` input and returns a
  stochastic clean-endpoint sample ``x0 = x_t + 0.1 xi`` -- enough to
  exercise the real :class:`PosteriorSamplerInferenceFn` wiring without a
  trained network.
  """

  @nn.compact
  def __call__(self, *, time, xt, conditioning=None, is_training=False):
    del time, conditioning, is_training
    c = xt.shape[-1] // 2
    return {"x0": xt[..., :c] + 0.1 * xt[..., c:]}


class BridgeEndToEndTest(absltest.TestCase):
  """The real posterior-sampler inference fn driving BridgeStep end-to-end.

  This is the canonical "how do you run it" recipe: a distributional
  network -> PosteriorSamplerInferenceFn -> BridgeStep in a plain
  DiffusionSampler.  Swapping any Gaussian stepper for BridgeStep is the
  only change versus a standard run.
  """

  def test_posterior_sampler_inference_fn_runs_with_bridge_step(self):
    process = gaussian.GaussianProcess(schedule=schedules.RFSchedule())
    net = _TinyPosteriorNet()
    variables = net.init(  # channel-concat input is 2 * C (= 6 for C = 3)
        jax.random.PRNGKey(0),
        time=jnp.zeros((2,), dtype=jnp.float64),
        xt=jnp.zeros((2, 6), dtype=jnp.float64),
    )
    params = variables.get("params", {})

    inference_fn = PosteriorSamplerInferenceFn(
        network=net,
        params=params,
        xi_injector=posterior.channel_concat_xi,
    )
    sampler = sampling.DiffusionSampler(
        time_schedule=time_scheduling.UniformTimeSchedule(),
        stepper=bridge_step_sampler.BridgeStep(corruption_process=process),
        num_steps=6,
    )
    initial_noise = process.sample_from_invariant(
        jax.random.PRNGKey(1),
        data_spec=jnp.zeros((4, 3), dtype=jnp.float64),
    )
    last_step, _ = sampler(
        inference_fn=inference_fn,
        rng=jax.random.PRNGKey(2),
        initial_noise=initial_noise,
        conditioning=None,
    )
    self.assertEqual(last_step.xt.shape, (4, 3))
    self.assertTrue(bool(jnp.all(jnp.isfinite(last_step.xt))))


if __name__ == "__main__":
  absltest.main()
