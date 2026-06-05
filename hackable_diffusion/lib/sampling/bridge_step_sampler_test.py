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

"""Tests for the unified posterior-bridge sampler step ``PosteriorBridgeStep``.

One stepper, every modality: Gaussian (continuous), categorical, Dirichlet
(simplicial), and multimodal (nested).  Each modality's reverse step lives
on its ``CorruptionProcess`` via ``sample_endpoint`` + ``bridge_step``.
"""

import jax
import pytest

jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True)
def _enable_x64():
  """This module's numerics require 64-bit precision (see conftest.py)."""
  jax.config.update("jax_enable_x64", True)
  yield

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from hackable_diffusion.lib import multimodal
from hackable_diffusion.lib import posterior
from hackable_diffusion.lib.corruption import discrete
from hackable_diffusion.lib.corruption import gaussian
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.corruption import simplicial
from hackable_diffusion.lib.inference.posterior_sampler import (
    PosteriorSamplerInferenceFn,
)
from hackable_diffusion.lib.sampling import base as sampling_base
from hackable_diffusion.lib.sampling import bridge_step_sampler
from hackable_diffusion.lib.sampling import sampling
from hackable_diffusion.lib.sampling import simplicial_step_sampler
from hackable_diffusion.lib.sampling import time_scheduling

from absl.testing import absltest

PosteriorBridgeStep = bridge_step_sampler.PosteriorBridgeStep


class GaussianBridgeTest(absltest.TestCase):

  def test_constant_posterior_lands_on_x0(self):
    # A constant posterior endpoint c => the chain lands on c (up to the
    # time schedule's safety-epsilon final time), for any trajectory.
    process = gaussian.GaussianProcess(schedule=schedules.RFSchedule())
    c = jnp.asarray([[2.0, -1.0, 0.5, 4.0]], dtype=jnp.float64)

    def inference_fn(xt, conditioning, time, rng=None):
      del conditioning, time, rng
      return {"x0": jnp.broadcast_to(c, xt.shape)}

    sampler = sampling.DiffusionSampler(
        time_schedule=time_scheduling.UniformTimeSchedule(),
        stepper=PosteriorBridgeStep(corruption_process=process),
        num_steps=8,
    )
    last_step, _ = sampler(
        inference_fn=inference_fn,
        rng=jax.random.PRNGKey(1),
        initial_noise=jax.random.normal(
            jax.random.PRNGKey(0), (1, 4), dtype=jnp.float64
        ),
        conditioning=None,
    )
    np.testing.assert_allclose(last_step.xt, c, atol=1e-3)

  def test_end_to_end_with_posterior_sampler_inference_fn(self):
    # The real PosteriorSamplerInferenceFn (+ a tiny distributional net)
    # driving PosteriorBridgeStep -- the canonical "how to run it" recipe.
    process = gaussian.GaussianProcess(schedule=schedules.RFSchedule())
    net = _TinyPosteriorNet()
    variables = net.init(  # channel-concat input is 2 * C (= 6 for C = 3)
        jax.random.PRNGKey(0),
        time=jnp.zeros((2,), dtype=jnp.float64),
        xt=jnp.zeros((2, 6), dtype=jnp.float64),
    )
    inference_fn = PosteriorSamplerInferenceFn(
        network=net,
        params=variables.get("params", {}),
        xi_injector=posterior.channel_concat_xi,
    )
    sampler = sampling.DiffusionSampler(
        time_schedule=time_scheduling.UniformTimeSchedule(),
        stepper=PosteriorBridgeStep(corruption_process=process),
        num_steps=6,
    )
    last_step, _ = sampler(
        inference_fn=inference_fn,
        rng=jax.random.PRNGKey(2),
        initial_noise=process.sample_from_invariant(
            jax.random.PRNGKey(1), jnp.zeros((4, 3), dtype=jnp.float64)
        ),
        conditioning=None,
    )
    self.assertEqual(last_step.xt.shape, (4, 3))
    self.assertTrue(bool(jnp.all(jnp.isfinite(last_step.xt))))


class _TinyPosteriorNet(nn.Module):
  """Minimal shape-preserving distributional backbone (``x0 = x_t + 0.1 xi``)."""

  @nn.compact
  def __call__(self, *, time, xt, conditioning=None, is_training=False):
    del time, conditioning, is_training
    c = xt.shape[-1] // 2
    return {"x0": xt[..., :c] + 0.1 * xt[..., c:]}


class CategoricalBridgeTest(absltest.TestCase):

  def test_masking_end_to_end(self):
    # A logits-emitting model + PosteriorBridgeStep on a masking process:
    # the run must preserve shape and yield valid token ids.
    k = 4
    process = discrete.CategoricalProcess.masking_process(
        schedule=schedules.LinearDiscreteSchedule(), num_categories=k,
    )

    def inference_fn(xt, conditioning, time):
      del conditioning, time
      return {"logits": jnp.zeros(xt.shape[:-1] + (k,), dtype=jnp.float32)}

    sampler = sampling.DiffusionSampler(
        time_schedule=time_scheduling.UniformTimeSchedule(),
        stepper=PosteriorBridgeStep(corruption_process=process),
        num_steps=8,
    )
    initial_noise = process.sample_from_invariant(
        jax.random.PRNGKey(0), jnp.zeros((2, 5, 1), dtype=jnp.int32)
    )
    last_step, _ = sampler(
        inference_fn=inference_fn,
        rng=jax.random.PRNGKey(1),
        initial_noise=initial_noise,
        conditioning=None,
    )
    self.assertEqual(last_step.xt.shape, (2, 5, 1))
    # Valid token ids in [0, process_num_categories).
    self.assertTrue(bool(jnp.all(last_step.xt >= 0)))
    self.assertTrue(
        bool(jnp.all(last_step.xt < process.process_num_categories))
    )


class SimplicialBridgeTest(absltest.TestCase):

  def _process(self):
    return simplicial.SimplicialProcess(
        schedule=schedules.CosineDiscreteSchedule(),
        invariant_probs=(1.0 / 3, 1.0 / 3, 1.0 / 3),
        num_categories=3,
        temperature=1.0,
    )

  def test_bridge_step_matches_simplicial_ddim_churn1(self):
    # SimplicialProcess.bridge_step must equal SimplicialDDIMStep(churn=1)
    # bit-for-bit (same 4-way key split), locking it to the tested stepper.
    process = self._process()
    key = jax.random.PRNGKey(0)
    logits = jax.random.normal(jax.random.PRNGKey(3), (2, 4, 3), dtype=jnp.float64)
    log_xt = jax.nn.log_softmax(
        jax.random.normal(jax.random.PRNGKey(4), (2, 4, 3), dtype=jnp.float64),
        axis=-1,
    )
    t = jnp.full((2,), 0.7, dtype=jnp.float64)
    s = jnp.full((2,), 0.3, dtype=jnp.float64)

    p0_hat = process.sample_endpoint(key, {"logits": logits}, log_xt, t)
    xs_process = process.bridge_step(key, p0_hat, log_xt, t, s)

    stepper = simplicial_step_sampler.SimplicialDDIMStep(
        corruption_process=process, churn=1.0,
    )
    current = sampling_base.DiffusionStep(
        xt=log_xt,
        step_info=sampling_base.StepInfo(
            step=jnp.array(1), time=t, rng=jax.random.PRNGKey(99),
        ),
        aux={},
    )
    next_info = sampling_base.StepInfo(step=jnp.array(2), time=s, rng=key)
    xs_ddim = stepper.update({"logits": logits}, current, next_info).xt

    np.testing.assert_allclose(xs_process, xs_ddim, atol=1e-10)


class NestedBridgeTest(absltest.TestCase):

  def test_bridge_step_maps_over_modalities(self):
    # One PosteriorBridge interface drives mixed modalities: the nested
    # bridge maps a Gaussian + categorical pair, preserving per-modality
    # dtype/shape.
    process = multimodal.NestedProcess(
        processes={
            "img": gaussian.GaussianProcess(schedule=schedules.RFSchedule()),
            "lbl": discrete.CategoricalProcess.masking_process(
                schedule=schedules.LinearDiscreteSchedule(), num_categories=4,
            ),
        }
    )
    key = jax.random.PRNGKey(0)
    x0 = {
        "img": jax.random.normal(key, (2, 3), dtype=jnp.float64),
        "lbl": jnp.zeros((2, 5, 1), dtype=jnp.int32),
    }
    xt = {
        "img": jax.random.normal(jax.random.PRNGKey(1), (2, 3), dtype=jnp.float64),
        "lbl": jnp.full((2, 5, 1), 4, dtype=jnp.int32),  # all masked
    }
    t = {"img": jnp.full((2,), 0.7), "lbl": jnp.full((2,), 0.7)}
    s = {"img": jnp.full((2,), 0.3), "lbl": jnp.full((2,), 0.3)}

    out = process.bridge_step(key, x0, xt, t, s)
    self.assertEqual(set(out), {"img", "lbl"})
    self.assertEqual(out["img"].shape, (2, 3))
    self.assertTrue(jnp.issubdtype(out["img"].dtype, jnp.floating))
    self.assertEqual(out["lbl"].shape, (2, 5, 1))
    self.assertTrue(jnp.issubdtype(out["lbl"].dtype, jnp.integer))


if __name__ == "__main__":
  absltest.main()
