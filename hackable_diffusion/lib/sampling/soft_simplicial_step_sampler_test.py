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

"""Tests for the simplex-valued simplicial DDIM stepper."""

from absl.testing import absltest
import chex
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.corruption import simplicial
from hackable_diffusion.lib.sampling import base as sampling_base
from hackable_diffusion.lib.sampling.soft_simplicial_step_sampler import (
    SoftSimplicialDDIMStep,
)
import jax
import jax.numpy as jnp


DiffusionStep = sampling_base.DiffusionStep
StepInfo = sampling_base.StepInfo


class SoftSimplicialDDIMStepTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.process = simplicial.SimplicialProcess.uniform_process(
        schedule=schedules.LinearDiscreteSchedule(),
        num_categories=3,
    )
    self.stepper = SoftSimplicialDDIMStep(
        corruption_process=self.process,
        churn=0.0,
    )

  def test_soft_clean_state_is_fixed_point_for_ddim_update(self):
    probs = jnp.asarray([0.1, 0.2, 0.7], dtype=jnp.float32)
    log_probs = jnp.log(probs)
    log_xt = jnp.broadcast_to(log_probs, (2, 4, 3))
    initial_step = self.stepper.initialize(
        initial_noise=log_xt,
        initial_step_info=StepInfo(
            step=0,
            time=jnp.array([0.5, 0.5])[:, None, None],
            rng=jax.random.PRNGKey(0),
        ),
    )
    prediction = {'logits': jnp.broadcast_to(log_probs, (2, 4, 3))}

    next_step = self.stepper.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=StepInfo(
            step=1,
            time=jnp.array([0.1, 0.1])[:, None, None],
            rng=jax.random.PRNGKey(1),
        ),
    )

    chex.assert_trees_all_close(
        jnp.exp(next_step.xt),
        jnp.broadcast_to(probs, (2, 4, 3)),
        atol=1e-5,
    )
    chex.assert_trees_all_close(
        jax.nn.logsumexp(next_step.xt, axis=-1),
        jnp.zeros((2, 4)),
        atol=1e-5,
    )

  def test_update_aux_contains_predicted_logits(self):
    log_xt = jax.nn.log_softmax(
        jax.random.normal(jax.random.PRNGKey(2), (2, 4, 3)),
        axis=-1,
    )
    logits = jax.random.normal(jax.random.PRNGKey(3), (2, 4, 3))
    initial_step = self.stepper.initialize(
        initial_noise=log_xt,
        initial_step_info=StepInfo(
            step=0,
            time=jnp.array([0.5, 0.5])[:, None, None],
            rng=jax.random.PRNGKey(0),
        ),
    )

    next_step = self.stepper.update(
        prediction={'logits': logits},
        current_step=initial_step,
        next_step_info=StepInfo(
            step=1,
            time=jnp.array([0.1, 0.1])[:, None, None],
            rng=jax.random.PRNGKey(1),
        ),
    )

    self.assertIn('logits', next_step.aux)
    chex.assert_trees_all_close(next_step.aux['logits'], logits)


if __name__ == '__main__':
  absltest.main()
