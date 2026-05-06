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

"""Tests for time scheduling."""

import chex
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.sampling import time_scheduling
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Tests
################################################################################


class TimeScheduleTest(parameterized.TestCase, absltest.TestCase):

  # MARK: UniformTimeSchedule tests

  def test_uniform_all_step_infos(self):
    time_schedule = time_scheduling.UniformTimeSchedule(
        span=utils.SafeSpan(safety_epsilon=0.1)
    )
    data_spec = jnp.zeros((2, 3))
    expected = jnp.array([
        [[0.9], [0.9]],
        [[0.7], [0.7]],
        [[0.5], [0.5]],
        [[0.3], [0.3]],
        [[0.1], [0.1]],
    ])
    chex.assert_trees_all_close(
        time_schedule.all_step_infos(
            rng=jax.random.PRNGKey(0),
            num_steps=5,
            data_spec=data_spec,
        ).time,
        expected,
    )

  def test_uniform_all_step_infos_with_starting_noise(self):
    time_schedule = time_scheduling.UniformTimeSchedule(
        span=utils.SafeSpan(_minval=0.0, _maxval=0.6, safety_epsilon=0.1)
    )
    data_spec = jnp.zeros((2, 3))
    expected = jnp.array([
        [[0.5], [0.5]],
        [[0.4], [0.4]],
        [[0.3], [0.3]],
        [[0.2], [0.2]],
        [[0.1], [0.1]],
    ])
    chex.assert_trees_all_close(
        time_schedule.all_step_infos(
            rng=jax.random.PRNGKey(0),
            num_steps=5,
            data_spec=data_spec,
        ).time,
        expected,
    )

  def test_uniform_all_step_infos_without_safety_epsilon(self):
    time_schedule = time_scheduling.UniformTimeSchedule(span=utils.SafeSpan())
    data_spec = jnp.zeros((2, 3))
    expected = jnp.array([
        [[1.0], [1.0]],
        [[0.75], [0.75]],
        [[0.5], [0.5]],
        [[0.25], [0.25]],
        [[0.0], [0.0]],
    ])
    chex.assert_trees_all_close(
        time_schedule.all_step_infos(
            rng=jax.random.PRNGKey(0),
            num_steps=5,
            data_spec=data_spec,
        ).time,
        expected,
    )

  # MARK: EDMTimeSchedule tests

  def test_all_step_infos(self):
    time_schedule = time_scheduling.EDMTimeSchedule(
        span=utils.SafeSpan(), rho=2.0
    )
    data_spec = jnp.zeros((2, 3))
    expected = jnp.array([
        [[1.0], [1.0]],
        [[0.5625], [0.5625]],
        [[0.25], [0.25]],
        [[0.0625], [0.0625]],
        [[0.0], [0.0]],
    ])
    chex.assert_trees_all_close(
        time_schedule.all_step_infos(
            rng=jax.random.PRNGKey(0),
            num_steps=5,
            data_spec=data_spec,
        ).time,
        expected,
    )

  def test_edm_all_step_infos_with_rho_one_is_uniform(self):
    uniform_time_schedule = time_scheduling.UniformTimeSchedule(
        span=utils.SafeSpan(safety_epsilon=0.1)
    )
    edm_time_schedule = time_scheduling.EDMTimeSchedule(
        span=utils.SafeSpan(safety_epsilon=0.1), rho=1.0
    )
    data_spec = jnp.zeros((2, 3))
    num_steps = 5
    uniform_steps = uniform_time_schedule.all_step_infos(
        rng=jax.random.PRNGKey(0),
        num_steps=num_steps,
        data_spec=data_spec,
    ).time
    edm_steps = edm_time_schedule.all_step_infos(
        rng=jax.random.PRNGKey(0),
        num_steps=num_steps,
        data_spec=data_spec,
    ).time
    chex.assert_trees_all_close(uniform_steps, edm_steps)

  @parameterized.parameters(0.0, -1.0)
  def test_edm_invalid_rho(self, rho):
    with self.assertRaisesRegex(ValueError, "rho must be positive"):
      time_scheduling.EDMTimeSchedule(span=utils.SafeSpan(), rho=rho)


if __name__ == "__main__":
  absltest.main()
