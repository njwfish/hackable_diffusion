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

"""Tests for simplicial step sampler."""

import chex
from hackable_diffusion.lib import random_utils
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.corruption import simplicial
from hackable_diffusion.lib.sampling import base as sampling_base
from hackable_diffusion.lib.sampling import simplicial_step_sampler
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Type Aliases
################################################################################

DiffusionStep = sampling_base.DiffusionStep
StepInfo = sampling_base.StepInfo
SimplicialProcess = simplicial.SimplicialProcess
SimplicialDDIMStep = simplicial_step_sampler.SimplicialDDIMStep
log_beta_shrinkage = simplicial_step_sampler.log_beta_shrinkage

################################################################################
# MARK: Tests
################################################################################


class BetaShrinkageTest(parameterized.TestCase):
  """Tests for beta shrinkage."""

  def test_identity(self):
    key = jax.random.PRNGKey(0)
    log_x = jax.nn.log_softmax(jax.random.normal(key, (2, 4, 3)))
    concentration = jnp.array(10.0)
    log_y = log_beta_shrinkage(key, log_x, concentration, kappa=1.0)
    chex.assert_trees_all_close(log_x, log_y)

  def test_shape(self):
    key = jax.random.PRNGKey(0)
    log_x = jax.nn.log_softmax(jax.random.normal(key, (2, 4, 3)))
    concentration = jnp.array(10.0)
    log_y = log_beta_shrinkage(key, log_x, concentration, kappa=0.5)
    self.assertEqual(log_x.shape, log_y.shape)
    # Check that it's still normalized (logsumexp == 0)
    chex.assert_trees_all_close(
        jax.nn.logsumexp(log_y, axis=-1), jnp.zeros(log_x.shape[:-1]), atol=1e-5
    )

  def test_stochasticity(self):
    key = jax.random.PRNGKey(0)
    log_x = jax.nn.log_softmax(jax.random.normal(key, (2, 4, 3)))
    concentration = jnp.array(10.0)
    key1, key2 = jax.random.split(key)
    log_y1 = log_beta_shrinkage(key1, log_x, concentration, kappa=0.5)
    log_y2 = log_beta_shrinkage(key2, log_x, concentration, kappa=0.5)
    # They should be different
    self.assertGreater(jnp.abs(log_y1 - log_y2).sum(), 0.0)

  def test_distribution(self):
    key = jax.random.PRNGKey(0)
    # Target concentration
    alpha = jnp.array([1.0, 2.0, 3.0])
    kappa = 0.5
    target_alpha = alpha * kappa  # [0.5, 1.0, 1.5]

    # Generate many samples to check distribution
    n_samples = 20_000

    key, subkey = jax.random.split(key)
    log_x = random_utils.log_dirichlet_fast(subkey, alpha, shape=(n_samples,))

    _, subkey = jax.random.split(key)
    log_y = log_beta_shrinkage(subkey, log_x, alpha, kappa=kappa)
    y = jnp.exp(log_y)

    # Theoretical moments for Dir(target_alpha)
    sum_alpha = jnp.sum(target_alpha)
    expected_mean = target_alpha / sum_alpha
    expected_var = (
        target_alpha
        * (sum_alpha - target_alpha)
        / (sum_alpha**2 * (sum_alpha + 1))
    )

    # Empirical moments
    empirical_mean = jnp.mean(y, axis=0)
    empirical_var = jnp.var(y, axis=0)

    # Check closeness
    chex.assert_trees_all_close(empirical_mean, expected_mean, atol=5e-3)
    chex.assert_trees_all_close(empirical_var, expected_var, atol=1e-3)


class SimplicialStepSamplerTest(parameterized.TestCase):
  """Tests for simplicial step samplers."""

  def setUp(self):
    super().setUp()
    self.schedule = schedules.LinearDiscreteSchedule()
    self.num_categories = 4
    self.process = SimplicialProcess.uniform_process(
        schedule=self.schedule, num_categories=self.num_categories
    )
    key = jax.random.PRNGKey(0)
    self.initial_noise = jax.random.normal(key, (2, 4, self.num_categories))
    self.ddim_step = SimplicialDDIMStep(
        corruption_process=self.process, churn=0.0
    )
    self.churn_step = SimplicialDDIMStep(
        corruption_process=self.process, churn=0.5
    )

  def _dummy_inference_fn(self, xt, conditioning, time):
    del conditioning, time
    # Return logits that will deterministically sample category 1.
    logits = jnp.zeros(xt.shape[:-1] + (self.num_categories,))
    logits = logits.at[..., 1].set(10.0)
    return {'logits': logits}

  def test_initialize(self):
    for sampler in [self.churn_step, self.ddim_step]:
      initial_step_info = StepInfo(
          step=0,
          time=jnp.array([1.0, 1.0])[:, None, None],
          rng=jax.random.PRNGKey(0),
      )
      initial_step = sampler.initialize(
          initial_noise=self.initial_noise,
          initial_step_info=initial_step_info,
      )

      chex.assert_trees_all_equal(
          initial_step,
          DiffusionStep(
              xt=self.initial_noise,
              step_info=initial_step_info,
              aux={'logits': self.initial_noise},
          ),
      )

  def test_update_shapes(self):
    for sampler in [self.churn_step, self.ddim_step]:
      initial_step_info = StepInfo(
          step=0,
          time=jnp.array([0.5, 0.5])[:, None, None],
          rng=jax.random.PRNGKey(0),
      )
      initial_step = sampler.initialize(
          initial_noise=self.initial_noise,
          initial_step_info=initial_step_info,
      )
      prediction = self._dummy_inference_fn(
          xt=initial_step.xt,
          conditioning={},
          time=initial_step.step_info.time,
      )

      next_step_info = StepInfo(
          step=1,
          time=jnp.array([0.1, 0.1])[:, None, None],
          rng=jax.random.PRNGKey(1),
      )
      next_step = sampler.update(
          prediction=prediction,
          current_step=initial_step,
          next_step_info=next_step_info,
      )

      self.assertEqual(next_step.xt.shape, self.initial_noise.shape)
      self.assertEqual(next_step.xt.dtype, self.initial_noise.dtype)

  def test_finalize(self):
    for sampler in [self.churn_step, self.ddim_step]:
      initial_step_info = StepInfo(
          step=0,
          time=jnp.array([0.1, 0.1])[:, None, None],
          rng=jax.random.PRNGKey(0),
      )
      initial_step = sampler.initialize(
          initial_noise=self.initial_noise,
          initial_step_info=initial_step_info,
      )
      prediction = self._dummy_inference_fn(
          xt=initial_step.xt,
          conditioning={},
          time=initial_step.step_info.time,
      )

      last_step_info = StepInfo(
          step=1,
          time=jnp.array([0.0, 0.0])[:, None, None],
          rng=jax.random.PRNGKey(1),
      )
      final_step = sampler.finalize(
          prediction=prediction,
          current_step=initial_step,
          last_step_info=last_step_info,
      )

      self.assertEqual(final_step.xt.shape, self.initial_noise.shape)

  def test_churn_variation(self):
    # Test that different churn values yield different results
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.ddim_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )
    prediction = self._dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )

    next_step_info = StepInfo(
        step=1,
        time=jnp.array([0.1, 0.1])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )

    next_step_ddim = self.ddim_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=next_step_info,
    )
    next_step_churn = self.churn_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=next_step_info,
    )

    # They should have the same shape but different values
    self.assertEqual(next_step_ddim.xt.shape, next_step_churn.xt.shape)
    # With high probability they are different
    diff = jnp.abs(next_step_ddim.xt - next_step_churn.xt).sum()
    self.assertGreater(diff, 0.0)


if __name__ == '__main__':
  absltest.main()
