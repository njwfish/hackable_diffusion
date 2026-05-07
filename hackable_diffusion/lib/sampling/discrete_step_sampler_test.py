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

"""Tests for discrete step sampler."""

import chex
from hackable_diffusion.lib.corruption import discrete
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.sampling import base as sampling_base
from hackable_diffusion.lib.sampling import discrete_step_sampler
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Type Aliases
################################################################################

DiffusionStep = sampling_base.DiffusionStep
StepInfo = sampling_base.StepInfo
CategoricalProcess = discrete.CategoricalProcess
UnMaskingStep = discrete_step_sampler.UnMaskingStep

################################################################################
# MARK: Tests
################################################################################


class RemaskingFnTest(parameterized.TestCase):
  """Tests for remasking functions."""

  def setUp(self):
    super().setUp()
    self.schedule = schedules.LinearDiscreteSchedule()
    self.s = jnp.array([0.1, 0.2])
    self.t = jnp.array([0.5, 0.6])

  def test_no_remasking(self):
    fn = discrete_step_sampler.NoRemaskingFn()
    result = fn(self.s, self.t)
    chex.assert_trees_all_close(result, jnp.zeros_like(self.s))

  @parameterized.named_parameters(
      ('invalid_switch', 0.5, 1.0, 0.0, ValueError),
      ('invalid_max_cap', -0.1, 0.0, 1.0, ValueError),
  )
  def test_max_capped_remasking_init_raises(
      self, max_cap, switch_min, switch_max, error
  ):
    with self.assertRaises(error):
      discrete_step_sampler.MaxCappedRemaskingFn(
          schedule=self.schedule,
          max_cap=max_cap,
          switch_min=switch_min,
          switch_max=switch_max,
      )

  def test_max_capped_remasking_call(self):
    fn = discrete_step_sampler.MaxCappedRemaskingFn(
        schedule=self.schedule, max_cap=0.8
    )
    alpha_s = self.schedule.alpha(self.s)
    alpha_t = self.schedule.alpha(self.t)
    expected = jnp.minimum((1.0 - alpha_s) / alpha_t, 0.8)
    result = fn(self.s, self.t)
    chex.assert_trees_all_close(result, expected)

  @parameterized.named_parameters(
      ('invalid_switch', 0.5, 1.0, 0.0, ValueError),
      ('invalid_rescale_factor_neg', -0.1, 0.0, 1.0, ValueError),
      ('invalid_rescale_factor_one', 1.0, 0.0, 1.0, ValueError),
  )
  def test_rescaled_remasking_init_raises(
      self, rescale_factor, switch_min, switch_max, error
  ):
    with self.assertRaises(error):
      discrete_step_sampler.RescaledRemaskingFn(
          schedule=self.schedule,
          rescale_factor=rescale_factor,
          switch_min=switch_min,
          switch_max=switch_max,
      )

  def test_rescaled_remasking_call(self):
    fn = discrete_step_sampler.RescaledRemaskingFn(
        schedule=self.schedule, rescale_factor=0.5
    )
    alpha_s = self.schedule.alpha(self.s)
    alpha_t = self.schedule.alpha(self.t)
    expected = 0.5 * jnp.minimum((1.0 - alpha_s) / alpha_t, 1.0)
    result = fn(self.s, self.t)
    chex.assert_trees_all_close(result, expected)


class CorruptedMaskFnTest(absltest.TestCase):
  """Tests for corrupted mask functions."""

  def setUp(self):
    super().setUp()
    self.xt = jnp.array([[[0], [1], [4]]])  # shape (1, 3, 1)
    self.schedule = schedules.LinearDiscreteSchedule()
    self.process = CategoricalProcess.masking_process(
        schedule=self.schedule, num_categories=4
    )  # mask_value is 4

  def test_all_corrupted_mask_fn(self):
    fn = discrete_step_sampler.AllCorruptedMaskFn()
    mask = fn(self.xt)
    expected_mask = jnp.ones_like(self.xt, dtype=jnp.bool_)
    chex.assert_trees_all_equal(mask, expected_mask)

  def test_mask_value_corrupted_mask_fn(self):
    fn = discrete_step_sampler.MaskValueCorruptedMaskFn(process=self.process)
    mask = fn(self.xt)
    expected_mask = jnp.array([[[False], [False], [True]]])
    chex.assert_trees_all_equal(mask, expected_mask)


class UnMaskingStepTest(absltest.TestCase):
  """Tests for the UnMaskingStep sampler."""

  def setUp(self):
    super().setUp()
    self.schedule = schedules.LinearDiscreteSchedule()
    self.num_categories = 4
    self.process = CategoricalProcess.masking_process(
        schedule=self.schedule, num_categories=self.num_categories
    )
    self.initial_noise = jnp.ones((2, 4, 1), dtype=jnp.int32) * (
        self.process.process_num_categories - 1
    )
    self.unmasking_step = UnMaskingStep(corruption_process=self.process)

  def _dummy_inference_fn(self, xt, conditioning, time):
    del conditioning, time
    # Return logits that will deterministically sample category 0.
    logits = jnp.zeros(xt.shape[:-1] + (self.process.num_categories,))
    logits = logits.at[..., 1].set(10.0)
    return {'logits': logits}

  def test_initialize(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([1.0, 1.0])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.unmasking_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )

    init_logits = jnp.repeat(
        self.initial_noise, self.process.num_categories, axis=-1
    )
    init_logits = jnp.zeros_like(init_logits, dtype=jnp.float32)

    chex.assert_trees_all_equal(
        initial_step,
        DiffusionStep(
            xt=self.initial_noise,
            step_info=initial_step_info,
            aux={'logits': init_logits},
        ),
    )

  def test_update(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.unmasking_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )
    prediction = self._dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )

    # Test case 1: Full unmasking
    next_step_info_full_unmask = StepInfo(
        step=1,
        time=jnp.array([0.0, 0.0])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    next_step_full_unmask = self.unmasking_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=next_step_info_full_unmask,
    )
    expected_xt_full_unmask = jnp.ones_like(self.initial_noise)
    chex.assert_trees_all_equal(
        next_step_full_unmask.xt, expected_xt_full_unmask
    )

    # Test case 2: No unmasking
    next_step_info_no_unmask = StepInfo(
        step=1,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    next_step_no_unmask = self.unmasking_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=next_step_info_no_unmask,
    )
    chex.assert_trees_all_equal(next_step_no_unmask.xt, initial_step.xt)

  def test_finalize(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.unmasking_step.initialize(
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
    final_step = self.unmasking_step.finalize(
        prediction=prediction,
        current_step=initial_step,
        last_step_info=last_step_info,
    )
    expected_xt = jnp.ones_like(self.initial_noise)
    chex.assert_trees_all_equal(final_step.xt, expected_xt)

  def test_fail_for_masking_process(self):
    process = CategoricalProcess.uniform_process(
        schedule=self.schedule, num_categories=self.num_categories
    )
    with self.assertRaisesRegex(
        ValueError, 'UnMaskingStep only supports masking processes.'
    ):
      discrete_step_sampler.UnMaskingStep(corruption_process=process)


class DiscreteDDIMStepTest(absltest.TestCase):
  """Tests for the DiscreteDDIMStep sampler."""

  def setUp(self):
    super().setUp()
    self.schedule = schedules.LinearDiscreteSchedule()
    self.num_categories = 4
    self.process = CategoricalProcess.uniform_process(
        schedule=self.schedule, num_categories=self.num_categories
    )
    key = jax.random.PRNGKey(0)
    self.initial_noise = jax.random.randint(
        key, (2, 4, 1), 0, self.process.process_num_categories
    )
    self.ddim_step = discrete_step_sampler.DiscreteDDIMStep(
        corruption_process=self.process
    )

  def _dummy_inference_fn(self, xt, conditioning, time):
    del conditioning, time
    # Return logits that will deterministically sample category 0.
    logits = jnp.zeros(xt.shape[:-1] + (self.process.process_num_categories,))
    logits = logits.at[..., 1].set(10.0)
    return {'logits': logits}

  def test_initialize(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([1.0, 1.0])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.ddim_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )

    init_logits = jnp.repeat(
        self.initial_noise, self.process.num_categories, axis=-1
    )
    init_logits = jnp.zeros_like(init_logits, dtype=jnp.float32)

    chex.assert_trees_all_equal(
        initial_step,
        DiffusionStep(
            xt=self.initial_noise,
            step_info=initial_step_info,
            aux={'logits': init_logits},
        ),
    )

  def test_update(self):
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
    next_step = self.ddim_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=next_step_info,
    )

    self.assertEqual(next_step.xt.shape, self.initial_noise.shape)
    self.assertEqual(next_step.xt.dtype, self.initial_noise.dtype)

  def test_finalize(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.1, 0.1])[:, None, None],
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

    last_step_info = StepInfo(
        step=1,
        time=jnp.array([0.0, 0.0])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    final_step = self.ddim_step.finalize(
        prediction=prediction,
        current_step=initial_step,
        last_step_info=last_step_info,
    )

    expected_xt = jnp.ones_like(self.initial_noise)
    chex.assert_trees_all_close(final_step.xt, expected_xt)

  def test_fail_for_masking_process(self):
    process = CategoricalProcess.masking_process(
        schedule=self.schedule, num_categories=self.num_categories
    )
    with self.assertRaisesRegex(
        ValueError, 'DiscreteDDIMStep does not support masking processes.'
    ):
      discrete_step_sampler.DiscreteDDIMStep(corruption_process=process)

  def test_fail_for_zero_invariant_probs(self):
    invariant_probs = (1.0, 0.0, 0.0)
    process = CategoricalProcess(
        schedule=self.schedule,
        invariant_probs=invariant_probs,
        num_categories=len(invariant_probs),
    )
    with self.assertRaisesRegex(
        ValueError,
        'DiscreteDDIMStep does not support invariant probabilities'
        ' with 0.0 probability mass for any element.',
    ):
      discrete_step_sampler.DiscreteDDIMStep(corruption_process=process)


class IntegratedDiscreteDDIMStepTest(absltest.TestCase):
  """Tests for the IntegratedDiscreteDDIMStep sampler."""

  def setUp(self):
    super().setUp()
    self.schedule = schedules.LinearDiscreteSchedule()
    self.num_categories = 4
    self.process = CategoricalProcess.uniform_process(
        schedule=self.schedule, num_categories=self.num_categories
    )
    key = jax.random.PRNGKey(0)
    self.initial_noise = jax.random.randint(
        key, (2, 4, 1), 0, self.process.process_num_categories
    )
    self.integrated_ddim_step = (
        discrete_step_sampler.IntegratedDiscreteDDIMStep(
            corruption_process=self.process
        )
    )

  def _dummy_inference_fn(self, xt, conditioning, time):
    del conditioning, time
    # Return logits that will deterministically sample category 0.
    logits = jnp.zeros(xt.shape[:-1] + (self.process.process_num_categories,))
    logits = logits.at[..., 1].set(10.0)
    return {'logits': logits}

  def test_initialize(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([1.0, 1.0])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.integrated_ddim_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )

    init_logits = jnp.repeat(
        self.initial_noise, self.process.num_categories, axis=-1
    )
    init_logits = jnp.zeros_like(init_logits, dtype=jnp.float32)

    chex.assert_trees_all_equal(
        initial_step,
        DiffusionStep(
            xt=self.initial_noise,
            step_info=initial_step_info,
            aux={'logits': init_logits},
        ),
    )

  def test_update(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.integrated_ddim_step.initialize(
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
    next_step = self.integrated_ddim_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=next_step_info,
    )

    self.assertEqual(next_step.xt.shape, self.initial_noise.shape)
    self.assertEqual(next_step.xt.dtype, self.initial_noise.dtype)

  def test_finalize(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.1, 0.1])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.integrated_ddim_step.initialize(
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
    final_step = self.integrated_ddim_step.finalize(
        prediction=prediction,
        current_step=initial_step,
        last_step_info=last_step_info,
    )

    expected_xt = jnp.ones_like(self.initial_noise)
    chex.assert_trees_all_close(final_step.xt, expected_xt)

  def test_fail_for_masking_process(self):
    process = CategoricalProcess.masking_process(
        schedule=self.schedule, num_categories=self.num_categories
    )
    with self.assertRaisesRegex(
        ValueError,
        'IntegratedDiscreteDDIMStep does not support masking processes.',
    ):
      discrete_step_sampler.IntegratedDiscreteDDIMStep(
          corruption_process=process
      )

  def test_fail_for_zero_invariant_probs(self):
    invariant_probs = (1.0, 0.0, 0.0)
    process = CategoricalProcess(
        schedule=self.schedule,
        invariant_probs=invariant_probs,
        num_categories=len(invariant_probs),
    )
    with self.assertRaisesRegex(
        ValueError,
        'IntegratedDiscreteDDIMStep does not support invariant probabilities'
        ' with 0.0 probability mass for any element.',
    ):
      discrete_step_sampler.IntegratedDiscreteDDIMStep(
          corruption_process=process
      )


class DiscreteFlowMatchingStepTest(absltest.TestCase):
  """Tests for the DiscreteFlowMatchingStep sampler."""

  def setUp(self):
    super().setUp()
    self.schedule = schedules.LinearDiscreteSchedule()
    self.num_categories = 4
    self.process = CategoricalProcess.uniform_process(
        schedule=self.schedule, num_categories=self.num_categories
    )
    key = jax.random.PRNGKey(0)
    self.initial_noise = jax.random.randint(
        key, (2, 4, 1), 0, self.process.process_num_categories
    )
    self.dfm_step = discrete_step_sampler.DiscreteFlowMatchingStep(
        corruption_process=self.process
    )

  def _dummy_inference_fn(self, xt, conditioning, time):
    del conditioning, time
    # Return logits that will deterministically sample category 1.
    logits = jnp.zeros(xt.shape[:-1] + (self.process.num_categories,))
    logits = logits.at[..., 1].set(10.0)
    return {'logits': logits}

  def test_initialize(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([1.0, 1.0])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.dfm_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )
    init_logits = jnp.repeat(
        self.initial_noise, self.process.num_categories, axis=-1
    )
    init_logits = jnp.zeros_like(init_logits, dtype=jnp.float32) - jnp.inf

    chex.assert_trees_all_equal(
        initial_step,
        DiffusionStep(
            xt=self.initial_noise,
            step_info=initial_step_info,
            aux={'logits': init_logits},
        ),
    )

  def test_update(self):
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(0),
    )
    initial_step = self.dfm_step.initialize(
        initial_noise=self.initial_noise,
        initial_step_info=initial_step_info,
    )
    prediction = self._dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )

    # Test case 1: Full unmasking (alpha_s=1.0, alpha_t=0.5 -> prob_jump=1.0)
    next_step_info_full = StepInfo(
        step=1,
        time=jnp.array([0.0, 0.0])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    next_step_full = self.dfm_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=next_step_info_full,
    )
    expected_xt_full = jnp.ones_like(self.initial_noise)
    chex.assert_trees_all_equal(next_step_full.xt, expected_xt_full)

    # Test case 2: No jump (alpha_s=0.5, alpha_t=0.5 -> prob_jump=0.0)
    next_step_info_no = StepInfo(
        step=1,
        time=jnp.array([0.5, 0.5])[:, None, None],
        rng=jax.random.PRNGKey(1),
    )
    next_step_no = self.dfm_step.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=next_step_info_no,
    )
    chex.assert_trees_all_equal(next_step_no.xt, initial_step.xt)

  def test_update_with_gamma(self):
    num_samples = 10
    initial_step_info = StepInfo(
        step=0,
        time=jnp.array([[0.5]] * num_samples)[:, :, None],
        rng=jax.random.PRNGKey(0),
    )
    # Start with more samples to ensure noise jump is detected.
    initial_xt = jnp.ones((num_samples, 4, 1), dtype=jnp.int32)
    initial_step = self.dfm_step.initialize(
        initial_noise=initial_xt,
        initial_step_info=initial_step_info,
    )

    # Predict category 1 (same as current).
    prediction = self._dummy_inference_fn(
        xt=initial_step.xt,
        conditioning={},
        time=initial_step.step_info.time,
    )

    # Use gamma that won't clip.
    dfm_step_gamma = discrete_step_sampler.DiscreteFlowMatchingStep(
        corruption_process=self.process, stoch_coeff=1.0
    )

    next_step_info = StepInfo(
        step=1,
        time=jnp.array([[0.4]] * num_samples)[:, :, None],
        rng=jax.random.PRNGKey(1),
    )
    next_step = dfm_step_gamma.update(
        prediction=prediction,
        current_step=initial_step,
        next_step_info=next_step_info,
    )

    # Some tokens should have changed to noise (not 1).
    self.assertTrue(jnp.any(next_step.xt != 1))


class DDIMRoutingEquivalenceTest(absltest.TestCase):
  """Verify routing-based DDIM matches the original logit-space computation.

  The original DiscreteDDIMStep computed the reverse posterior in full
  M-dimensional logit space:

    first_logit[k]  = log(r * 1[k=xt] + (1-r) * π(xt))
    second_logit[k] = log(αs * 1[k=x0] + (1-αs) * π(k))
    total_logit = first_logit + second_logit

  The routing reformulation decomposes this into 3-way routing weights.
  This test checks that both produce exactly the same distribution over
  output tokens, including the edge case where x0 == xt.
  """

  def _posterior_distribution(
      self, xt, x0, alpha_s, alpha_t, invariant_probs_vec
  ):
    """Compute the exact posterior in probability space.

    p(x_s | x_t, x_0) ∝ p(x_t | x_s) * p(x_s | x_0)

    Evaluated for every x_s in {0, ..., M-1}.

    Args:
      xt: Current token.
      x0: Predicted clean token.
      alpha_s: Diffusion schedule value at time s.
      alpha_t: Diffusion schedule value at time t.
      invariant_probs_vec: Invariant distribution.

    Returns:
      The M-dimensional posterior distribution.
    """
    voc_size = int(invariant_probs_vec.shape[0])
    ratio = alpha_t / alpha_s

    # Build unnormalized weight for each x_s value.
    weights = []
    for xs in range(voc_size):
      # p(x_t | x_s) = r * 1[xs=xt] + (1-r) * π(xt)
      p_xt_given_xs = ratio * float(xs == xt) + (1.0 - ratio) * float(
          invariant_probs_vec[xt]
      )
      # p(x_s | x_0) = α_s * 1[xs=x0] + (1-α_s) * π(xs)
      p_xs_given_x0 = alpha_s * float(xs == x0) + (1.0 - alpha_s) * float(
          invariant_probs_vec[xs]
      )
      weights.append(p_xt_given_xs * p_xs_given_x0)

    weights = jnp.array(weights)
    return weights / jnp.sum(weights)

  def _routing_distribution(
      self, xt, x0, alpha_s, alpha_t, invariant_probs_vec
  ):
    """Compute the routing-based posterior distribution.

    Mirrors the actual code in DiscreteDDIMStep.update.

    Args:
      xt: Current token.
      x0: Predicted clean token.
      alpha_s: Diffusion schedule value at time s.
      alpha_t: Diffusion schedule value at time t.
      invariant_probs_vec: Invariant distribution.

    Returns:
      The M-dimensional posterior distribution.
    """
    ratio = alpha_t / alpha_s
    pi_xt = float(invariant_probs_vec[xt])

    # T2 → stay, T4 → noise, T3 → clean
    p_stay = ratio * (1.0 - alpha_s) * pi_xt
    p_noise = (1.0 - ratio) * (1.0 - alpha_s) * pi_xt
    p_clean = (1.0 - ratio) * alpha_s * pi_xt

    # When x0 == xt, CLEAN is a no-op. Merge T1 and p_clean into p_stay.
    if x0 == xt:
      p_stay = p_stay + ratio * alpha_s + p_clean
      p_clean = 0.0

    total = p_stay + p_noise + p_clean
    p_stay_norm = p_stay / total
    p_noise_norm = p_noise / total
    p_clean_norm = p_clean / total

    # Build the M-dimensional output distribution by marginalizing
    # over the routing action:
    #   P(output=k) = P(STAY)*1[k=xt] + P(NOISE)*π(k) + P(CLEAN)*1[k=x0]
    inv_probs = [float(p) for p in invariant_probs_vec]
    dist = [p_noise_norm * inv_probs[k] for k in range(len(inv_probs))]
    dist[xt] += p_stay_norm
    dist[x0] += p_clean_norm
    return jnp.array(dist)

  def test_equivalence_x0_neq_xt(self):
    """Test routing matches posterior when x0 != xt."""
    voc_size = 5
    invariant_probs = jnp.array([0.1, 0.3, 0.2, 0.25, 0.15])

    for xt_val in range(voc_size):
      for x0_val in range(voc_size):
        if x0_val == xt_val:
          continue
        for alpha_s_val in [0.2, 0.5, 0.8]:
          alpha_t = 0.05
          p_exact = self._posterior_distribution(
              xt_val, x0_val, alpha_s_val, alpha_t, invariant_probs
          )
          p_route = self._routing_distribution(
              xt_val, x0_val, alpha_s_val, alpha_t, invariant_probs
          )
          chex.assert_trees_all_close(p_exact, p_route, atol=1e-6)

  def test_equivalence_x0_eq_xt(self):
    """Test routing matches posterior when x0 == xt (the T1 cross-term)."""
    voc_size = 5
    invariant_probs = jnp.array([0.1, 0.3, 0.2, 0.25, 0.15])

    for xt_val in range(voc_size):
      x0_val = xt_val
      for alpha_s_val in [0.2, 0.5, 0.8]:
        alpha_t = 0.05
        p_exact = self._posterior_distribution(
            xt_val, x0_val, alpha_s_val, alpha_t, invariant_probs
        )
        p_route = self._routing_distribution(
            xt_val, x0_val, alpha_s_val, alpha_t, invariant_probs
        )
        chex.assert_trees_all_close(p_exact, p_route, atol=1e-6)

  def test_equivalence_nonuniform_invariant(self):
    """Test with a highly non-uniform invariant distribution."""
    voc_size = 3
    invariant_probs = jnp.array([0.01, 0.01, 0.98])

    for xt_val in range(voc_size):
      for x0_val in range(voc_size):
        p_exact = self._posterior_distribution(
            xt_val, x0_val, 0.3, 0.7, invariant_probs
        )
        p_route = self._routing_distribution(
            xt_val, x0_val, 0.3, 0.7, invariant_probs
        )
        chex.assert_trees_all_close(p_exact, p_route, atol=1e-6)


class ApplyRoutingTest(absltest.TestCase):
  """Tests for the _sample_routing helper."""

  def test_deterministic_stay(self):
    # routing_weights = [1, 0, 0] means stay.
    routing_weights = discrete_step_sampler.RoutingWeights(
        stay=jnp.array([[[1.0], [1.0]]]),
        noise=jnp.array([[[0.0], [0.0]]]),
        clean=jnp.array([[[0.0], [0.0]]]),
    )
    xt = jnp.array([[[3], [5]]])
    x0 = jnp.array([[[0], [1]]])
    x_noise = jnp.array([[[2], [2]]])
    key = jax.random.PRNGKey(0)

    new_xt = discrete_step_sampler._sample_routing(
        routing_weights=routing_weights,
        xt=xt,
        x0=x0,
        x_noise=x_noise,
        key=key,
    )
    chex.assert_trees_all_equal(new_xt, xt)

  def test_deterministic_clean(self):
    # routing_weights = [0, 0, 1] means jump to x0.
    routing_weights = discrete_step_sampler.RoutingWeights(
        stay=jnp.array([[[0.0], [0.0]]]),
        noise=jnp.array([[[0.0], [0.0]]]),
        clean=jnp.array([[[1.0], [1.0]]]),
    )
    xt = jnp.array([[[3], [5]]])
    x0 = jnp.array([[[0], [1]]])
    x_noise = jnp.array([[[2], [2]]])
    key = jax.random.PRNGKey(0)

    new_xt = discrete_step_sampler._sample_routing(
        routing_weights=routing_weights,
        xt=xt,
        x0=x0,
        x_noise=x_noise,
        key=key,
    )
    chex.assert_trees_all_equal(new_xt, x0)

  def test_deterministic_noise(self):
    # routing_weights = [0, 1, 0] means jump to noise.
    routing_weights = discrete_step_sampler.RoutingWeights(
        stay=jnp.array([[[0.0], [0.0]]]),
        noise=jnp.array([[[1.0], [1.0]]]),
        clean=jnp.array([[[0.0], [0.0]]]),
    )
    xt = jnp.array([[[3], [5]]])
    x0 = jnp.array([[[0], [1]]])
    x_noise = jnp.array([[[2], [2]]])
    key = jax.random.PRNGKey(0)

    new_xt = discrete_step_sampler._sample_routing(
        routing_weights=routing_weights,
        xt=xt,
        x0=x0,
        x_noise=x_noise,
        key=key,
    )
    chex.assert_trees_all_equal(new_xt, x_noise)

  def test_mixed_routing(self):
    # Position 0: deterministic stay, Position 1: deterministic clean.
    routing_weights = discrete_step_sampler.RoutingWeights(
        stay=jnp.array([[[1.0], [0.0]]]),
        noise=jnp.array([[[0.0], [0.0]]]),
        clean=jnp.array([[[0.0], [1.0]]]),
    )
    xt = jnp.array([[[3], [5]]])
    x0 = jnp.array([[[0], [1]]])
    x_noise = jnp.array([[[2], [2]]])
    key = jax.random.PRNGKey(0)

    new_xt = discrete_step_sampler._sample_routing(
        routing_weights=routing_weights,
        xt=xt,
        x0=x0,
        x_noise=x_noise,
        key=key,
    )
    expected = jnp.array([[[3], [1]]])
    chex.assert_trees_all_equal(new_xt, expected)

  def test_stochastic_routing(self):
    # 50/50 stay vs clean — results should vary across seeds.
    routing_weights = discrete_step_sampler.RoutingWeights(
        stay=jnp.array([[[0.5]]]),
        noise=jnp.array([[[0.0]]]),
        clean=jnp.array([[[0.5]]]),
    )
    xt = jnp.array([[[3]]])
    x0 = jnp.array([[[0]]])
    x_noise = jnp.array([[[2]]])

    results = set()
    for seed in range(50):
      new_xt = discrete_step_sampler._sample_routing(
          routing_weights=routing_weights,
          xt=xt,
          x0=x0,
          x_noise=x_noise,
          key=jax.random.PRNGKey(seed),
      )
      results.add(int(new_xt[0, 0, 0]))

    # Should see both stay (3) and clean (0).
    self.assertIn(3, results)
    self.assertIn(0, results)

  def test_routing_constants(self):
    self.assertEqual(discrete_step_sampler.RoutingAction.STAY, 0)
    self.assertEqual(discrete_step_sampler.RoutingAction.NOISE, 1)
    self.assertEqual(discrete_step_sampler.RoutingAction.CLEAN, 2)


class PlannerProtocolTest(absltest.TestCase):

  def test_identity_planner(self):

    class IdentityPlanner:

      def __call__(self, routing_weights, logits, x0, xt, time, next_time, key):
        return routing_weights

    planner = IdentityPlanner()
    routing_weights = discrete_step_sampler.RoutingWeights(
        stay=jnp.array([[[0.2]]]),
        noise=jnp.array([[[0.3]]]),
        clean=jnp.array([[[0.5]]]),
    )
    # dummy args
    logits = jnp.zeros((1, 1, 5))
    x0 = jnp.zeros((1, 1, 1))
    xt = jnp.zeros((1, 1, 1))
    time = jnp.array([1.0])
    next_time = jnp.array([0.5])
    key = jax.random.PRNGKey(0)

    out = planner(routing_weights, logits, x0, xt, time, next_time, key)
    chex.assert_trees_all_equal(out, routing_weights)


class GreedyPlannerTest(absltest.TestCase):

  def test_greedy_planner_budget(self):
    planner = discrete_step_sampler.GreedyPlanner()
    # 1 batch, 4 seq len. Realistic routing: all eligible with stay/noise > 0.
    routing_weights = discrete_step_sampler.RoutingWeights(
        stay=jnp.full((1, 4, 1), 0.3),
        noise=jnp.full((1, 4, 1), 0.2),
        clean=jnp.full((1, 4, 1), 0.5),
    )
    logits = jnp.array(
        [[[10.0, 0.0], [5.0, 0.0], [2.0, 0.0], [1.0, 0.0]]]
    )  # high confidence first
    x0 = jnp.zeros((1, 4, 1), dtype=jnp.int32)
    xt = jnp.ones((1, 4, 1), dtype=jnp.int32)
    time = jnp.array([1.0])
    next_time = jnp.array([0.5])  # frac = 0.5 -> budget = 4 eligible * 0.5 = 2
    key = jax.random.PRNGKey(0)

    out_probs = planner(routing_weights, logits, x0, xt, time, next_time, key)

    # Top 2 positions → force CLEAN (stay=0, noise=0, clean=1).
    # Non-selected → keep original stay/noise, zero out clean.
    expected = discrete_step_sampler.RoutingWeights(
        stay=jnp.array([[[0.0], [0.0], [0.3], [0.3]]]),
        noise=jnp.array([[[0.0], [0.0], [0.2], [0.2]]]),
        clean=jnp.array([[[1.0], [1.0], [0.0], [0.0]]]),
    )
    chex.assert_trees_all_close(out_probs.stay, expected.stay)
    chex.assert_trees_all_close(out_probs.noise, expected.noise)
    chex.assert_trees_all_close(out_probs.clean, expected.clean)

  def test_greedy_planner_eligibility(self):
    planner = discrete_step_sampler.GreedyPlanner()
    # Position 0 is NOT eligible (p_clean = 0), has original stay=1.0.
    routing_weights = discrete_step_sampler.RoutingWeights(
        stay=jnp.array([[[1.0], [0.3], [0.3], [0.3]]]),
        noise=jnp.array([[[0.0], [0.2], [0.2], [0.2]]]),
        clean=jnp.array([[[0.0], [0.5], [0.5], [0.5]]]),
    )
    logits = jnp.array(
        [[[10.0, 0.0], [5.0, 0.0], [2.0, 0.0], [1.0, 0.0]]]
    )  # Pos 0 has highest logit but ineligible
    x0 = jnp.zeros((1, 4, 1), dtype=jnp.int32)
    xt = jnp.ones((1, 4, 1), dtype=jnp.int32)
    time = jnp.array([1.0])
    next_time = jnp.array([0.5])  # frac = 0.5 -> budget = 3 eligible * 0.5 = 1
    key = jax.random.PRNGKey(0)

    out_probs = planner(routing_weights, logits, x0, xt, time, next_time, key)

    # Pos 0 is ineligible (p_clean=0), so num_eligible=3.
    # Budget = 3 * 0.5 = 1 (truncated to int).
    # Top 1 eligible position by confidence: Pos 1 → force CLEAN.
    # Non-selected (Pos 0, 2, 3) → keep original stay/noise, zero clean.
    expected = discrete_step_sampler.RoutingWeights(
        stay=jnp.array([[[1.0], [0.0], [0.3], [0.3]]]),
        noise=jnp.array([[[0.0], [0.0], [0.2], [0.2]]]),
        clean=jnp.array([[[0.0], [1.0], [0.0], [0.0]]]),
    )
    chex.assert_trees_all_close(out_probs.stay, expected.stay)
    chex.assert_trees_all_close(out_probs.noise, expected.noise)
    chex.assert_trees_all_close(out_probs.clean, expected.clean)

  def test_greedy_planner_k_zero(self):
    """When clean weight is small, budget k=0. Keep original stay/noise."""
    planner = discrete_step_sampler.GreedyPlanner()
    routing_weights = discrete_step_sampler.RoutingWeights(
        stay=jnp.full((1, 4, 1), 0.7),
        noise=jnp.full((1, 4, 1), 0.2),
        clean=jnp.full((1, 4, 1), 0.1),
    )
    logits = jnp.array([[[10.0, 0.0], [5.0, 0.0], [2.0, 0.0], [1.0, 0.0]]])
    x0 = jnp.zeros((1, 4, 1), dtype=jnp.int32)
    xt = jnp.ones((1, 4, 1), dtype=jnp.int32)
    time = jnp.array([1.0])
    next_time = jnp.array([1.0])
    key = jax.random.PRNGKey(0)

    out_probs = planner(routing_weights, logits, x0, xt, time, next_time, key)

    # k=0: no positions selected. All keep original stay/noise, clean zeroed.
    expected = discrete_step_sampler.RoutingWeights(
        stay=jnp.full((1, 4, 1), 0.7),
        noise=jnp.full((1, 4, 1), 0.2),
        clean=jnp.zeros((1, 4, 1)),
    )
    chex.assert_trees_all_close(out_probs.stay, expected.stay)
    chex.assert_trees_all_close(out_probs.noise, expected.noise)
    chex.assert_trees_all_close(out_probs.clean, expected.clean)

  def test_greedy_planner_2d_spatial(self):
    """GreedyPlanner must work with 2D spatial data (e.g.

    adjacency matrices).
    """
    planner = discrete_step_sampler.GreedyPlanner()
    # Shape (1, 3, 3, 1): batch=1, spatial=(3,3), vocab_trailing=1.
    routing_weights = discrete_step_sampler.RoutingWeights(
        stay=jnp.full((1, 3, 3, 1), 0.3),
        noise=jnp.full((1, 3, 3, 1), 0.2),
        clean=jnp.full((1, 3, 3, 1), 0.5),
    )
    # 9 positions total, 2 vocab classes.
    # Logits: position (0,0) has highest confidence, then (0,1), etc.
    logits_flat = jnp.array([
        [10.0, 0.0],
        [9.0, 0.0],
        [8.0, 0.0],
        [7.0, 0.0],
        [6.0, 0.0],
        [5.0, 0.0],
        [4.0, 0.0],
        [3.0, 0.0],
        [2.0, 0.0],
    ])
    logits = logits_flat.reshape(1, 3, 3, 2)
    x0 = jnp.zeros((1, 3, 3, 1), dtype=jnp.int32)
    xt = jnp.ones((1, 3, 3, 1), dtype=jnp.int32)
    time = jnp.array([1.0])
    next_time = jnp.array([0.5])  # frac = 0.5 -> budget = 9 * 0.5 = 4
    key = jax.random.PRNGKey(0)

    out = planner(routing_weights, logits, x0, xt, time, next_time, key)

    # Output must have the same spatial shape.
    self.assertEqual(out.stay.shape, (1, 3, 3, 1))
    self.assertEqual(out.noise.shape, (1, 3, 3, 1))
    self.assertEqual(out.clean.shape, (1, 3, 3, 1))

    # Budget k = 4. Top-4 positions (by confidence) → forced CLEAN.
    # Positions (0,0), (0,1), (0,2), (1,0) should be selected.
    selected = out.clean[0, :, :, 0]  # (3, 3)
    num_selected = int(jnp.sum(selected > 0))
    self.assertEqual(num_selected, 4)
    # Selected positions have stay=0, noise=0.
    self.assertEqual(float(jnp.sum(out.stay[out.clean > 0])), 0.0)
    self.assertEqual(float(jnp.sum(out.noise[out.clean > 0])), 0.0)


if __name__ == '__main__':
  absltest.main()
