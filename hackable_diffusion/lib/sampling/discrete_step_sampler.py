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

"""Actual implementation of the sampling steps.

This module proposes various implementations but they all have in common
the core logic:

* An `initialize` function that takes a starting state and returns the
  first step of the diffusion process.
* An `update` function that takes the current state and returns the next step.
* A `finalize` function that takes the last state and returns the final
  state.

At every step, the update function takes the current state and returns the next
state. The update is also in charge of computing other auxiliary informations
such as volatility, drifts, etc.

The `InferenceFn` is also called within the step and converted into the
relevant representation, for instance score, velocity, etc.

This module also introduces the concepts of Routing and Planning for discrete
diffusion:
- Routing: Defines the transition probabilities at each position among the
  three actions: stay at current token, sample from invariant distribution
  (noise),
  or use predicted clean token. Samplers compute these weights based on the
  diffusion posterior.
- Planning: An optional mechanism that intercepts and modifies these routing
  weights before they are applied. For example, a planner might force the
  most confident positions to go clean (budgeting) instead of sampling
  stochastically.

How they interact:
Samplers first compute the default routing weights. If a planner is present,
it transforms these weights (e.g., zeroing out some pathways, forcing others).
Finally, `_sample_routing` samples the next state based on these (possibly
modified) weights.
"""

import dataclasses
import enum
from typing import Protocol

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.corruption import discrete
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.sampling import base
import jax
import jax.numpy as jnp
import kauldron.ktyping as kt

################################################################################
# MARK: Type Aliases
################################################################################

Float = hd_typing.Float

DataArray = hd_typing.DataArray
TargetInfo = hd_typing.TargetInfo
TimeArray = hd_typing.TimeArray
PRNGKey = hd_typing.PRNGKey

DiffusionStep = base.DiffusionStep
StepInfo = base.StepInfo
SamplerStep = base.SamplerStep

CategoricalProcess = discrete.CategoricalProcess
DiscreteSchedule = schedules.DiscreteSchedule

################################################################################
# MARK: Remasking strategy
################################################################################


class RemaskingFn(Protocol):
  """Remasking strategy protocol.

  We follow the original implementation of remasking in
  https://arxiv.org/abs/2503.00307.

  We return the probability of unmasking in the forward process. This is denoted
  σ(t) in https://arxiv.org/abs/2503.00307. The return probability has the same
  shape as the input time. Note that σ(t) in https://arxiv.org/abs/2503.00307 is
  in fact dependent of `s` (the next time) and `t` (the current time).
  """

  def __call__(self, s: TimeArray, t: TimeArray) -> TimeArray:
    """Returns the probability of unmasking."""
    ...


@dataclasses.dataclass(kw_only=True, frozen=True)
class NoRemaskingFn(RemaskingFn):
  """No remasking strategy."""

  @kt.typechecked
  def __call__(self, s: TimeArray, t: TimeArray) -> TimeArray:
    return jnp.zeros_like(s)


@dataclasses.dataclass(kw_only=True, frozen=True)
class MaxCappedRemaskingFn(RemaskingFn):
  """Max-capped remasking strategy.

  This is the original implementation of remasking in
  https://arxiv.org/abs/2503.00307. We follow the implementation of the switch
  function in the paper with the "max-capped" schedule (see Section 4.1). On
  top of this remasking strategy, we consider a switch to turn on and off the
  remasking based on the time, see Section 4.2.

  Attributes:
    schedule: The schedule to use for the corruption process.
    max_cap: The maximum value of the remasking probability.
    switch_min: The minimum value of the switch function.
    switch_max: The maximum value of the switch function.
  """

  schedule: DiscreteSchedule
  max_cap: float = 0.0
  switch_min: float = 0.0
  switch_max: float = 1.0

  def __post_init__(self):
    if self.switch_min > self.switch_max:
      raise ValueError(
          'switch_min must be smaller than switch_max, got'
          f' {self.switch_min} > {self.switch_max}'
      )
    if self.max_cap < 0.0:
      raise ValueError(f'max_cap must be non-negative, got {self.max_cap}')

  @kt.typechecked
  def __call__(self, s: TimeArray, t: TimeArray) -> TimeArray:
    alpha_s = self.schedule.alpha(s)
    alpha_t = self.schedule.alpha(t)
    return jnp.minimum((1.0 - alpha_s) / alpha_t, self.max_cap)


@dataclasses.dataclass(kw_only=True, frozen=True)
class RescaledRemaskingFn(RemaskingFn):
  """Rescaled remasking strategy.

  This is the original implementation of remasking in
  https://arxiv.org/abs/2503.00307. We follow the implementation of the switch
  function in the paper with the "rescaled" schedule (see Section 4.1). On top
  of this remasking strategy, we consider a switch to turn on and off the
  remasking based on the time, see Section 4.2.

  Attributes:
    schedule: The schedule to use for the corruption process.
    rescale_factor: The rescale factor to apply to the remasking probability.
    switch_min: The minimum value of the switch function.
    switch_max: The maximum value of the switch function.
  """

  def __post_init__(self):
    if self.switch_min > self.switch_max:
      raise ValueError(
          'switch_min must be smaller than switch_max, got'
          f' {self.switch_min} > {self.switch_max}'
      )
    if self.rescale_factor <= 0.0:
      raise ValueError(
          f'rescale_factor must be positive, got {self.rescale_factor}'
      )
    if self.rescale_factor >= 1.0:
      raise ValueError(
          f'rescale_factor must be smaller than 1.0, got {self.rescale_factor}'
      )

  schedule: DiscreteSchedule
  rescale_factor: float
  switch_min: float = 0.0
  switch_max: float = 1.0

  @kt.typechecked
  def __call__(self, s: TimeArray, t: TimeArray) -> TimeArray:
    alpha_s = self.schedule.alpha(s)
    alpha_t = self.schedule.alpha(t)
    return self.rescale_factor * jnp.minimum((1.0 - alpha_s) / alpha_t, 1.0)


################################################################################
# MARK: Corrupted mask function
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class CorruptedMaskFn(Protocol):
  """Corrupted mask function protocol.

  This function takes the current xt, i.e., the "noisy" data and returns a mask
  which indicates which regions are corrupted. The mask is True if the region is
  corrupted and False otherwise. In the mask setting, this simply corresponds to
  identifying the tokens which have the mask value. More complex masking schemes
  can be defined as in https://arxiv.org/abs/2410.06264.
  """

  def __call__(self, xt: DataArray) -> DataArray:
    """Returns the corrupted mask."""
    ...


@dataclasses.dataclass(kw_only=True, frozen=True)
class AllCorruptedMaskFn(CorruptedMaskFn):
  """Assume all tokens are corrupted."""

  @kt.typechecked
  def __call__(self, xt: DataArray) -> DataArray:
    return jnp.ones_like(xt, dtype=jnp.bool_)


@dataclasses.dataclass(kw_only=True, frozen=True)
class MaskValueCorruptedMaskFn(CorruptedMaskFn):
  """Corrupted mask function based on the mask value.

  Note that this function only makes sense for masking processes.
  """

  process: CategoricalProcess

  def __post_init__(self):
    """MaskValueCorruptedMaskFn only supports masking processes."""
    if not self.process.is_masking:
      raise ValueError(
          'MaskValueCorruptedMaskFn only supports masking processes.'
      )

  @kt.typechecked
  def __call__(self, xt: DataArray) -> DataArray:
    mask_value = self.process.process_num_categories - 1
    return xt == mask_value


################################################################################
# MARK: Routing and planning
################################################################################

# Almost all discrete samplers compute a 3-way routing for each token position:
#   0 = stay at current token (xt)
#   1 = sample from invariant distribution (noise)
#   2 = use predicted clean token (x0)
#
# The routing weights (p_stay, p_noise, p_clean) are computed by each
# sampler and applied via the shared `_sample_routing` helper.
# IntegratedDiscreteDDIMStep is an exception as it integrates the routing
# probabilities into the update rule.


class RoutingAction(enum.IntEnum):
  STAY = 0
  NOISE = 1
  CLEAN = 2


@dataclasses.dataclass(frozen=True, kw_only=True)
class RoutingWeights:
  stay: Float['...']
  noise: Float['...']
  clean: Float['...']


def _sample_routing(
    *,
    routing_weights: RoutingWeights,
    xt: DataArray,
    x0: DataArray,
    x_noise: DataArray,
    key: PRNGKey,
) -> DataArray:
  """Apply 3-way routing to construct the next state.

  3-way routing determines the next state by sampling from a mixture of three
  possible actions at each position:
  1. STAY: Keep the current token `xt`.
  2. NOISE: Sample a new token from the invariant distribution `x_noise`.
  3. CLEAN: Use the predicted clean token `x0`.

  The computation operates by:
  1. Concatenating the weights for stay, noise, and clean along the last axis.
  2. Sampling an action index (0, 1, or 2) for each position using
  `jax.random.categorical`.
  3. Selecting the corresponding token (xt, x_noise, or x0) based on the sampled
  action.

  Args:
    routing_weights: Routing weights containing stay, noise, and clean arrays.
    xt: Current state. Shape ``(*, 1)``.
    x0: Predicted clean state. Shape ``(*, 1)``.
    x_noise: Sample from invariant distribution. Shape ``(*, 1)``.
    key: Random key for categorical sampling.

  Returns:
    The new state ``new_xt``. Shape ``(*, 1)``.
  """
  weights = jnp.concatenate(
      [routing_weights.stay, routing_weights.noise, routing_weights.clean],
      axis=-1,
  )
  action = jax.random.categorical(
      key=key, logits=jnp.log(jnp.maximum(weights, 1e-12))
  )
  new_xt = jnp.where(
      action[..., None] == RoutingAction.CLEAN,
      x0,
      jnp.where(action[..., None] == RoutingAction.NOISE, x_noise, xt),
  )
  return new_xt


def _generate_candidates(
    corruption_process: CategoricalProcess,
    prediction: TargetInfo,
    xt: DataArray,
    time_bcast: TimeArray,
    key: PRNGKey,
    temperature: float,
) -> tuple[DataArray, DataArray, Float['... M']]:
  """Generate candidate x0, x_noise samples and logits."""
  logits = corruption_process.convert_predictions(prediction, xt, time_bcast)[
      'logits'
  ]
  logits = logits / temperature

  x0_key, noise_key = jax.random.split(key)
  x0 = jax.random.categorical(key=x0_key, logits=logits)[..., None]
  x_noise = corruption_process.sample_from_invariant(noise_key, data_spec=xt)

  return x0, x_noise, logits


class RoutingStrategy(Protocol):
  """Protocol for transforming routing weights.

  A planner takes the routing weights computed by a sampler and
  optionally transforms them before they are applied via ``_sample_routing``.
  This allows injecting different selection strategies (e.g. greedy top-k)
  without modifying the sampler logic.

  When no planner is used (``planner=None``), the routing weights are
  applied as-is via stochastic categorical sampling.
  """

  def __call__(
      self,
      routing_weights: RoutingWeights,
      logits: Float['... M'],
      x0: DataArray,
      xt: DataArray,
      time: TimeArray,
      next_time: TimeArray,
      key: PRNGKey,
  ) -> RoutingWeights:
    """Transforms routing weights.

    Args:
      routing_weights: Per-position routing weights.
      logits: Model logits ``(*, M)``.
      x0: Sampled clean token ``(*, 1)``.
      xt: Current state ``(*, 1)``.
      time: Current diffusion time.
      next_time: Next diffusion time.
      key: Random key.

    Returns:
      Transformed routing weights.
    """
    ...


################################################################################
# MARK: UnMasking Step
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class UnMaskingStep(SamplerStep):
  """Unmasking step following https://arxiv.org/abs/2406.04329.

  This sampler uses the 3-way routing representation. For each token position
  we compute the probabilities of three actions:
    - STAY: keep the current token.
    - NOISE: sample from the invariant distribution (remasking).
    - CLEAN: use the predicted clean token x0.

  For masked tokens:
    p_clean = (alpha_s - (1 - p_st) * alpha_t) / (1 - alpha_t)
    p_noise = p_st
    p_stay = 1 - p_clean - p_noise
  For unmasked tokens:
    p_clean = 0
    p_noise = p_st
    p_stay = 1 - p_st
  where p_st is the remasking rate.

  Attributes:
    corruption_process: The corruption process to use.
    remasking_fn: The remasking function to use, see
      https://arxiv.org/abs/2503.00307v1. This is optional with the default
        being no remasking.
    corruption_mask_fn: The corrupted mask function to use. This is optional
      with the default being all tokens corrupted.
    temperature: The temperature to use. This is optional with the default being
      a temperature of 1.0.
  """

  corruption_process: CategoricalProcess
  planner: RoutingStrategy | None = None
  remasking_fn: RemaskingFn = NoRemaskingFn()
  corruption_mask_fn: CorruptedMaskFn = AllCorruptedMaskFn()
  temperature: float = 1.0
  logits_dtype: jnp.dtype = jnp.float32

  def __post_init__(self):
    """UnMaskingStep only supports masking processes.

    We refer to update for more details.
    """
    if not self.corruption_process.is_masking:
      raise ValueError('UnMaskingStep only supports masking processes.')

  @kt.typechecked
  def initialize(
      self,
      initial_noise: DataArray,
      initial_step_info: StepInfo,
  ) -> DiffusionStep:

    init_logits = jnp.repeat(
        initial_noise, self.corruption_process.num_categories, axis=-1
    )
    init_logits = jnp.zeros_like(init_logits, dtype=self.logits_dtype)

    return DiffusionStep(
        xt=initial_noise,
        step_info=initial_step_info,
        aux={
            'logits': init_logits,
            # `logits` need to be passed in `aux` dictionary to a performance
            # bug when using TPU. Needs to be investigated.
        },
    )

  @kt.typechecked
  def update(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
  ) -> DiffusionStep:

    current_step_info = current_step.step_info
    xt = current_step.xt

    unused_mask = xt == self.corruption_process.unused_token
    # The mask is True if the token is unused.

    time = current_step_info.time
    next_time = next_step_info.time
    time_bcast = utils.bcast_right(time, xt.ndim)
    next_time_bcast = utils.bcast_right(next_time, xt.ndim)
    key = next_step_info.rng

    # Get model predictions and candidates
    _, candidate_key, plan_key, route_key = jax.random.split(key, 4)
    x0, x_noise, logits = _generate_candidates(
        self.corruption_process,
        prediction,
        xt,
        time_bcast,
        candidate_key,
        self.temperature,
    )

    currently_masked = self.corruption_mask_fn(xt)  # (bsz, seq_len, 1)

    # Denoising rates
    alpha_s = self.corruption_process.schedule.alpha(next_time_bcast)
    alpha_t = self.corruption_process.schedule.alpha(time_bcast)

    # Routing decomposition logic
    # See docstring for formulae and https://arxiv.org/abs/2503.00307 for
    # details.

    p_st = self.remasking_fn(s=next_time_bcast, t=time_bcast)

    p_clean_masked = (alpha_s - (1.0 - p_st) * alpha_t) / (1.0 - alpha_t)
    p_noise_masked = p_st
    p_stay_masked = 1.0 - p_clean_masked - p_noise_masked
    # Denoising probability following https://arxiv.org/abs/2503.00307
    # If no remasking (https://arxiv.org/abs/2503.00307), p_st = 0,
    # so p_clean = (alpha_s - alpha_t) / (1.0 - alpha_t)
    # These are the routing weights for masked positions xt.
    # With prob p_clean, we replace xt with the predicted token x0.
    # With prob p_noise, we replace xt with the invariant token x_noise.
    # With prob p_stay, we keep the current token xt.

    # Routing weights for unmasked tokens:
    p_stay_unmasked = 1.0 - p_st
    p_noise_unmasked = p_st
    p_clean_unmasked = jnp.zeros_like(p_st)
    # Same as above, but for unmasked positions.
    # Note that if p_st = 0, then p_noise = 0, and p_stay = 1, which means
    # that unmasked tokens are never remasked.

    # Combine based on masking state
    # See https://arxiv.org/abs/2503.00307 for an example of the combination of
    # probabilities for masked and unmasked tokens.
    p_stay = jnp.where(currently_masked, p_stay_masked, p_stay_unmasked)
    p_noise = jnp.where(currently_masked, p_noise_masked, p_noise_unmasked)
    p_clean = jnp.where(currently_masked, p_clean_masked, p_clean_unmasked)

    routing_weights = RoutingWeights(stay=p_stay, noise=p_noise, clean=p_clean)
    # (bsz, seq_len, 3)

    # Apply planner transformation (if any)
    if self.planner:
      routing_weights = self.planner(
          routing_weights, logits, x0, xt, time, next_time, plan_key
      )

    # xt ~ p(x_s|x_0, x_t) (with optional remasking and planning)
    new_xt = _sample_routing(
        routing_weights=routing_weights,
        xt=xt,
        x0=x0,
        x_noise=x_noise,
        key=route_key,
    )

    # This is the new state after sampling using the routing weights.

    new_xt = self.corruption_process.post_corruption_fn(new_xt)

    # Replace the unused tokens with the unused_token.
    new_xt = jnp.where(
        unused_mask, self.corruption_process.unused_token, new_xt
    )

    return DiffusionStep(
        xt=new_xt,
        step_info=next_step_info,
        aux={'logits': logits},
    )
    # `logits` need to be passed in `aux` dictionary to a performance
    # bug when using TPU. Needs to be investigated.

  @kt.typechecked
  def finalize(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      last_step_info: StepInfo,
  ) -> DiffusionStep:
    return self.update(
        prediction,
        current_step,
        last_step_info,
    )


################################################################################
# MARK: DDIM Step
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class DiscreteDDIMStep(SamplerStep):
  """Discrete version of the DDIM step.

  This sampler is inspired by the discrete sampler of "Structured Denoising
  Diffusion Models in Discrete State-Spaces" (known as D3PM, see
  https://arxiv.org/abs/2107.03006).

  This sampler uses the 3-way routing representation. Given the forward process
  with density p(x_t|x_0), we decompose the reverse posterior
  p(x_s|x_t, x_0) into three components:

    p(x_s|x_t,x_0) = p_stay * δ_{x_t}(x_s) + p_noise * π(x_s)
                      + p_clean * δ_{x_0}(x_s)

  where:
    - p_stay: probability of staying at x_t
    - p_noise: probability of jumping to invariant noise
    - p_clean: probability of jumping to the predicted x_0

  **Derivation.** Recall that for the forward process:

    p(x_t|x_s) = r * δ_{x_t}(x_s) + (1 - r) * π(x_t)
    p(x_s|x_0) = α_s * δ_{x_0}(x_s) + (1 - α_s) * π(x_s)

  where r = α_t/α_s. The posterior is proportional to their product:

    p(x_s|x_t,x_0) ∝ p(x_t|x_s) * p(x_s|x_0)

  Expanding gives four cross-terms:

    (T1)  r * α_s         * δ_{x_t}(x_s) * δ_{x_0}(x_s)
    (T2)  r * (1-α_s)     * δ_{x_t}(x_s) * π(x_s)   = r*(1-α_s)*π(x_t) * δ_{x_t}
    (T3)  (1-r) * α_s     * π(x_t)       * δ_{x_0}(x_s)
    (T4)  (1-r) * (1-α_s) * π(x_t)       * π(x_s)

  Collecting by routing outcome:
    - p_stay  ∝ r * (1-α_s) * π(x_t)     [T2: δ_{x_t} · π gives π(x_t)*δ_{x_t}]
    - p_noise ∝ (1-r) * (1-α_s) * π(x_t) [T4: π(x_t) · π(x_s)]
    - p_clean ∝ (1-r) * α_s * π(x_t)     [T3: π(x_t) · δ_{x_0}]

  **Handling x_0 = x_t.** When x_0 = x_t, routing to CLEAN produces the same
  output as STAY (both emit x_t). We therefore merge all such mass into p_stay:

    1. T1 contributes r * α_s to p_stay (the fourth cross-term
       δ_{x_t} * δ_{x_0}, which is non-zero only when x_0 = x_t).
    2. p_clean is added to p_stay and then zeroed out, since the CLEAN action
       would be a no-op.

  This ensures p_clean = 0 whenever x_0 = x_t, which is important for
  planners that use p_clean > 0 as an eligibility signal (e.g. GreedyPlanner).

  Note: when π = δ_MASK (masking process) and x_t = MASK, this reduces to:
    P(unmask) = (α_s - α_t) / (1 - α_t),
  which coincides with the UnMaskingStep formula (without remasking).
  """

  corruption_process: CategoricalProcess
  planner: RoutingStrategy | None = None
  temperature: float = 1.0
  logits_dtype: jnp.dtype = jnp.float32

  def __post_init__(self):
    """DiscreteDDIMStep does not support masking processes.

    We refer to update for more details.
    """
    if self.corruption_process.is_masking:
      raise ValueError('DiscreteDDIMStep does not support masking processes.')
    if 0.0 in self.corruption_process.invariant_probs:
      raise ValueError(
          'DiscreteDDIMStep does not support invariant probabilities'
          ' with 0.0 probability mass for any element.'
      )

  @kt.typechecked
  def initialize(
      self,
      initial_noise: DataArray,
      initial_step_info: StepInfo,
  ) -> DiffusionStep:

    init_logits = jnp.repeat(
        initial_noise, self.corruption_process.num_categories, axis=-1
    )
    init_logits = jnp.zeros_like(init_logits, dtype=self.logits_dtype)

    return DiffusionStep(
        xt=initial_noise,
        step_info=initial_step_info,
        aux={'logits': init_logits},
    )
    # `logits` need to be passed in `aux` dictionary to a performance
    # bug when using TPU. Needs to be investigated.

  @kt.typechecked
  def update(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
  ) -> DiffusionStep:

    current_step_info = current_step.step_info
    xt = current_step.xt

    unused_mask = xt == self.corruption_process.unused_token

    time = current_step_info.time
    next_time = next_step_info.time
    time_bcast = utils.bcast_right(time, xt.ndim)
    next_time_bcast = utils.bcast_right(next_time, xt.ndim)
    key = next_step_info.rng

    # Get model predictions and candidates
    _, candidate_key, plan_key, route_key = jax.random.split(key, 4)
    x0, x_noise, logits = _generate_candidates(
        self.corruption_process,
        prediction,
        xt,
        time_bcast,
        candidate_key,
        self.temperature,
    )

    # Schedule
    alpha_s = self.corruption_process.schedule.alpha(next_time_bcast)
    alpha_t = self.corruption_process.schedule.alpha(time_bcast)
    ratio = alpha_t / alpha_s
    # (bsz, *seq_len, 1)

    # Routing weights (unnormalized).
    # See the class docstring for the full derivation of terms T1–T4.
    pi_xt = self.corruption_process.invariant_probs_vec[xt[..., 0]][..., None]

    # T2 → stay, T4 → noise, T3 → clean
    p_stay = ratio * (1.0 - alpha_s) * pi_xt
    p_noise = (1.0 - ratio) * (1.0 - alpha_s) * pi_xt
    p_clean = (1.0 - ratio) * alpha_s * pi_xt

    # When x_0 = x_t, routing to CLEAN produces the same output as STAY.
    # Merge T1 (r * α_s) and p_clean into p_stay, and zero out p_clean.
    # This ensures planners see p_clean = 0 for no-op positions.
    x0_eq_xt = (x0 == xt).astype(jnp.float32)
    p_stay = p_stay + x0_eq_xt * (ratio * alpha_s + p_clean)
    p_clean = (1.0 - x0_eq_xt) * p_clean

    routing_weights = RoutingWeights(stay=p_stay, noise=p_noise, clean=p_clean)
    # (bsz, *seq_len, 3)

    # Apply planner transformation (if any)
    if self.planner:
      routing_weights = self.planner(
          routing_weights, logits, x0, xt, time, next_time, plan_key
      )

    # xt ~ p(x_s|x_0, x_t)
    # This is the new state after sampling using the routing weights.
    new_xt = _sample_routing(
        routing_weights=routing_weights,
        xt=xt,
        x0=x0,
        x_noise=x_noise,
        key=route_key,
    )

    new_xt = self.corruption_process.post_corruption_fn(new_xt)

    # Replace the unused tokens with the unused_token.
    new_xt = jnp.where(
        unused_mask, self.corruption_process.unused_token, new_xt
    )

    return DiffusionStep(
        xt=new_xt,
        step_info=next_step_info,
        aux={'logits': logits},
    )

  @kt.typechecked
  def finalize(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      last_step_info: StepInfo,
  ) -> DiffusionStep:
    return self.update(
        prediction,
        current_step,
        last_step_info,
    )


################################################################################
# MARK: Discrete Flow Matching Step
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class DiscreteFlowMatchingStep(SamplerStep):
  """Discrete Flow Matching step following https://arxiv.org/abs/2407.15595.

  This sampler uses the 3-way routing representation. The update rule
  decomposes naturally into:

    p(x_s) = p_stay * δ_{x_t} + p_up * p_x0 + p_down * π

  where:
    - p_stay = 1 - p_up - p_down
    - p_up = (α_s - α_t) / (1 - α_t) * (1 + stoch_coeff)
    - p_down = (α_s - α_t) / α_t * stoch_coeff

  Attributes:
    corruption_process: The corruption process to use.
    temperature: The temperature to use.
    stoch_coeff: The stochasticity coefficient (default 0.0). Higher values
      introduce more noise during the denoising process.
  """

  corruption_process: CategoricalProcess
  planner: RoutingStrategy | None = None
  temperature: float = 1.0
  stoch_coeff: float = 0.0

  @kt.typechecked
  def initialize(
      self,
      initial_noise: DataArray,
      initial_step_info: StepInfo,
  ) -> DiffusionStep:

    init_logits = jnp.repeat(
        initial_noise, self.corruption_process.num_categories, axis=-1
    )
    init_logits = jnp.zeros_like(init_logits, dtype=jnp.float32) - jnp.inf

    return DiffusionStep(
        xt=initial_noise,
        step_info=initial_step_info,
        aux={'logits': init_logits},
    )

  @kt.typechecked
  def update(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
  ) -> DiffusionStep:

    current_step_info = current_step.step_info
    xt = current_step.xt

    unused_mask = xt == self.corruption_process.unused_token

    time = current_step_info.time
    next_time = next_step_info.time
    time_bcast = utils.bcast_right(time, xt.ndim)
    next_time_bcast = utils.bcast_right(next_time, xt.ndim)
    key = next_step_info.rng

    # Get model predictions and candidates
    _, candidate_key, plan_key, route_key = jax.random.split(key, 4)
    x0, x_noise, logits = _generate_candidates(
        self.corruption_process,
        prediction,
        xt,
        time_bcast,
        candidate_key,
        self.temperature,
    )

    # Denoising rates
    alpha_s = self.corruption_process.schedule.alpha(next_time_bcast)
    alpha_t = self.corruption_process.schedule.alpha(time_bcast)

    prob_up = (
        (alpha_s - alpha_t)
        / jnp.maximum(1.0 - alpha_t, 1e-12)
        * (1.0 + self.stoch_coeff)
    )
    prob_down = (
        (alpha_s - alpha_t) / jnp.maximum(alpha_t, 1e-12) * self.stoch_coeff
    )

    # Clip and rescale to ensure valid probabilities
    raw_p_up = jnp.maximum(prob_up, 0.0)
    raw_p_down = jnp.maximum(prob_down, 0.0)
    sum_jumps = raw_p_up + raw_p_down
    scale_factor = jnp.maximum(1.0, sum_jumps)

    # Compute the probabilities for the three routing options.
    # This is computed according to https://arxiv.org/abs/2407.15595.
    p_clean = raw_p_up / scale_factor
    p_noise = raw_p_down / scale_factor
    p_stay = 1.0 - p_clean - p_noise

    routing_weights = RoutingWeights(stay=p_stay, noise=p_noise, clean=p_clean)
    # (bsz, *seq_len, 3)

    # Apply planner transformation (if any)
    if self.planner:
      routing_weights = self.planner(
          routing_weights, logits, x0, xt, time, next_time, plan_key
      )

    # xt ~ p(x_s|x_0, x_t)
    # This is the new state after sampling using the routing weights.
    new_xt = _sample_routing(
        routing_weights=routing_weights,
        xt=xt,
        x0=x0,
        x_noise=x_noise,
        key=route_key,
    )
    new_xt = self.corruption_process.post_corruption_fn(new_xt)

    # Replace the unused tokens with the unused_token.
    new_xt = jnp.where(
        unused_mask, self.corruption_process.unused_token, new_xt
    )

    return DiffusionStep(
        xt=new_xt,
        step_info=next_step_info,
        aux={'logits': logits},
    )

  @kt.typechecked
  def finalize(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      last_step_info: StepInfo,
  ) -> DiffusionStep:
    return self.update(
        prediction,
        current_step,
        last_step_info,
    )


################################################################################
# MARK: Integrated DDIM Step
################################################################################
# Note: IntegratedDiscreteDDIMStep does NOT fit the 3-way routing scheme
# because it marginalizes over x_0 rather than sampling a single x_0.
# It is kept as-is with direct categorical sampling.
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class IntegratedDiscreteDDIMStep(SamplerStep):
  """Integrated discrete version of the DDIM step.

  This sampler is inspired by the discrete sampler of "Structured Denoising
  Diffusion Models in Discrete State-Spaces" (known as D3PM, see
  https://arxiv.org/abs/2107.03006).

  Remember that the `DiscreteDDIMStep` does the following.
  Given the forward process with density p(x_t|x_0) it computes the reverse
  process by first sampling from p(x_0|x_t) to obtain x_0.

  Then it samples x_s (for s < t) using the following formula:

    p(x_s|x_t,x_0) ∝ p(x_s|x_0) * p(x_t|x_s) (1)

  In order to compute (1) we recall that for any s, t such that s < t we have:

    p(x_t|x_s) = (α_t/α_s) * δ_{x_s}(x_t) + (1 - α_t/α_s) * π(x_t) (1)

  The computation of the probability happens in the logits space.

  In the `IntegratedDiscreteDDIMStep`, instead of sampling from p(x_0|x_t) and
  then sampling x_s using (1), we directly sample x_s using a formula that
  integrates over the (unknown) samples of x_0.

  In particular, we use the following formula:

    p(x_s|x_t) = p(x_t|x_s) * sum_{x_0} (p(x_0|x_t) / p(x_t|x_0)) p(x_s|x_0)
    (2)

  Denoting w(x_0, x_t) =  p(x_0|x_t) / p(x_t|x_0) and W(x_t) = sum_{x_0} w(x_0,
  x_t) we have:

    p(x_s|x_0) = α_s * w(x_s, x_t) + (1 - α_s) * W(X_t) * π(x_s) (3)
  """

  corruption_process: CategoricalProcess
  temperature: float = 1.0
  logits_dtype: jnp.dtype = jnp.float32

  def __post_init__(self):
    """IntegratedDiscreteDDIMStep does not support masking processes.

    We refer to update for more details.
    """
    if self.corruption_process.is_masking:
      raise ValueError(
          'IntegratedDiscreteDDIMStep does not support masking processes.'
      )
    if 0.0 in self.corruption_process.invariant_probs:
      raise ValueError(
          'IntegratedDiscreteDDIMStep does not support invariant probabilities'
          ' with 0.0 probability mass for any element.'
      )

  @kt.typechecked
  def initialize(
      self,
      initial_noise: DataArray,
      initial_step_info: StepInfo,
  ) -> DiffusionStep:

    init_logits = jnp.repeat(
        initial_noise, self.corruption_process.num_categories, axis=-1
    )
    init_logits = jnp.zeros_like(init_logits, dtype=self.logits_dtype)

    return DiffusionStep(
        xt=initial_noise,
        step_info=initial_step_info,
        aux={'logits': init_logits},
    )
    # `logits` need to be passed in `aux` dictionary to a performance
    # bug when using TPU. Needs to be investigated.

  @kt.typechecked
  def update(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
  ) -> DiffusionStep:

    xt = current_step.xt
    unused_mask = xt == self.corruption_process.unused_token

    time = utils.bcast_right(current_step.step_info.time, xt.ndim)
    next_time = utils.bcast_right(next_step_info.time, xt.ndim)
    key = next_step_info.rng

    # Extract predictions.
    logits = self.corruption_process.convert_predictions(prediction, xt, time)[
        'logits'
    ]
    logits = logits / self.temperature
    p_x0 = jax.nn.softmax(logits, axis=-1)
    # (bsz, *seq_len, M)

    # One-hot encoding for the current state
    xt_oh = jax.nn.one_hot(
        xt[..., 0], num_classes=self.corruption_process.process_num_categories
    )
    # (bsz, *seq_len, M)

    # Calculate schedule alphas.
    alpha_s = self.corruption_process.schedule.alpha(next_time)
    alpha_t = self.corruption_process.schedule.alpha(time)
    alpha_s = jnp.broadcast_to(alpha_s, xt_oh.shape)
    alpha_t = jnp.broadcast_to(alpha_t, xt_oh.shape)
    ratio = alpha_t / alpha_s
    # (bsz, *seq_len, M)

    # Extract invariant probabilities.
    pi = self.corruption_process.invariant_probs_vec
    pi_xt = pi[xt[..., 0]][..., None]  # The prior prob of the current token
    # (bsz, *seq_len, 1)

    # Calculate q(x_t | x_s).
    q_xt_given_xs = ratio * xt_oh + (1.0 - ratio) * pi_xt
    # (bsz, *seq_len, M)

    # Calculate q(x_t | x_0)'
    q_xt_given_x0 = alpha_t * xt_oh + (1.0 - alpha_t) * pi_xt
    # (bsz, *seq_len, M)

    # Calculate integration weights: W(x_0) = p(x_0 | x_t) / q(x_t | x_0).
    w_x0 = p_x0 / jnp.clip(q_xt_given_x0, min=1e-12)
    # (bsz, *seq_len, M)
    sum_w = jnp.sum(w_x0, axis=-1, keepdims=True)
    # (bsz, *seq_len, 1)

    # Compute Sum_{x_0} W(x_0) * q(x_s | x_0).
    expected_xs_given_x0 = alpha_s * w_x0 + (1.0 - alpha_s) * pi * sum_w
    # (bsz, *seq_len, M)

    # Final marginalized probability p(x_s | x_t).
    p_xs = q_xt_given_xs * expected_xs_given_x0
    # (bsz, *seq_len, M)

    # Convert to logits and sample.
    total_logit = jnp.log(jnp.clip(p_xs, min=1e-12))
    new_xt = jax.random.categorical(key=key, logits=total_logit)[..., None]
    new_xt = self.corruption_process.post_corruption_fn(new_xt)

    # Replace the unused tokens with the unused_token.
    new_xt = jnp.where(
        unused_mask, self.corruption_process.unused_token, new_xt
    )

    return DiffusionStep(
        xt=new_xt,
        step_info=next_step_info,
        aux={'logits': logits},
    )
    # `logits` need to be passed in `aux` dictionary to a performance
    # bug when using TPU. Needs to be investigated.

  @kt.typechecked
  def finalize(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      last_step_info: StepInfo,
  ) -> DiffusionStep:
    return self.update(
        prediction,
        current_step,
        last_step_info,
    )
