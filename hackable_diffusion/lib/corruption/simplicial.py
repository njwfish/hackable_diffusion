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

"""Simplicial noise processes."""

from __future__ import annotations

import dataclasses
import enum
from typing import Protocol, Sequence

from hackable_diffusion.lib import fast_random
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import jax_helpers
from hackable_diffusion.lib.corruption import base
from hackable_diffusion.lib.corruption import schedules
import jax
import jax.numpy as jnp
import kauldron.ktyping as kt

################################################################################
# MARK: Constants
################################################################################

UNUSED_TOKEN = -1
LOGITS_INF = 1e9

################################################################################
# MARK: Type Aliases
################################################################################

Float = hd_typing.Float
PRNGKey = hd_typing.PRNGKey

DataArray = hd_typing.DataArray
TargetInfo = hd_typing.TargetInfo
TimeArray = hd_typing.TimeArray

CorruptionProcess = base.CorruptionProcess
SimplicialSchedule = schedules.SimplicialSchedule

################################################################################
# MARK: Enums
################################################################################


class SamplingPrecisionMode(enum.StrEnum):
  """Sampling precision mode.

  See
  https://docs.jax.dev/en/latest/_autosummary/jax.random.choice.html#jax.random.choice
  for more details about how `mode` is used in random samplers.
  """

  HIGH = 'high'
  LOW = 'low'


################################################################################
# MARK: Post-Corruption Functions
################################################################################


class SimplicialPostCorruptionFn(Protocol):
  """Post-corruption function protocol for simplicial (log-prob) data.

  The purpose of a post-corruption function is to project the noisy
  log-probability array onto a constrained subspace after each corruption or
  reverse-diffusion step.  The canonical example is symmetrising a
  simplex-valued edge-attribute matrix so that edge (i, j) and edge (j, i)
  share the same categorical distribution — the simplicial analogue of
  ``SymmetricPostCorruptionFn`` from the discrete process.

  Unlike the discrete variant, the input and output here are log-probability
  arrays of shape (*batch, num_categories), not integer token arrays.
  """

  def __call__(self, log_x: DataArray) -> DataArray:
    """Project the log-probability array."""
    ...


class IdentitySimplicialPostCorruptionFn(SimplicialPostCorruptionFn):
  """Identity post-corruption function (no projection)."""

  def __call__(self, log_x: DataArray) -> DataArray:
    return log_x


class SymmetricSimplicialPostCorruptionFn(SimplicialPostCorruptionFn):
  """Symmetric post-corruption function for simplex-valued edge matrices.

  This is the simplicial analogue of ``SymmetricPostCorruptionFn`` from the
  discrete process, used in DiGress https://arxiv.org/abs/2209.14734 in order
  to noise the adjacency graph.  This function also zeroes out the diagonal
  entries, thereby removing any self-loop.

  Input shape must be (batch, N, N, num_categories) where N is the number of
  nodes.
  """

  def __call__(self, log_x: DataArray) -> DataArray:
    """Project the log-probability array to be symmetric."""

    if log_x.ndim != 4:
      raise ValueError(f'Expected 4D (B, N, N, K) input, got {log_x.ndim=}.')
    if log_x.shape[1] != log_x.shape[2]:
      raise ValueError(
          f'Spatial dimensions must be equal, got {log_x.shape[1]=} and'
          f' {log_x.shape[2]=}.'
      )

    _, n, _, num_categories = log_x.shape

    # Move the category axis before the spatial axes so that jnp.triu
    # operates on the last two (N, N) dimensions: (B, N, N, K) -> (B, K, N, N).
    # Doing so, the operations are identical to the discrete case,
    # see SymmetricPostCorruptionFn.
    log_x_bknn = jnp.moveaxis(log_x, -1, 1)
    log_x_tri = jnp.triu(log_x_bknn, k=1)
    log_x_sym = log_x_tri + jnp.transpose(log_x_tri, axes=(0, 1, 3, 2))
    log_y = jnp.moveaxis(log_x_sym, 1, -1)
    # Zero out diagonal entries: set to the no-edge log-probability vector
    # (all mass on category 0).
    no_edge_log = jnp.full((num_categories,), -LOGITS_INF, dtype=log_x.dtype)
    no_edge_log = no_edge_log.at[0].set(0.0)
    diag_mask = jnp.eye(n, dtype=jnp.bool_)[None, :, :, None]
    log_y = jnp.where(diag_mask, no_edge_log, log_y)
    return log_y


################################################################################
# MARK: SimplicialProcess
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class SimplicialProcess(CorruptionProcess):
  """Simplicial noise processes that corrupt towards a Dirichlet distribution.

  We are mostly using two special cases of this process:
  - masking: invariant_probs = (0.0,) * K + (1.0,)
  - uniform: invariant_probs = (1.0 / K,) * K
  where K is the number of categories.
  In that case denoting π this invariant probability distribution, the
  corruption process targets Dir(τ π) where τ is a temperature parameter and Dir
  is the Dirichlet distribution.
  For each 0 <= t <= 1, the forward process is given by:

    p_{t|0}(X(t)|X(0)) = Dir(τ(h(t) δ(X(0)) + π)) ,

  where h(t) is a function of the time t and δ(X(0)) is the one-hot encoding of
  X(0). X(t) represents the corrupted data which is a sample from the Dirichlet
  distribution p_{t|0} and is therefore a categorical distribution.

  The function h(t) is given by the formula:

    h(t) = α(t) / (1 - α(t))

  In particular, we have that h(0) = +inf and h(1) = 0.

  NOTE: We perform the corruption in log-space for numerical stability.

  Attributes:
    schedule: The schedule to use for the corruption process.
    invariant_probs: The invariant probability distribution of the process. At
      time one, the process will corrupt towards the distribution defined by
      invariant_probs.
    num_categories: The number of categories in the distribution. Note that this
      might be different from the length of invariant_probs, which might contain
      K+1 elements in the case of masking.
    unused_token: If a token is unused then it should have this value. Note that
      we require that this unused_token is not in the range of the vocabulary,
      i,e., unused_token < 0 or unused_token >= len(invariant_probs) (which is
      the same as process_num_categories).
    temperature: The temperature parameter of the Dirichlet distribution. This
      parameter controls the sharpness of the distribution.
    mode: The mode to use in `jax.random.choice` and `jax.random.bernoulli`. Can
      be set to "high" or "low" for how many bits to use in the Gumbel sampler.
      See https://jax.readthedocs.io/en/latest/jax.random.html#jax.random.choice
      for more information.
    safety_epsilon: A small constant added to the denominator of the h-function
      to avoid division by zero.
    post_corruption_fn: A projection applied to the corrupted log-prob array
      after each forward-corruption step and after each reverse-diffusion step.
      Used to enforce structural constraints such as symmetry of edge-attribute
      matrices.  Defaults to the identity (no projection).
  """

  schedule: SimplicialSchedule
  invariant_probs: Sequence[float]
  num_categories: int
  unused_token: int = UNUSED_TOKEN
  temperature: float = 1.0
  mode: SamplingPrecisionMode = SamplingPrecisionMode.HIGH
  safety_epsilon: float = 1e-6
  post_corruption_fn: SimplicialPostCorruptionFn = (
      IdentitySimplicialPostCorruptionFn()
  )

  def __post_init__(self):
    if (
        self.unused_token >= 0
        and self.unused_token < self.process_num_categories
    ):
      raise ValueError(
          'unused_token must be outside of the range of the vocabulary.'
          f' Got: {self.unused_token=} and {self.num_categories=}'
      )

  ##############################################################################
  # MARK: Properties
  ##############################################################################

  @property
  def invariant_probs_vec(self) -> Float['M']:
    """Returns the invariant probability distribution as a vector."""
    return jnp.array(self.invariant_probs)

  @property
  def process_num_categories(self) -> int:
    """Returns the number of categories in the process.

    Note that this might be different from the number of categories in the
    distribution, which might contain K+1 elements in the case of masking.
    """
    return len(self.invariant_probs)

  @property
  def is_masking(self) -> bool:
    """Returns whether the process is masking."""
    if self.process_num_categories == self.num_categories:
      return False
    else:
      invariant_probs_masking = (0.0,) * self.num_categories + (1.0,)
      invariant_probs_masking_vec = jnp.array(invariant_probs_masking)
      return jnp.all(
          self.invariant_probs_vec == invariant_probs_masking_vec
      ).item()

  ##############################################################################
  # MARK: h-function
  ##############################################################################

  @kt.typechecked
  def h(self, time: TimeArray) -> TimeArray:
    """Returns the h-function of the process."""
    denominator = 1.0 - self.schedule.alpha(time) + self.safety_epsilon
    return self.schedule.alpha(time) / denominator

  ##############################################################################
  # MARK: Methods
  ##############################################################################

  @kt.typechecked
  def sample_from_invariant(
      self,
      key: PRNGKey,
      data_spec: DataArray,
  ) -> DataArray:
    """Sample from the invariant distribution."""
    invariant_dirichlet_params = self.temperature * self.invariant_probs_vec
    # data_spec is [B, T, 1]
    # invariant_dirichlet_params is [B, T, K]
    # output is [B, T, K]
    return fast_random.log_dirichlet_fast(
        key, alpha=invariant_dirichlet_params, shape=data_spec.shape[:-1]
    )

  @kt.typechecked
  def corrupt(
      self,
      key: PRNGKey,
      x0: DataArray,
      time: TimeArray,
  ) -> tuple[DataArray, TargetInfo]:
    """Corrupt the data according to the schedule and invariant probs.

    The target information contains:
    - x0: The uncorrupted data.
    - logits: The logits of the corrupted data.
    - mask: The mask of the corruption.
    Note that the mask is True if the original data is present in x0. The mask
    is False if the original data is replaced by noise.

    Supports two input formats:
    - Integer tokens: x0 shape (*batch, 1) with int dtype.  Converted to
      one-hot internally (original behaviour).
    - Simplex vectors: x0 shape (*batch, process_num_categories) with float
      dtype.  Used directly as the clean direction for Dirichlet corruption.
      This is needed for multiscale training where coarse-level data is a
      composition (block-average of one-hot vectors).

    Args:
      key: The random key.
      x0: The uncorrupted data.
      time: The time of the corruption.

    Returns:
      xt: The corrupted data.
      target_info: The target info for the corrupted data.
    """
    # Accept both integer tokens and float simplex vectors.  Integer input
    # is the original behaviour (one-hot then Dirichlet); float input lets
    # multiscale training pass coarse-level compositional probabilities
    # directly without the lossy argmax round-trip through integers.
    if jnp.issubdtype(x0.dtype, jnp.integer):
      unused_mask = x0 == self.unused_token
      x0_oh = jax.nn.one_hot(x0[..., 0], self.process_num_categories)
    else:
      unused_mask = None
      x0_oh = x0

    time = jax_helpers.bcast_right(time, x0_oh.ndim)

    # compute Dirichlet parameters
    dirichlet_param = self.invariant_probs_vec + self.h(time) * x0_oh
    dirichlet_param = self.temperature * dirichlet_param
    xt = fast_random.log_dirichlet_fast(key, alpha=dirichlet_param)

    logits = x0_oh

    # Apply post-corruption projection (e.g. symmetrisation).
    xt = self.post_corruption_fn(xt)

    target_info = {
        'x0': x0,  # Int[*b 1] or Float[*b K]; uncorrupted input.
        'logits': logits,  # Float[*b K]; one-hot or simplex direction.
    }

    # Replace the unused probabilities with the unused_token (integer input only).
    if unused_mask is not None:
      xt = jnp.where(unused_mask, self.unused_token, xt)

    return xt, target_info

  @kt.typechecked
  def convert_predictions(
      self,
      prediction: TargetInfo,
      xt: DataArray,
      time: TimeArray,
  ) -> TargetInfo:
    del time  # Unused
    if len(prediction) != 1 or 'logits' not in prediction:
      raise KeyError(
          f'Only logits prediction is supported. Got: {prediction.keys()=}'
      )
    logits = prediction['logits']
    x0_pred = jnp.argmax(logits, axis=-1)
    x0_pred = jnp.expand_dims(x0_pred, axis=-1)
    return {
        'x0': x0_pred,  # Int[*b 1]; Argmax of the predicted distribution.
        'logits': logits,  # Float[*b K]; Raw logits
    }

  @kt.typechecked
  def get_schedule_info(self, time: TimeArray) -> dict[str, TimeArray]:
    """Get the schedule info for the given time."""
    return self.schedule.evaluate(time)

  ##############################################################################
  # MARK: Posterior-bridge interface
  ##############################################################################

  @kt.typechecked
  def sample_endpoint(
      self,
      key: PRNGKey,
      prediction: TargetInfo,
      xt: DataArray,
      time: TimeArray,
  ) -> DataArray:
    """Materialize the predicted clean simplex ``P_hat_0 = softmax(logits)``.

    The simplicial model reports the clean-endpoint posterior as logits;
    the endpoint the bridge consumes is the predicted probability vector
    itself (the simplex is continuous, so there is no token to draw).
    ``key`` is unused.
    """
    del key
    logits = self.convert_predictions(prediction, xt, time)['logits']
    return jax.nn.softmax(logits, axis=-1)

  @kt.typechecked
  def bridge_step(
      self,
      key: PRNGKey,
      x0: DataArray,
      xt: DataArray,
      t: TimeArray,
      s: TimeArray,
      eps: float = 1e-6,
  ) -> DataArray:
    r"""Dirichlet two-endpoint bridge step ``K_{s|0,t}`` (log-simplex state).

    The bare (``churn = 1``) simplicial bridge: given the predicted clean
    simplex ``x_0 = P_hat_0`` and the current log-simplex state ``x_t``,

        W ~ Beta(tau/(1 - alpha_t),  tau/(1 - alpha_s) - tau/(1 - alpha_t)),
        V ~ Dir(tau (alpha_s/(1 - alpha_s) - alpha_t/(1 - alpha_t)) P_hat_0),
        P_s = W P_t + (1 - W) V,

    in log-space.  This is exactly
    :class:`~hackable_diffusion.lib.sampling.simplicial_step_sampler.SimplicialDDIMStep`
    at ``churn = 1`` (the key is split four ways to match it bit-for-bit);
    that stepper is the feature-rich variant exposing the ``churn``
    shrinkage knob.
    """
    log_xt = xt
    t_b = jax_helpers.bcast_right(t, log_xt.ndim)
    s_b = jax_helpers.bcast_right(s, log_xt.ndim)
    temperature = self.temperature
    alpha_t = self.schedule.alpha(t_b)
    alpha_s = self.schedule.alpha(s_b)

    # Four-way split mirrors SimplicialDDIMStep (its churn=1 path leaves the
    # shrinkage key unused), so the two implementations coincide exactly.
    _, beta_key, dir_key, _shrink_key = jax.random.split(key, 4)

    target_shape = log_xt.shape[:-1] + (1,)
    a_w = temperature / (1.0 - alpha_t)
    b_w = temperature / (1.0 - alpha_s) - temperature / (1.0 - alpha_t)
    a_w = jnp.broadcast_to(a_w, target_shape)
    b_w = jnp.broadcast_to(b_w, target_shape)
    log_w, log_1_minus_w = fast_random.sample_log_beta_joint(
        beta_key, a_w, b_w, shape=a_w.shape
    )

    pred_weight = (
        temperature * alpha_s / (1.0 - alpha_s)
        - temperature * alpha_t / (1.0 - alpha_t)
    )
    beta_v = jnp.maximum(pred_weight * x0, eps)
    log_v = fast_random.log_dirichlet_fast(dir_key, beta_v)

    new_xt = jnp.logaddexp(log_w + log_xt, log_1_minus_w + log_v)
    return self.post_corruption_fn(new_xt)

  ##############################################################################
  # MARK: Factory Methods
  ##############################################################################

  @classmethod
  def masking_process(
      cls,
      schedule: SimplicialSchedule,
      num_categories: int,
      unused_token: int = UNUSED_TOKEN,
      temperature: float = 1.0,
      mode: SamplingPrecisionMode = SamplingPrecisionMode.HIGH,
      safety_epsilon: float = 1e-6,
      post_corruption_fn: SimplicialPostCorruptionFn = IdentitySimplicialPostCorruptionFn(),
  ) -> SimplicialProcess:
    """Create a SimplicialProcess with masking invariant distribution."""
    if num_categories < 1:
      raise ValueError(
          f'num_categories must be positive. Got: {num_categories=}'
      )

    invariant_probs = (0.0,) * num_categories + (1.0,)
    return cls(
        schedule=schedule,
        invariant_probs=invariant_probs,
        num_categories=num_categories,
        unused_token=unused_token,
        temperature=temperature,
        mode=mode,
        safety_epsilon=safety_epsilon,
        post_corruption_fn=post_corruption_fn,
    )

  @classmethod
  def uniform_process(
      cls,
      schedule: SimplicialSchedule,
      num_categories: int,
      unused_token: int = UNUSED_TOKEN,
      temperature: float = 1.0,
      mode: SamplingPrecisionMode = SamplingPrecisionMode.HIGH,
      safety_epsilon: float = 1e-6,
      post_corruption_fn: SimplicialPostCorruptionFn = IdentitySimplicialPostCorruptionFn(),
  ) -> SimplicialProcess:
    """Create a SimplicialProcess with uniform invariant distribution."""
    if num_categories < 1:
      raise ValueError(
          f'num_categories must be positive. Got: {num_categories=}'
      )
    invariant_probs = (1.0 / num_categories,) * num_categories
    return cls(
        schedule=schedule,
        invariant_probs=invariant_probs,
        num_categories=num_categories,
        unused_token=unused_token,
        temperature=temperature,
        mode=mode,
        safety_epsilon=safety_epsilon,
        post_corruption_fn=post_corruption_fn,
    )
