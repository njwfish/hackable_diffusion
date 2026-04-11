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

The `InferenceFn is also called within the step and converted into the
relevant representation, for instance score, velocity, etc.
"""

import dataclasses

from hackable_diffusion.lib import fast_random
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.corruption import simplicial
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

DiffusionStep = base.DiffusionStep
StepInfo = base.StepInfo
SamplerStep = base.SamplerStep

SimplicialProcess = simplicial.SimplicialProcess
SimplicialSchedule = schedules.SimplicialSchedule


################################################################################
# MARK: Beta Shrinkage
################################################################################


@kt.typechecked
def log_beta_shrinkage(
    key: jax.Array,
    log_x: jax.Array,
    concentration: jax.Array,
    kappa: float,
    safety_epsilon: float = 0.0,
) -> jax.Array:
  """Beta shrinkage of a Dirichlet sample.

  Let log(X) such that X ~ Dir(concentration) and kappa in [0, 1]. Then this
  function returns log(Y) such that Y ~ Dir(kappa * concentration).

  To do so we leverage the following identity.
  Let B ~ Beta(a, b) with a = kappa * concentration and b = (1 - kappa) *
  concentration.
  Then B X / sum(B X) has the same distribution as Dir(kappa * concentration).
  We call this process "Beta-shrinkage".

  Args:
    key: the random key.
    log_x: the log-Dirichlet sample of shape (..., num_categories).
    concentration: the concentration scalar or array.
    kappa: the shrinkage parameter in [0, 1].
    safety_epsilon: a small positive value added to both Beta concentration
      parameters to avoid the degenerate Beta(0, b) at kappa=0. When
      safety_epsilon > 0, kappa=0 produces a Beta highly concentrated near zero
      rather than a point mass, keeping all computations finite.

  Returns:
    the shrunk log-sample.
  """
  if kappa == 1.0:
    return log_x
  alpha_vec = jnp.broadcast_to(concentration, log_x.shape)

  log_b, _ = fast_random.sample_log_beta_joint(
      key,
      kappa * alpha_vec + safety_epsilon,
      (1.0 - kappa) * alpha_vec + safety_epsilon,
      shape=alpha_vec.shape,
  )

  log_y = log_b + log_x
  log_y = log_y - jax.nn.logsumexp(log_y, axis=-1, keepdims=True)
  return log_y


################################################################################
# MARK: DDIM Step
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class SimplicialDDIMStep(SamplerStep):
  """Simplicial DDIM step with churn support.

  Implements Algorithm 1 from the Simplicial Diffusion manuscript.
  The churn parameter κ ∈ [0, 1] controls stochasticity during sampling:
    - κ = 1 (default): Deterministic DDIM-like update. P_t is used directly
      (no Beta-shrinkage). Equivalent to the original implementation.
    - κ = 0: Fully stochastic. P_t is ignored; V is sampled independently.
    - 0 < κ < 1: Intermediate. P_t is shrunk via Beta-shrinkage toward
      Dir(κε), injecting controlled noise.

  The update rule is:
    1. Predict P̂_0 from the model.
    2. Compute W ~ Beta(κε/(1-α_t), ε/(1-α_s) - κε/(1-α_t)).
    3. Compute V ~ Dir((1-κ)ε·π + (ε·α_s/(1-α_s) - κε·α_t/(1-α_t))·P̂_0).
    4. Apply Beta-shrinkage: P_t^κ = F_κ(P_t) (when κ < 1).
    5. Update: P_s = W·P_t^κ + (1-W)·V.
  """

  corruption_process: SimplicialProcess
  churn: float = 1.0  # κ ∈ [0, 1]
  safety_epsilon: float = 1e-6

  @kt.typechecked
  def initialize(
      self,
      initial_noise: DataArray,
      initial_step_info: StepInfo,
  ) -> DiffusionStep:
    return DiffusionStep(
        xt=initial_noise,
        step_info=initial_step_info,
        aux={'logits': initial_noise},
    )

  @kt.typechecked
  def update(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
      eps: Float = 1e-6,
  ) -> DiffusionStep:

    current_step_info = current_step.step_info
    log_xt = current_step.xt  # log-probabilities on the simplex

    time = current_step_info.time
    next_time = next_step_info.time

    # Broadcast time to match batch dimensions
    time = utils.bcast_right(time, log_xt.ndim)
    next_time = utils.bcast_right(next_time, log_xt.ndim)
    key = next_step_info.rng

    temperature = self.corruption_process.temperature
    kappa = self.churn

    # Get model prediction (logits for P̂_0)
    logits = self.corruption_process.convert_predictions(
        prediction,
        log_xt,
        time,
    )['logits']

    # Schedule values
    alpha_t = self.corruption_process.schedule.alpha(time)
    alpha_s = self.corruption_process.schedule.alpha(next_time)

    key, beta_key, dir_key, shrink_key = jax.random.split(key, 4)

    # ------------------------------------------------------------------
    # Step 2: Sample mixing weight W ~ Beta(a_W, b_W)
    # a_W = κε/(1-α_t),  b_W = ε/(1-α_s) - κε/(1-α_t)
    # ------------------------------------------------------------------
    target_shape = log_xt.shape[:-1] + (1,)
    a_w = kappa * temperature / (1.0 - alpha_t)
    b_w = temperature / (1.0 - alpha_s) - kappa * temperature / (1.0 - alpha_t)
    a_w = jnp.broadcast_to(a_w, target_shape)
    b_w = jnp.broadcast_to(b_w, target_shape)

    log_w, log_1_minus_w = fast_random.sample_log_beta_joint(
        beta_key, a_w, b_w, shape=a_w.shape
    )

    # ------------------------------------------------------------------
    # Step 3: Sample target direction V ~ Dir(β_V)
    # β_V = (1-κ)ε·π + (ε·α_s/(1-α_s) - κε·α_t/(1-α_t))·P̂_0
    # ------------------------------------------------------------------
    pi = self.corruption_process.invariant_probs_vec  # [K]
    # Softmax of logits gives P̂_0 (predicted clean distribution)
    log_p0_hat = jax.nn.log_softmax(logits)
    p0_hat = jnp.exp(log_p0_hat)

    prior_weight = (1.0 - kappa) * temperature  # scalar
    pred_weight = (
        temperature * alpha_s / (1.0 - alpha_s)
        - kappa * temperature * alpha_t / (1.0 - alpha_t)
    )  # [broadcastable]

    beta_v = prior_weight * pi + pred_weight * p0_hat  # [..., K]
    # Clamp to avoid zero/negative concentration params
    beta_v = jnp.maximum(beta_v, eps)

    log_v = fast_random.log_dirichlet_fast(dir_key, beta_v)

    # ------------------------------------------------------------------
    # Step 4: Beta-shrinkage F_κ(P_t) when κ < 1
    # For each k: b_k ~ Beta(κε·P_{t,k}, (1-κ)ε·P_{t,k})
    # P_t^κ = normalize(b ⊙ P_t)
    # When κ = 1, F_κ is identity (no shrinkage needed).
    # ------------------------------------------------------------------
    if kappa < 1.0:
      # P_t on the simplex (from log-space)
      p_t = jax.nn.softmax(log_xt)  # [..., K]

      shrink_a = kappa * temperature * p_t        # [..., K]
      shrink_b = (1.0 - kappa) * temperature * p_t  # [..., K]
      # Clamp to avoid zero Beta params
      shrink_a = jnp.maximum(shrink_a, eps)
      shrink_b = jnp.maximum(shrink_b, eps)

      # Sample b_k ~ Beta(shrink_a_k, shrink_b_k) for each category k
      log_bk, _ = fast_random.sample_log_beta_joint(
          shrink_key, shrink_a, shrink_b, shape=shrink_a.shape
      )
      # P_t^κ = normalize(b ⊙ P_t) in log-space
      log_pt_kappa = log_bk + log_xt
      log_pt_kappa = jax.nn.log_softmax(log_pt_kappa)
    else:
      log_pt_kappa = log_xt

    # ------------------------------------------------------------------
    # Step 5: Update P_s = W·P_t^κ + (1-W)·V
    # ------------------------------------------------------------------
    new_xt = jnp.logaddexp(
        log_w + log_pt_kappa,
        log_1_minus_w + log_v,
    )

    # Apply the post-corruption projection (e.g. symmetrisation) exactly as
    # DiscreteDDIMStep / IntegratedDiscreteDDIMStep do after sampling new_xt.
    new_xt = self.corruption_process.post_corruption_fn(new_xt)

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
