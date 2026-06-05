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

"""Simplicial DDIM stepper for simplex-valued clean states.

``SimplicialDDIMStep`` samples a hard categorical clean token from the
predicted logits before each reverse step. That is appropriate for leaf
discrete tokens, but not for clean states that are themselves points on
the simplex (e.g. composition or pooled head states): for those, the
"clean" prediction is already a probability distribution. This stepper
keeps the same reverse Dirichlet/DDIM update, replacing the sampled
one-hot clean state with the predicted clean simplex probabilities.
"""

from __future__ import annotations

import dataclasses

from hackable_diffusion.lib import fast_random
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import jax_helpers as utils
from hackable_diffusion.lib.sampling import base
from hackable_diffusion.lib.sampling.simplicial_step_sampler import (
    SimplicialDDIMStep,
    log_beta_shrinkage,
)
import jax
import jax.numpy as jnp
import kauldron.ktyping as kt


DataArray = hd_typing.DataArray
TargetInfo = hd_typing.TargetInfo
DiffusionStep = base.DiffusionStep
StepInfo = base.StepInfo


@dataclasses.dataclass(frozen=True, kw_only=True)
class SoftSimplicialDDIMStep(SimplicialDDIMStep):
  """DDIM step that treats predicted clean logits as a soft simplex state."""

  min_prob: float = 1e-30

  @kt.typechecked
  def update(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
  ) -> DiffusionStep:
    current_step_info = current_step.step_info
    log_xt = current_step.xt

    time = utils.bcast_right(current_step_info.time, log_xt.ndim)
    next_time = utils.bcast_right(next_step_info.time, log_xt.ndim)
    key = next_step_info.rng

    logits = self.corruption_process.convert_predictions(
        prediction,
        log_xt,
        time,
    )['logits']
    x0_probs = jax.nn.softmax(logits, axis=-1)
    log_x0 = jnp.log(jnp.maximum(x0_probs, self.min_prob))

    eps = self.corruption_process.temperature
    alpha_t = self.corruption_process.schedule.alpha(time)
    alpha_s = self.corruption_process.schedule.alpha(next_time)

    bar_beta_t = eps / (1.0 - alpha_t)
    bar_beta_s = eps / (1.0 - alpha_s)
    target_shape = log_xt.shape[:-1] + (1,)

    if self.churn == 0.0:
      shape_0 = bar_beta_t
      shape_1 = bar_beta_s - shape_0

      _, beta_key = jax.random.split(key)
      log_w, log_1_minus_w = fast_random.sample_log_beta_joint(
          beta_key,
          shape_0,
          shape_1,
          shape=target_shape,
      )

      term_1 = log_w + log_xt
      term_2 = log_1_minus_w + log_x0
    else:
      pi = self.corruption_process.invariant_probs_vec
      h_t = alpha_t / (1.0 - alpha_t)
      h_s = alpha_s / (1.0 - alpha_s)

      concentration = eps * (pi + h_t * x0_probs)

      key, f_key = jax.random.split(key)
      log_pt_kappa = log_beta_shrinkage(
          f_key,
          log_x=log_xt,
          concentration=concentration,
          kappa=1.0 - self.churn,
          safety_epsilon=self.safety_epsilon,
      )

      key, v_key = jax.random.split(key)
      alpha_v = (
          self.churn * eps * pi
          + (eps * h_s - (1.0 - self.churn) * eps * h_t) * x0_probs
      )
      alpha_v = jnp.maximum(alpha_v, self.safety_epsilon)
      log_v = fast_random.log_dirichlet_fast(v_key, alpha=alpha_v, shape=())

      _, beta_key = jax.random.split(key)
      log_w, log_1_minus_w = fast_random.sample_log_beta_joint(
          beta_key,
          (1.0 - self.churn) * bar_beta_t + self.safety_epsilon,
          bar_beta_s - (1.0 - self.churn) * bar_beta_t + self.safety_epsilon,
          shape=target_shape,
      )

      term_1 = log_w + log_pt_kappa
      term_2 = log_1_minus_w + log_v

    new_xt = jnp.logaddexp(term_1, term_2)
    new_xt = self.corruption_process.post_corruption_fn(new_xt)

    return DiffusionStep(
        xt=new_xt,
        step_info=next_step_info,
        aux={'logits': logits},
    )
