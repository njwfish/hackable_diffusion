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

"""One-step SMC proposal log-ratio.

The math lives on each stepper's :meth:`kernel` method, which returns a
concrete :class:`StepKernel` (``GaussianStepKernel``,
``SimplicialStepKernel``, ...).  This module is three lines: short-circuit
on an identity correction, call the kernel, return its log-density ratio.

Extending to a new stepper is local to its class -- implement
``kernel`` returning a ``StepKernel`` that encodes its transition
structure.  Users who can't modify an upstream stepper class can still
plug in by wrapping the stepper (see ``docs/composable_guidance.md`` for
the recipe) -- no framework-side registry needed.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def proposal_log_ratio(
    *,
    stepper: Any,
    outputs_uncorrected: dict[str, jax.Array],
    outputs_corrected: dict[str, jax.Array],
    xt_prev: jax.Array,
    xt_next: jax.Array,
    time_prev: jax.Array,
    time_next: jax.Array,
    correction_identity: bool,
) -> jax.Array:
  """``log p_theta(xt_next | xt_prev) - log q(xt_next | xt_prev)`` for one step.

  Returns zero when the correction is an identity; otherwise dispatches
  to ``stepper.kernel(...).log_density_ratio(xt_prev, xt_next)``.  Any
  stepper with a working ``kernel`` method satisfies this -- no
  isinstance dispatch, no registry.
  """
  if correction_identity:
    return jnp.zeros(xt_next.shape[0], dtype=xt_next.dtype)
  kernel = stepper.kernel(
      prediction_uncorrected=outputs_uncorrected,
      prediction_corrected=outputs_corrected,
      xt=xt_prev,
      time_prev=time_prev,
      time_next=time_next,
  )
  return kernel.log_density_ratio(xt_prev, xt_next)
