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

"""Adapters from legacy guidance classes to the :class:`CorrectionFn` protocol.

Many guidance classes (project-specific Pi-GDM / cov-aware / multi-head
implementations) have ``__call__`` signatures with a concrete
``aggregate_target`` / ``aggregate_targets`` argument rather than the
framework's generic ``conditioning`` slot.

:class:`BoundAggregateGuidanceFn` is a thin adapter that binds the target
and exposes the ``CorrectionFn`` interface, so any such class can plug
into :class:`ConditionalDiffusionSampler` without modification.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax

from hackable_diffusion.lib.guidance.protocols import CorrectionFn


@dataclasses.dataclass(kw_only=True, frozen=True)
class BoundAggregateGuidanceFn(CorrectionFn):
  """Bind ``aggregate_target`` to a legacy guidance; expose ``CorrectionFn``.

  Attributes:
    guidance: the underlying guidance object whose ``__call__`` takes
      ``(outputs, xt, time, schedule, corruption_process, aggregate_target=...)``
      or the plural ``aggregate_targets=...`` variant.
    aggregate_target: the pre-broadcast target ``y``.  Shape must match
      what ``guidance`` expects (typically ``(batch, m)`` or
      ``(batch, *spatial)``).
    multi_head: set to ``True`` when ``guidance`` expects
      ``aggregate_targets`` (plural) -- e.g. for multi-head Gaussian or
      moment-family guidance.  When ``True``, ``aggregate_target`` must
      be a sequence.
  """

  guidance: Any
  aggregate_target: Any
  multi_head: bool = False

  def __call__(
      self,
      outputs: dict[str, jax.Array],
      xt: jax.Array,
      time: jax.Array,
      *,
      schedule: Any,
      corruption_process: Any,
      conditioning: Any = None,
      rng: jax.Array | None = None,
  ) -> dict[str, jax.Array]:
    del conditioning, rng
    kwarg_name = "aggregate_targets" if self.multi_head else "aggregate_target"
    return self.guidance(
        outputs=outputs,
        xt=xt,
        time=time,
        schedule=schedule,
        corruption_process=corruption_process,
        **{kwarg_name: self.aggregate_target},
    )
