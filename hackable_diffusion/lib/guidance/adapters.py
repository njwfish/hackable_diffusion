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

"""Adapters from other guidance APIs to the :class:`CorrectionFn` protocol.

- :class:`BoundAggregateGuidanceFn` wraps legacy guidance classes whose
  ``__call__`` takes a concrete ``aggregate_target`` argument.
- :class:`CFGCorrectionFn` wraps a ``lib.inference.guidance.GuidanceFn``
  (classifier-free guidance) and an unconditional inference fn, combining
  the conditional / unconditional outputs per the GuidanceFn's formula.
  Lets CFG plug into :class:`ConditionalDiffusionSampler` alongside
  Pi-GDM, TDS, DPS, etc.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import jax

from hackable_diffusion.lib.inference.guidance import GuidanceFn
from hackable_diffusion.lib.guidance.protocols import CorrectionFn
from hackable_diffusion.lib.guidance.utils import call_inference_fn


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


@dataclasses.dataclass(kw_only=True, frozen=True)
class CFGCorrectionFn(CorrectionFn):
  """Classifier-free guidance as a :class:`CorrectionFn`.

  Calls ``unconditional_inference_fn`` at every step to obtain the
  unconditional prediction, then combines conditional / unconditional
  outputs via ``guidance_fn`` (any
  :class:`hackable_diffusion.lib.inference.guidance.GuidanceFn`).

  Composition:
      Scalar CFG          : ``guidance_fn=ScalarGuidanceFn(guidance=w)``
      Limited-interval    : ``guidance_fn=LimitedIntervalGuidanceFn(...)``
      Nested / multi-CFG  : ``guidance_fn=NestedGuidanceFn(...)``

  Plugs into :class:`ConditionalDiffusionSampler` alongside any other
  correction / twist / resampler -- e.g. CFG + TDS is CFG-guided
  posterior sampling.
  """

  unconditional_inference_fn: Callable
  guidance_fn: GuidanceFn

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
    del schedule, corruption_process
    uncond_outputs = call_inference_fn(
        self.unconditional_inference_fn,
        xt=xt, time=time, conditioning=None, rng=rng,
    )
    return self.guidance_fn(
        xt=xt,
        conditioning=conditioning,
        time=time,
        cond_outputs=outputs,
        uncond_outputs=uncond_outputs,
    )
