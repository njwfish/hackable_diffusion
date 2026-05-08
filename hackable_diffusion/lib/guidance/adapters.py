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

"""Adapters from project-specific guidance APIs to :class:`CorrectionFn`.

:class:`BoundAggregateGuidanceFn` wraps legacy / project-specific
guidance classes whose ``__call__`` takes a concrete ``aggregate_target``
argument and an ``outputs`` dict (the pre-refactor signature).  It
adapts to the new :class:`CorrectionFn` sig -- ``(x0, xt, time, *,
denoiser_fn, schedule) -> x0_new`` -- by internally roundtripping
``x0`` through the bound ``corruption_process``.

CFG intentionally has no adapter in this file: CFG is denoiser
composition, not an observation-driven correction.  Use
:func:`hackable_diffusion.lib.guidance.denoisers.make_cfg_inference_fn`
to blend a conditional + unconditional model into a single
``inference_fn`` before handing it to the sampler.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax

from hackable_diffusion.lib.guidance.protocols import (
    CorrectionFn, DenoiserFn, PosteriorCloudFn,
)


@dataclasses.dataclass(kw_only=True, frozen=True)
class BoundAggregateGuidanceFn(CorrectionFn):
  """Adapt a legacy ``aggregate_target``-taking guidance class.

  The legacy shape is

      guidance(outputs, xt, time, *, schedule, corruption_process,
               aggregate_target=y) -> outputs_new

  which this adapter wraps as

      correction(x0, xt, time, *, denoiser_fn, schedule) -> x0_new

  by reconstructing a one-key outputs dict from ``x0``, calling the
  legacy ``guidance`` with the bound ``corruption_process`` +
  ``aggregate_target``, and converting the result back to ``x0`` via
  ``convert_predictions``.  ``denoiser_fn`` is ignored by the legacy
  class (it has no notion of a denoiser closure).

  Attributes:
    guidance: the legacy guidance instance.
    aggregate_target: the pre-broadcast target ``y``.
    corruption_process: bound at adapter-construction time so the
      :class:`CorrectionFn` signature can stay clean of it.
    multi_head: set to ``True`` if the legacy guidance takes plural
      ``aggregate_targets=`` instead of singular ``aggregate_target=``.
  """

  guidance: Any
  aggregate_target: Any
  corruption_process: Any
  multi_head: bool = False

  def __call__(
      self,
      x0: jax.Array,
      xt: jax.Array,
      time: jax.Array,
      *,
      denoiser_fn: DenoiserFn,
      schedule: Any,
      cloud_fn: PosteriorCloudFn | None = None,
      rng: jax.Array | None = None,
  ) -> jax.Array:
    del denoiser_fn, cloud_fn, rng
    kwarg = "aggregate_targets" if self.multi_head else "aggregate_target"
    legacy_outputs = self.guidance(
        outputs={"x0": x0},
        xt=xt,
        time=time,
        schedule=schedule,
        corruption_process=self.corruption_process,
        **{kwarg: self.aggregate_target},
    )
    return self.corruption_process.convert_predictions(
        legacy_outputs, xt, time,
    )["x0"]
