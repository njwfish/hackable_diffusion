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

"""Gaussian corruption: shim over the composed ``InterpolantProcess``.

``GaussianProcess(schedule=...)`` wraps an ``InterpolantProcess`` with
``(StandardNormalSource(), LinearInterpolant(schedule),
GaussianSourceTargets())``.  The legacy parameterisations (``x0``,
``x1``, ``score``, ``velocity``, ``v``) and their bidirectional
conversions all live in ``targets.py``'s ``CONVERTERS`` table -- moved
out of this module but byte-identical in behaviour.

The shim is a named class (not a factory) so type annotations
``corruption_process: GaussianProcess`` throughout downstream code --
including ``kauldron.ktyping`` runtime checks -- continue to work
unchanged.  It delegates :class:`CorruptionProcess` methods to an
internally-constructed :class:`InterpolantProcess`; this avoids the
field-vs-property name collision that would arise from subclassing.

See ``lib/corruption/base.py`` for the protocol design and
``docs/interpolant_refactor_plan.md`` for the refactor rationale.
"""

from __future__ import annotations

import dataclasses

from hackable_diffusion.lib.corruption import base
from hackable_diffusion.lib.corruption import couplings
from hackable_diffusion.lib.corruption import interpolants
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.corruption import targets

# Re-export the conversion table so pinned-legacy imports
# (``from .gaussian import CONVERTERS``) keep working.  Identity object
# -- the table's home is ``targets.py``.
CONVERTERS = targets.CONVERTERS

GaussianSchedule = schedules.GaussianSchedule


@dataclasses.dataclass(kw_only=True, frozen=True)
class GaussianProcess(base.CorruptionProcess):
  """Gaussian corruption ``xt = alpha(t) x_0 + sigma(t) epsilon``.

  Shim over :class:`InterpolantProcess` configured with
  ``(StandardNormalSource(), LinearInterpolant, GaussianSourceTargets)``.
  Every method delegates to the internally built ``_process``;
  behaviour is byte-identical to the pre-refactor implementation.
  """

  schedule: GaussianSchedule
  _process: base.InterpolantProcess = dataclasses.field(
      default=None, init=False, compare=False, repr=False,
  )

  def __post_init__(self):
    object.__setattr__(
        self, '_process',
        base.InterpolantProcess(
            coupling=couplings.StandardNormalSource(),
            interpolant=interpolants.LinearInterpolant(schedule=self.schedule),
            targets=targets.GaussianSourceTargets(),
        ),
    )

  def corrupt(self, key, x0, time):
    return self._process.corrupt(key, x0, time)

  def sample_from_invariant(self, key, data_spec):
    return self._process.sample_from_invariant(key, data_spec)

  def convert_predictions(self, prediction, xt, time):
    return self._process.convert_predictions(prediction, xt, time)

  def get_schedule_info(self, time):
    return self._process.get_schedule_info(time)
