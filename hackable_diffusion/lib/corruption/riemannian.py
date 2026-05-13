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

"""Riemannian flow matching: shim over the composed ``InterpolantProcess``.

``RiemannianProcess(manifold=..., schedule=...)`` wraps an
``InterpolantProcess`` with ``(prior=UniformManifoldPrior(manifold),
interpolant=GeodesicInterpolant, targets=VelocityOnlyTargets)``.
Based on https://arxiv.org/abs/2302.03660.

See ``lib/corruption/base.py`` for the protocol design.
"""

from __future__ import annotations

import dataclasses

from hackable_diffusion.lib import manifolds
from hackable_diffusion.lib.corruption import base
from hackable_diffusion.lib.corruption import interpolants
from hackable_diffusion.lib.corruption import priors
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.corruption import targets


@dataclasses.dataclass(kw_only=True, frozen=True)
class RiemannianProcess(base.CorruptionProcess):
  """Riemannian flow-matching corruption on a manifold.

  Given a schedule with interpolation parameter ``alpha(t)``:

      x_t = geodesic(x_1, x_0, alpha(t))
      target = alpha_dot(t) * manifold.velocity(x_1, x_0, alpha(t))

  Shim over :class:`InterpolantProcess` with
  ``(prior=UniformManifoldPrior(manifold), interpolant=GeodesicInterpolant,
  targets=VelocityOnlyTargets)`` -- default :class:`IndependentCoupling`.
  """

  manifold: manifolds.Manifold
  schedule: schedules.RiemannianSchedule
  _process: base.InterpolantProcess = dataclasses.field(
      default=None, init=False, compare=False, repr=False,
  )

  def __post_init__(self):
    object.__setattr__(
        self, '_process',
        base.InterpolantProcess(
            prior=priors.UniformManifoldPrior(manifold=self.manifold),
            interpolant=interpolants.GeodesicInterpolant(
                manifold=self.manifold, schedule=self.schedule,
            ),
            targets=targets.VelocityOnlyTargets(),
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
