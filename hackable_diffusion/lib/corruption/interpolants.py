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

"""Concrete :class:`Interpolant` implementations.

- :class:`LinearInterpolant`: ``x_t = alpha(t) x_0 + sigma(t) x_1``.
  Wraps a :class:`GaussianSchedule`.  Byte-equivalent to the Gaussian
  interpolation in legacy ``GaussianProcess.corrupt``.
- :class:`GeodesicInterpolant`: ``x_t = geodesic(x_1, x_0, alpha(t))``.
  Wraps a :class:`RiemannianSchedule` + :class:`Manifold`.
  Byte-equivalent to the Riemannian interpolation in legacy
  ``RiemannianProcess.corrupt``.
- :class:`StochasticInterpolant`: deferred to M5.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import manifolds
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.corruption import base
from hackable_diffusion.lib.corruption import schedules

DataTree = hd_typing.DataTree
TimeTree = hd_typing.TimeTree

Interpolant = base.Interpolant
GaussianSchedule = schedules.GaussianSchedule
RiemannianSchedule = schedules.RiemannianSchedule


@dataclasses.dataclass(kw_only=True, frozen=True)
class LinearInterpolant(Interpolant):
  """``x_t = alpha(t) * x_0 + sigma(t) * x_1``.

  The Gaussian-diffusion interpolation expressed in the generic
  data-to-data language.  ``x_1`` plays the role of the noise
  ``epsilon`` in the standard diffusion formulation (or the target
  sample when the coupling is data-to-data).
  """

  schedule: GaussianSchedule
  needs_noise: ClassVar[bool] = False

  def eval(
      self,
      x0: DataTree,
      x1: DataTree,
      t: TimeTree,
      z: DataTree | None = None,
  ) -> tuple[DataTree, DataTree]:
    del z
    t_b = utils.bcast_right(t, x0.ndim)
    alpha = self.schedule.alpha(t_b)
    sigma = self.schedule.sigma(t_b)
    alpha_der = utils.egrad(self.schedule.alpha)(t_b)
    sigma_der = utils.egrad(self.schedule.sigma)(t_b)
    xt = alpha * x0 + sigma * x1
    dxt_dt = alpha_der * x0 + sigma_der * x1
    return xt, dxt_dt


@dataclasses.dataclass(kw_only=True, frozen=True)
class GeodesicInterpolant(Interpolant):
  """``x_t = geodesic(x_1, x_0, alpha(t))`` on a Riemannian manifold.

  Byte-equivalent to the legacy ``RiemannianProcess.corrupt``
  interpolation; the velocity is ``alpha_dot(t) * manifold.velocity(x1,
  x0, alpha(t))`` from the geodesic chain rule.
  """

  manifold: manifolds.Manifold
  schedule: RiemannianSchedule
  needs_noise: ClassVar[bool] = False

  def eval(
      self,
      x0: DataTree,
      x1: DataTree,
      t: TimeTree,
      z: DataTree | None = None,
  ) -> tuple[DataTree, DataTree]:
    del z
    alpha_t = utils.bcast_right(self.schedule.alpha(t), x0.ndim)
    alpha_dot_t = utils.bcast_right(self.schedule.alpha_dot(t), x0.ndim)
    xt = manifolds.geodesic(self.manifold, x=x1, y=x0, t=alpha_t)
    dxt_dt = alpha_dot_t * self.manifold.velocity(x=x1, y=x0, t=alpha_t)
    return xt, dxt_dt
