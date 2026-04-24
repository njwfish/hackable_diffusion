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
- :class:`StochasticInterpolant`: ``x_t = alpha(t) x_0 + beta(t) x_1
  + gamma(t) z``.  "Just another interpolant" -- pair with
  :class:`VelocityOnlyTargets` and train on ``x_0`` or ``velocity``.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, ClassVar

import jax
import jax.numpy as jnp

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


@dataclasses.dataclass(kw_only=True, frozen=True)
class StochasticInterpolant(Interpolant):
  """``x_t = alpha(t) x_0 + beta(t) x_1 + gamma(t) z``, ``z ~ N(0, I)``.

  Pair with :class:`VelocityOnlyTargets` and train on ``x_0`` or
  ``velocity`` like any other interpolant.  Endpoint conditions
  ``gamma(0) = gamma(1) = 0`` are checked at construction;
  :func:`canonical_gamma` provides ``sqrt(t(1-t))``.  ``gamma = 0``
  recovers linear flow matching, ``beta = 0`` recovers Gaussian
  diffusion with ``gamma`` in place of ``sigma``.

  The interpolant is its own schedule: ``self.schedule is self`` and
  ``evaluate(t)`` yields ``{alpha, beta, gamma}``.  A stochastic
  sampler can query ``gamma(t)`` off ``corruption_process.schedule``
  without any new plumbing.
  """

  alpha: Callable[[jax.Array], jax.Array]
  beta: Callable[[jax.Array], jax.Array]
  gamma: Callable[[jax.Array], jax.Array]
  needs_noise: ClassVar[bool] = True
  _gamma_tol: ClassVar[float] = 1e-5

  def __post_init__(self):
    g0 = float(self.gamma(jnp.asarray(0.0)))
    g1 = float(self.gamma(jnp.asarray(1.0)))
    if abs(g0) > self._gamma_tol or abs(g1) > self._gamma_tol:
      raise ValueError(
          f'StochasticInterpolant requires gamma(0) = gamma(1) = 0; '
          f'got gamma(0) = {g0:.2e}, gamma(1) = {g1:.2e}.'
      )

  @property
  def schedule(self) -> 'StochasticInterpolant':
    return self

  def evaluate(self, time: jax.Array) -> dict[str, jax.Array]:
    return {
        'alpha': self.alpha(time),
        'beta': self.beta(time),
        'gamma': self.gamma(time),
    }

  def eval(
      self,
      x0: DataTree,
      x1: DataTree,
      t: TimeTree,
      z: DataTree | None = None,
  ) -> tuple[DataTree, DataTree]:
    if z is None:
      raise ValueError(
          'StochasticInterpolant.eval requires ``z`` '
          '(drawn by ``InterpolantProcess`` when ``needs_noise = True``).'
      )
    t_b = utils.bcast_right(t, x0.ndim)
    alpha = self.alpha(t_b)
    beta = self.beta(t_b)
    gamma = self.gamma(t_b)
    alpha_der = utils.egrad(self.alpha)(t_b)
    beta_der = utils.egrad(self.beta)(t_b)
    gamma_der = utils.egrad(self.gamma)(t_b)
    xt = alpha * x0 + beta * x1 + gamma * z
    dxt_dt = alpha_der * x0 + beta_der * x1 + gamma_der * z
    return xt, dxt_dt


def canonical_gamma(t: jax.Array) -> jax.Array:
  """``gamma(t) = sqrt(t (1 - t))``: smooth, zero at both endpoints."""
  return jnp.sqrt(jnp.clip(t * (1.0 - t), 0.0, None))
