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

"""Concrete :class:`TargetAdapter` implementations.

- :class:`GaussianSourceTargets`: emits ``{x0, x1, score, velocity, v}``.
  Owns the ``CONVERTERS`` table of bidirectional parameterisation
  conversions.  Valid only when the coupling's source is
  ``StandardNormalSource``.
- :class:`VelocityOnlyTargets`: emits ``{x0, x1, velocity}`` for any
  interpolant / source.  Threads ``z`` into ``interpolant.eval`` so it
  works unchanged for :class:`GeodesicInterpolant` (``z`` is ``None``
  and ignored) and :class:`StochasticInterpolant` (``z`` contributes
  the ``gamma(t) z`` term).

The conversion tables for Gaussian parameterisations (``x0 <-> x1 <->
score <-> velocity <-> v``) live here; see LOCAL_PATCHES.md patch 7
for the ``epsilon -> x1`` rename.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.corruption import base
import immutabledict
import jax.numpy as jnp

DataTree = hd_typing.DataTree
TimeTree = hd_typing.TimeTree
TargetInfoTree = hd_typing.TargetInfoTree

TargetAdapter = base.TargetAdapter


################################################################################
# MARK: Gaussian parameterisation conversion table
################################################################################


def _x0_to_x1(x0, xt, alpha, sigma, alpha_der, sigma_der):
  del alpha_der, sigma_der
  return (xt - alpha * x0) / sigma


def _x0_to_score(x0, xt, alpha, sigma, alpha_der, sigma_der):
  del alpha_der, sigma_der
  return (alpha * x0 - xt) / jnp.square(sigma)


def _x0_to_velocity(x0, xt, alpha, sigma, alpha_der, sigma_der):
  return alpha_der * x0 + sigma_der * ((xt - alpha * x0) / sigma)


def _x0_to_v(x0, xt, alpha, sigma, alpha_der, sigma_der):
  del alpha_der, sigma_der
  return alpha * ((xt - alpha * x0) / sigma) - sigma * x0


def _x1_to_x0(x1, xt, alpha, sigma, alpha_der, sigma_der):
  del alpha_der, sigma_der
  return (xt - sigma * x1) / alpha


def _x1_to_score(x1, xt, alpha, sigma, alpha_der, sigma_der):
  del xt, alpha, alpha_der, sigma_der
  return -x1 / sigma


def _x1_to_velocity(x1, xt, alpha, sigma, alpha_der, sigma_der):
  return alpha_der * ((xt - sigma * x1) / alpha) + sigma_der * x1


def _x1_to_v(x1, xt, alpha, sigma, alpha_der, sigma_der):
  del alpha_der, sigma_der
  return alpha * x1 - sigma * ((xt - sigma * x1) / alpha)


def _score_to_x0(score, xt, alpha, sigma, alpha_der, sigma_der):
  del alpha_der, sigma_der
  return (xt + jnp.square(sigma) * score) / alpha


def _score_to_x1(score, xt, alpha, sigma, alpha_der, sigma_der):
  del xt, alpha, alpha_der, sigma_der
  return -score * sigma


def _score_to_velocity(score, xt, alpha, sigma, alpha_der, sigma_der):
  return alpha_der * ((xt + jnp.square(sigma) * score) / alpha) + sigma_der * (
      -score * sigma
  )


def _score_to_v(score, xt, alpha, sigma, alpha_der, sigma_der):
  del alpha_der, sigma_der
  return alpha * (-score * sigma) - sigma * (
      (xt + jnp.square(sigma) * score) / alpha
  )


def _velocity_to_x0(velocity, xt, alpha, sigma, alpha_der, sigma_der):
  common_denominator = alpha_der * sigma - sigma_der * alpha
  return (velocity * sigma - sigma_der * xt) / common_denominator


def _velocity_to_x1(velocity, xt, alpha, sigma, alpha_der, sigma_der):
  common_denominator = alpha_der * sigma - sigma_der * alpha
  return (alpha_der * xt - alpha * velocity) / common_denominator


def _velocity_to_score(velocity, xt, alpha, sigma, alpha_der, sigma_der):
  return (alpha * velocity - alpha_der * xt) / (
      sigma * (alpha_der * sigma - sigma_der * alpha)
  )


def _velocity_to_v(velocity, xt, alpha, sigma, alpha_der, sigma_der):
  common_denominator = alpha_der * sigma - sigma_der * alpha
  numerator = (alpha * alpha_der + sigma * sigma_der) * xt - (
      jnp.square(alpha) + jnp.square(sigma)
  ) * velocity
  return numerator / common_denominator


def _v_to_x0(v, xt, alpha, sigma, alpha_der, sigma_der):
  del alpha_der, sigma_der
  common_denominator = jnp.square(alpha) + jnp.square(sigma)
  return (alpha * xt - sigma * v) / common_denominator


def _v_to_x1(v, xt, alpha, sigma, alpha_der, sigma_der):
  del alpha_der, sigma_der
  common_denominator = jnp.square(alpha) + jnp.square(sigma)
  return (sigma * xt + alpha * v) / common_denominator


def _v_to_score(v, xt, alpha, sigma, alpha_der, sigma_der):
  del alpha_der, sigma_der
  return -(sigma * xt + alpha * v) / (
      sigma * (jnp.square(alpha) + jnp.square(sigma))
  )


def _v_to_velocity(v, xt, alpha, sigma, alpha_der, sigma_der):
  common_denominator = jnp.square(alpha) + jnp.square(sigma)
  numerator = (alpha_der * alpha + sigma_der * sigma) * xt + (
      sigma_der * alpha - alpha_der * sigma
  ) * v
  return numerator / common_denominator


def _identity(y, xt, alpha, sigma, alpha_der, sigma_der):
  del xt, alpha, sigma, alpha_der, sigma_der
  return y


CONVERTERS = immutabledict.immutabledict({
    'x0': {
        'x0': _identity,
        'x1': _x0_to_x1,
        'score': _x0_to_score,
        'velocity': _x0_to_velocity,
        'v': _x0_to_v,
    },
    'x1': {
        'x0': _x1_to_x0,
        'x1': _identity,
        'score': _x1_to_score,
        'velocity': _x1_to_velocity,
        'v': _x1_to_v,
    },
    'score': {
        'x0': _score_to_x0,
        'x1': _score_to_x1,
        'score': _identity,
        'velocity': _score_to_velocity,
        'v': _score_to_v,
    },
    'velocity': {
        'x0': _velocity_to_x0,
        'x1': _velocity_to_x1,
        'score': _velocity_to_score,
        'velocity': _identity,
        'v': _velocity_to_v,
    },
    'v': {
        'x0': _v_to_x0,
        'x1': _v_to_x1,
        'score': _v_to_score,
        'velocity': _v_to_velocity,
        'v': _identity,
    },
})


################################################################################
# MARK: Target adapters
################################################################################


def _alpha_sigma_and_der(schedule, time, ndim):
  """Return ``(alpha, sigma, alpha_der, sigma_der)`` broadcast to ``ndim``."""
  time = utils.bcast_right(time, ndim)
  alpha = schedule.alpha(time)
  sigma = schedule.sigma(time)
  alpha_der = utils.egrad(schedule.alpha)(time)
  sigma_der = utils.egrad(schedule.sigma)(time)
  return alpha, sigma, alpha_der, sigma_der


@dataclasses.dataclass(kw_only=True, frozen=True)
class GaussianSourceTargets(TargetAdapter):
  """Emits ``{x0, x1, score, velocity, v}`` for a Gaussian source.

  Byte-equivalent to legacy ``GaussianProcess.corrupt``'s ``target_info``
  dict when composed with :class:`LinearInterpolant` and a
  :class:`StandardNormalSource`.  The identities
  ``score = -x1/sigma`` and ``v = alpha*x1 - sigma*x0`` only hold when
  ``x_1 ~ N(0, I)``; composing this adapter with a non-Gaussian source
  produces incorrect target dicts.  Construction-time checks in
  downstream stepper wiring should catch this.
  """

  emitted_keys: ClassVar[frozenset[str]] = frozenset(
      {'x0', 'x1', 'score', 'velocity', 'v'}
  )

  def emit(
      self,
      *,
      x0: DataTree,
      x1: DataTree,
      z: DataTree | None,
      xt: DataTree,
      dxt_dt: DataTree,
      t: TimeTree,
      interpolant,
  ) -> TargetInfoTree:
    del z, xt
    time = utils.bcast_right(t, x0.ndim)
    alpha = interpolant.schedule.alpha(time)
    sigma = interpolant.schedule.sigma(time)
    return {
        'x0': x0,
        'x1': x1,
        'score': -x1 / sigma,
        'velocity': dxt_dt,
        'v': alpha * x1 - sigma * x0,
    }

  def convert(
      self,
      *,
      prediction: TargetInfoTree,
      xt: DataTree,
      t: TimeTree,
      interpolant,
  ) -> TargetInfoTree:
    if len(prediction) != 1:
      raise ValueError(
          f'Exactly one prediction is required. Got: {prediction.keys()=}'
      )
    source_type, source_value = next(iter(prediction.items()))
    converters = CONVERTERS[source_type]
    alpha, sigma, alpha_der, sigma_der = _alpha_sigma_and_der(
        interpolant.schedule, t, xt.ndim,
    )
    return {
        pred_type: converter(
            source_value, xt=xt,
            alpha=alpha, sigma=sigma,
            alpha_der=alpha_der, sigma_der=sigma_der,
        )
        for pred_type, converter in converters.items()
    }


@dataclasses.dataclass(kw_only=True, frozen=True)
class VelocityOnlyTargets(TargetAdapter):
  """Emits ``{x0, x1, velocity}`` for an arbitrary interpolant / source.

  The minimum valid target set under data-to-data flow matching: the
  Gaussian identities (``score = -x1/sigma`` etc.) do not hold for a
  non-Gaussian ``x_1``, so they're not emitted.  ``velocity`` is
  whatever :meth:`Interpolant.eval` returns as its second element.

  Works unchanged for all three interpolants:

  - :class:`LinearInterpolant` -- ``z`` is ``None``, ignored.
  - :class:`GeodesicInterpolant` -- ``z`` is ``None``, ignored;
    geodesic-velocity chain rule lives in ``eval``.  This is the
    Riemannian flow matching adapter (byte-equivalent to legacy
    ``RiemannianProcess.corrupt``'s ``target_info``).
  - :class:`StochasticInterpolant` -- ``z`` threads through so
    ``velocity`` carries the ``gamma_dot z`` term; training reduces
    to a single-head regression on ``x_0`` or ``velocity``.
  """

  emitted_keys: ClassVar[frozenset[str]] = frozenset({'x0', 'x1', 'velocity'})

  def emit(
      self,
      *,
      x0: DataTree,
      x1: DataTree,
      z: DataTree | None,
      xt: DataTree,
      dxt_dt: DataTree,
      t: TimeTree,
      interpolant,
  ) -> TargetInfoTree:
    del z, xt, t, interpolant
    return {'x0': x0, 'x1': x1, 'velocity': dxt_dt}

  def convert(
      self,
      *,
      prediction: TargetInfoTree,
      xt: DataTree,
      t: TimeTree,
      interpolant,
  ) -> TargetInfoTree:
    del xt, t, interpolant
    if 'velocity' in prediction:
      return prediction
    raise NotImplementedError(
        'VelocityOnlyTargets only supports velocity predictions.  For '
        'other parameterisations, use GaussianSourceTargets (requires '
        'a StandardNormalSource).'
    )
