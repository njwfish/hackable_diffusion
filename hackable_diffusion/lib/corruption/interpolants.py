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

Each interpolant exposes the *same* geometry from both ends:

- ``eval`` -- the forward path used in training (``x_t`` from the
  endpoints, optionally ``+ gamma(t) z``).
- ``bridge_step`` -- the reverse two-endpoint step used in sampling
  (``x_t`` at ``t`` and a clean endpoint ``x_0`` -> ``x_s`` at
  ``s in [0, t)``), with the same noise the forward path used.  This is
  the posterior-bridge step (see
  :class:`hackable_diffusion.lib.sampling.bridge_step_sampler.PosteriorBridgeStep`);
  because it is the same object that defined the corruption, the sampling
  step is consistent with training by construction.

The shipped interpolants:

- :class:`LinearInterpolant`: ``x_t = alpha(t) x_0 + sigma(t) x_1``.
  Wraps a :class:`GaussianSchedule`.  Byte-equivalent to the Gaussian
  interpolation in legacy ``GaussianProcess.corrupt``.  Its bridge step
  is the deterministic affine map (the ODE-limit / DDIM update).
- :class:`GeodesicInterpolant`: ``x_t = geodesic(x_1, x_0, alpha(t))``.
  Wraps a :class:`RiemannianSchedule` + :class:`Manifold`.
  Byte-equivalent to the Riemannian interpolation in legacy
  ``RiemannianProcess.corrupt``.  Its bridge step walks the geodesic from
  ``x_t`` toward ``x_0``.
- :class:`StochasticInterpolant`: ``x_t = alpha(t) x_0 + beta(t) x_1
  + gamma(t) z``.  "Just another interpolant" -- pair with
  :class:`VelocityOnlyTargets` and train on ``x_0`` or ``velocity``.  Its
  bridge step is the Brownian-bridge conditional, carrying the same
  ``gamma`` noise into sampling.

The shared Gaussian closed form (:func:`_gaussian_bridge_coeffs`) makes
the ``LinearInterpolant`` step the ``gamma = 0`` special case of the
``StochasticInterpolant`` step.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, ClassVar

import jax
import jax.numpy as jnp

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import manifolds
from hackable_diffusion.lib import jax_helpers
from hackable_diffusion.lib.corruption import base
from hackable_diffusion.lib.corruption import schedules

DataTree = hd_typing.DataTree
TimeTree = hd_typing.TimeTree
PRNGKey = hd_typing.PRNGKey

Interpolant = base.Interpolant
GaussianSchedule = schedules.GaussianSchedule
RiemannianSchedule = schedules.RiemannianSchedule


def _gaussian_bridge_coeffs(
    *,
    alpha_s: jax.Array,
    alpha_t: jax.Array,
    weight_s: jax.Array,
    weight_t: jax.Array,
    gamma_s: jax.Array,
    gamma_t: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  r"""Affine coefficients ``(coeff_x0, coeff_xt, sigma_step)`` of the bridge.

  The Gaussian bridge step is ``x_s = coeff_x0 x_0 + coeff_xt x_t +
  sigma_step Z`` with ``coeff_xt = rho = weight(s)/weight(t)``,
  ``coeff_x0 = alpha(s) - rho alpha(t)`` and
  ``sigma_step = sqrt(max(gamma(s)^2 - rho^2 gamma(t)^2, 0))``.  Factored
  out of :func:`_gaussian_bridge_step` so the affine map and its noise std
  are computed in one place.
  """
  rho = weight_s / weight_t
  coeff_x0 = alpha_s - rho * alpha_t
  coeff_xt = rho
  sigma_step = jnp.sqrt(jnp.clip(gamma_s**2 - rho**2 * gamma_t**2, 0.0, None))
  return coeff_x0, coeff_xt, sigma_step


def _gaussian_bridge_step(
    *,
    x0: DataTree,
    xt: DataTree,
    alpha_s: jax.Array,
    alpha_t: jax.Array,
    weight_s: jax.Array,
    weight_t: jax.Array,
    gamma_s: jax.Array,
    gamma_t: jax.Array,
    key: PRNGKey | None,
) -> DataTree:
  r"""Two-endpoint Gaussian bridge step ``K_{s|0,t}(. | x_0, x_t)``.

  The shared closed form behind every linear-Gaussian endpoint bridge
  (``LinearInterpolant`` and ``StochasticInterpolant``).  Write the path
  as ``x_t = alpha(t) x_0 + M_t`` where ``M_t = weight(t) x_1 + N_t`` is a
  Gaussian bridge from ``M_0 = 0`` (clean) to ``M_1 = x_1`` (terminal
  endpoint), ``weight`` is the coefficient on the terminal endpoint
  ``x_1`` (``sigma`` for ``LinearInterpolant``, ``beta`` for
  ``StochasticInterpolant``) and ``N_t`` is the optional Brownian-bridge
  noise with marginal std ``gamma(t)``.

  For ``s < t``, the Brownian / Gaussian-bridge Markov property makes
  ``M_s`` given ``(M_0 = 0, M_t)`` independent of the terminal endpoint
  ``x_1``: it depends only on the current state through
  ``M_t = x_t - alpha(t) x_0``.  With retention ``rho = weight(s)/weight(t)``,

      x_s = alpha(s) x_0 + rho (x_t - alpha(t) x_0)
            + sqrt(max(gamma(s)^2 - rho^2 gamma(t)^2, 0)) Z,  Z ~ N(0, I).

  For the canonical Brownian-bridge interpolant
  (``alpha = 1 - lambda``, ``beta = lambda``,
  ``gamma = sigma_br sqrt(lambda (1 - lambda))``) this reduces exactly to
  the Posterior Bridges Euclidean step
  ``x_s = (1 - rho) x_0 + rho x_t + sigma_br sqrt(lambda_s (1 - rho)) Z``
  with ``rho = lambda_s / lambda_t``.  When ``gamma == 0`` the step is the
  deterministic affine map -- equivalently the ODE-limit DDIM update.
  """
  # coeff_xt = rho = weight(s)/weight(t); weight(t) > 0 for s < t away from
  # the clean endpoint, and weight(s) -> 0 as s -> 0 so the step collapses
  # to x_0 (K_{0|0,t} = delta_{x_0}).
  coeff_x0, coeff_xt, sigma_step = _gaussian_bridge_coeffs(
      alpha_s=alpha_s, alpha_t=alpha_t,
      weight_s=weight_s, weight_t=weight_t,
      gamma_s=gamma_s, gamma_t=gamma_t,
  )
  mean = coeff_x0 * x0 + coeff_xt * xt
  if key is None:
    return mean
  z = jax.random.normal(key, shape=xt.shape, dtype=xt.dtype)
  return mean + sigma_step * z


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
    t_b = jax_helpers.bcast_right(t, x0.ndim)
    alpha = self.schedule.alpha(t_b)
    sigma = self.schedule.sigma(t_b)
    alpha_der = jax_helpers.egrad(self.schedule.alpha)(t_b)
    sigma_der = jax_helpers.egrad(self.schedule.sigma)(t_b)
    xt = alpha * x0 + sigma * x1
    dxt_dt = alpha_der * x0 + sigma_der * x1
    return xt, dxt_dt

  def bridge_step(
      self,
      *,
      x0: DataTree,
      xt: DataTree,
      t: TimeTree,
      s: TimeTree,
      key: PRNGKey | None = None,
  ) -> DataTree:
    """Deterministic endpoint bridge ``x_s = alpha(s) x_0 + (sigma(s)/sigma(t))(x_t - alpha(t) x_0)``.

    The terminal endpoint ``x_1 = (x_t - alpha(t) x_0) / sigma(t)`` is
    fully determined by ``(x_0, x_t)`` (there is no bridge noise), so the
    step is a deterministic affine map -- this is exactly the ODE-limit /
    deterministic DDIM update written in the data-to-data language.  The
    ``key`` is accepted for interface uniformity and ignored.
    """
    del key
    t_b = jax_helpers.bcast_right(t, x0.ndim)
    s_b = jax_helpers.bcast_right(s, x0.ndim)
    return _gaussian_bridge_step(
        x0=x0,
        xt=xt,
        alpha_s=self.schedule.alpha(s_b),
        alpha_t=self.schedule.alpha(t_b),
        weight_s=self.schedule.sigma(s_b),
        weight_t=self.schedule.sigma(t_b),
        gamma_s=jnp.zeros_like(s_b),
        gamma_t=jnp.zeros_like(t_b),
        key=None,
    )


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
    alpha_t = jax_helpers.bcast_right(self.schedule.alpha(t), x0.ndim)
    alpha_dot_t = jax_helpers.bcast_right(self.schedule.alpha_dot(t), x0.ndim)
    xt = manifolds.geodesic(self.manifold, x=x1, y=x0, t=alpha_t)
    dxt_dt = alpha_dot_t * self.manifold.velocity(x=x1, y=x0, t=alpha_t)
    return xt, dxt_dt

  def bridge_step(
      self,
      *,
      x0: DataTree,
      xt: DataTree,
      t: TimeTree,
      s: TimeTree,
      key: PRNGKey | None = None,
  ) -> DataTree:
    """Deterministic geodesic bridge: advance along the geodesic toward ``x_0``.

    ``x_t`` and the clean endpoint ``x_0`` lie on a single minimizing
    geodesic (``x_t = geodesic(x_1, x_0, alpha(t))``, ``x_0`` at
    ``alpha = 1``).  The state at ``s < t`` is the same geodesic
    reparametrised on the sub-segment from ``x_t`` to ``x_0``:

        frac = (alpha(s) - alpha(t)) / (1 - alpha(t)),
        x_s  = geodesic(x_t, x_0, frac).

    ``frac = 0`` at ``s = t`` (stay) and ``frac = 1`` at ``s = 0`` (reach
    ``x_0``), matching ``K_{0|0,t} = delta_{x_0}``.  This is the
    deterministic minimizing-geodesic branch; the heat-kernel stochastic
    bridge is manifold-specific and not implemented here.  ``key`` is
    ignored.
    """
    del key
    alpha_s = jax_helpers.bcast_right(self.schedule.alpha(s), x0.ndim)
    alpha_t = jax_helpers.bcast_right(self.schedule.alpha(t), x0.ndim)
    frac = (alpha_s - alpha_t) / jnp.clip(1.0 - alpha_t, 1e-12, None)
    return manifolds.geodesic(self.manifold, x=xt, y=x0, t=frac)


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
    t_b = jax_helpers.bcast_right(t, x0.ndim)
    alpha = self.alpha(t_b)
    beta = self.beta(t_b)
    gamma = self.gamma(t_b)
    alpha_der = jax_helpers.egrad(self.alpha)(t_b)
    beta_der = jax_helpers.egrad(self.beta)(t_b)
    gamma_der = jax_helpers.egrad(self.gamma)(t_b)
    xt = alpha * x0 + beta * x1 + gamma * z
    dxt_dt = alpha_der * x0 + beta_der * x1 + gamma_der * z
    return xt, dxt_dt

  def bridge_step(
      self,
      *,
      x0: DataTree,
      xt: DataTree,
      t: TimeTree,
      s: TimeTree,
      key: PRNGKey | None = None,
  ) -> DataTree:
    r"""Stochastic Brownian-bridge step ``K_{s|0,t}(. | x_0, x_t)``.

    With retention ``rho = beta(s)/beta(t)`` on the terminal-endpoint
    direction,

        x_s = alpha(s) x_0 + rho (x_t - alpha(t) x_0)
              + sqrt(max(gamma(s)^2 - rho^2 gamma(t)^2, 0)) Z.

    This is exact for the canonical Brownian-bridge interpolant
    (``alpha = 1 - lambda``, ``beta = lambda``,
    ``gamma = sigma_br sqrt(lambda(1 - lambda))``), where it equals the
    Posterior Bridges Euclidean step
    ``(1 - rho) x_0 + rho x_t + sigma_br sqrt(lambda_s (1 - rho)) Z``,
    and more generally for any ``(alpha, beta, gamma)`` whose noise is a
    Brownian bridge in the ``beta`` coordinate (the only cross-time
    coupling that the per-time marginals leave well defined).  ``gamma = 0``
    recovers the deterministic affine step (== flow-matching ODE / DDIM).

    A ``key`` is required whenever the bridge carries noise; pass ``None``
    only for the deterministic ``gamma = 0`` case.
    """
    t_b = jax_helpers.bcast_right(t, x0.ndim)
    s_b = jax_helpers.bcast_right(s, x0.ndim)
    return _gaussian_bridge_step(
        x0=x0,
        xt=xt,
        alpha_s=self.alpha(s_b),
        alpha_t=self.alpha(t_b),
        weight_s=self.beta(s_b),
        weight_t=self.beta(t_b),
        gamma_s=self.gamma(s_b),
        gamma_t=self.gamma(t_b),
        key=key,
    )


def canonical_gamma(t: jax.Array) -> jax.Array:
  """``gamma(t) = sqrt(t (1 - t))``: smooth, zero at both endpoints."""
  return jnp.sqrt(jnp.clip(t * (1.0 - t), 0.0, None))
