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

"""Linear-algebra helpers for hard linear observations.

For a linear observation ``Y = A X_0`` and an approximate clean
posterior ``X_0 | X_t = x_t ~= N(m_t(x_t), C_t(x_t))``, the Doob
transform for the hard observation ``Y = y`` involves three building
blocks: the row-space pseudo-inverse of ``A C_t A^T``, the singular-
Gaussian log-density on the row-space, and the materialisation of
``A C_t A^T`` itself.  This module exposes those building blocks;
the consumers live in
:mod:`hackable_diffusion.lib.guidance.corrections` (the pinv path of
:class:`KalmanCorrectionFn`) and
:mod:`hackable_diffusion.lib.guidance.twists` (the
:class:`PosteriorPredictiveGaussianTwistFn`).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from hackable_diffusion.lib.guidance.linalg import linear_adjoint
from hackable_diffusion.lib.guidance.protocols import ForwardFn


def _broadcast_observation(observation: jax.Array, predicted: jax.Array) -> jax.Array:
  obs = jnp.asarray(observation, dtype=predicted.dtype)
  return jnp.broadcast_to(obs, predicted.shape)


def _flatten_observation(value: jax.Array) -> jax.Array:
  return value.reshape(value.shape[0], -1)


def _materialize_observation_covariance(
    *,
    forward_fn: ForwardFn,
    cov_matvec,
    x0: jax.Array,
) -> tuple[jax.Array, Any, tuple[int, ...]]:
  """Return ``(A C A^T, A^T, observation_shape)`` for each batch item."""
  predicted = forward_fn.forward(x0)
  observation_shape = tuple(predicted.shape[1:])
  obs_dim = math.prod(observation_shape)
  adjoint = linear_adjoint(forward_fn, x0)
  eye = jnp.eye(obs_dim, dtype=predicted.dtype)

  def apply_column(e_flat: jax.Array) -> jax.Array:
    e_obs = jnp.broadcast_to(e_flat.reshape((1,) + observation_shape), predicted.shape)
    lifted = adjoint(e_obs)
    return _flatten_observation(forward_fn.forward(cov_matvec(lifted)))

  columns = jax.vmap(apply_column)(eye)  # (obs_dim, B, obs_dim)
  system = jnp.transpose(columns, (1, 2, 0))
  system = 0.5 * (system + jnp.swapaxes(system, -1, -2))
  return system, adjoint, observation_shape


def _pinv_eigendecomposition(
    system: jax.Array,
    *,
    rtol: float,
    atol: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  eigvals, eigvecs = jnp.linalg.eigh(system)
  max_eval = jnp.max(jnp.abs(eigvals), axis=-1, keepdims=True)
  threshold = jnp.maximum(
      jnp.asarray(atol, dtype=system.dtype),
      jnp.asarray(rtol, dtype=system.dtype) * max_eval,
  )
  active = eigvals > threshold
  inv_eigvals = jnp.where(active, 1.0 / jnp.maximum(eigvals, threshold), 0.0)
  return eigvals, eigvecs, active, inv_eigvals


def psd_pinv_solve(
    system: jax.Array,
    rhs: jax.Array,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> jax.Array:
  """Solve ``system^+ rhs`` for batched symmetric PSD systems."""
  _, eigvecs, _, inv_eigvals = _pinv_eigendecomposition(
      system, rtol=rtol, atol=atol,
  )
  coeffs = jnp.einsum("bmi,bm->bi", eigvecs, rhs)
  return jnp.einsum("bmi,bi->bm", eigvecs, inv_eigvals * coeffs)


def singular_gaussian_logpdf(
    residual: jax.Array,
    covariance: jax.Array,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    enforce_support: bool = True,
    support_atol: float = 1e-5,
    support_rtol: float = 1e-4,
) -> jax.Array:
  """Log density of a batched singular Gaussian on its affine support."""
  eigvals, eigvecs, active, inv_eigvals = _pinv_eigendecomposition(
      covariance, rtol=rtol, atol=atol,
  )
  coeffs = jnp.einsum("bmi,bm->bi", eigvecs, residual)
  quad = jnp.sum(jnp.where(active, coeffs ** 2 * inv_eigvals, 0.0), axis=-1)
  logdet = jnp.sum(jnp.where(active, jnp.log(jnp.maximum(eigvals, atol)), 0.0), axis=-1)
  rank = jnp.sum(active.astype(residual.dtype), axis=-1)
  logp = -0.5 * (quad + logdet + rank * jnp.log(2.0 * jnp.pi))

  if enforce_support:
    projected = jnp.einsum("bmi,bi->bm", eigvecs, jnp.where(active, coeffs, 0.0))
    off_support = residual - projected
    off_support_norm = jnp.linalg.norm(off_support, axis=-1)
    residual_norm = jnp.linalg.norm(residual, axis=-1)
    tolerance = float(support_atol) + float(support_rtol) * jnp.maximum(residual_norm, 1.0)
    logp = jnp.where(off_support_norm <= tolerance, logp, -jnp.inf)
  return logp


