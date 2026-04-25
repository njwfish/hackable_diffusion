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

"""Linear-algebra helpers for guidance corrections.

Primitives used by :class:`PiGDMCorrectionFn` and its extensions:

- :func:`batch_inner`: per-particle dot product summed over all
  non-batch axes.
- :func:`batched_cg`: conjugate-gradient solver running one independent
  system per batch element.  Every per-particle inner-product and scalar
  update is taken elementwise along the batch axis, so B sub-problems
  are solved in parallel within a single ``lax.while_loop``.
- :func:`linear_adjoint`: return the adjoint map ``A^T`` of a linear
  ``ForwardFn`` at a given input via ``jax.vjp``.
- :func:`randomized_svd_jvp`: rank-``k`` randomized SVD of a linear
  operator given by a JVP callable, in the Halko-Martinsson-Tropp
  sketch.  Used by :class:`LowRankTweediePosteriorCovarianceFn` to
  approximate the denoiser Jacobian from a handful of JVPs.
"""

from __future__ import annotations

import math
from typing import Callable

import jax
import jax.numpy as jnp

from hackable_diffusion.lib.guidance.protocols import ForwardFn


def batch_inner(x: jax.Array, y: jax.Array) -> jax.Array:
  """Per-particle inner product summed over all non-batch axes.

  ``x, y`` share the same shape ``(B, *spatial)``.  Returns shape ``(B,)``.
  """
  return jnp.sum(
      x.reshape(x.shape[0], -1) * y.reshape(y.shape[0], -1), axis=-1,
  )


def _broadcast_scalars(values: jax.Array, target_ndim: int) -> jax.Array:
  """Reshape ``(B,)`` to broadcast against a rank-``target_ndim`` tensor."""
  return values.reshape((values.shape[0],) + (1,) * (target_ndim - 1))


def batched_cg(
    matvec: Callable[[jax.Array], jax.Array],
    residual: jax.Array,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> jax.Array:
  """Conjugate gradient with one independent solve per batch element.

  Solves ``M_i z_i = residual_i`` for every ``i`` in the leading batch
  axis, where ``matvec(p)`` applies the batched operator ``M`` to every
  particle simultaneously.  ``M`` is assumed positive-definite (or PSD
  with consistent RHS); inactive particles -- those already below
  tolerance -- are zeroed out of the updates so they don't degrade the
  active particles' convergence.
  """
  z = jnp.zeros_like(residual)
  r = residual
  p = r
  rs_old = batch_inner(r, r)
  rs_init = jnp.maximum(rs_old, 1e-12)
  tol_sq = tol ** 2

  def cond(state):
    _, _, _, rs_old, rs_init, i = state
    return (i < max_iter) & jnp.any(rs_old > tol_sq * rs_init)

  def body(state):
    z, r, p, rs_old, rs_init, i = state
    mp = matvec(p)
    pmp = batch_inner(p, mp)
    active = rs_old > tol_sq * rs_init

    alpha = jnp.where(active & (pmp > 0), rs_old / pmp, 0.0)
    z = z + _broadcast_scalars(alpha, residual.ndim) * p
    r = r - _broadcast_scalars(alpha, residual.ndim) * mp

    rs_new = batch_inner(r, r)
    beta = jnp.where(active & (rs_old > 0), rs_new / rs_old, 0.0)
    p = r + _broadcast_scalars(beta, residual.ndim) * p
    return z, r, p, rs_new, rs_init, i + 1

  init = (z, r, p, rs_old, rs_init, jnp.int32(0))
  z, *_ = jax.lax.while_loop(cond, body, init)
  return z


def linear_adjoint(
    forward_fn: ForwardFn, x: jax.Array,
) -> Callable[[jax.Array], jax.Array]:
  """Return the adjoint ``A^T`` of a linear ``ForwardFn`` via VJP at ``x``.

  For a linear operator the VJP at any primal equals the adjoint, so we
  use ``x`` only to pin a shape; the returned closure is independent of
  it.  Non-linear forwards yield a Jacobian-transpose at ``x``, which is
  the correct local adjoint for gradient-based methods but not the
  globally-linear one Pi-GDM's closed form assumes.
  """
  _, vjp = jax.vjp(forward_fn.forward, x)
  return lambda w: vjp(w)[0]


def randomized_svd_jvp(
    jvp_fn: Callable[[jax.Array], jax.Array],
    example: jax.Array,
    *,
    num_components: int,
    key: jax.Array,
    oversample: int = 5,
    num_power_iters: int = 1,
) -> tuple[jax.Array, jax.Array]:
  """Rank-``num_components`` randomized SVD of a linear operator given via JVP.

  The operator ``A`` is specified implicitly by ``jvp_fn(v) = A v`` (shape
  must match ``example``).  Returns ``(Q, T)`` where ``Q`` has
  orthonormal columns spanning a good approximation of the top
  left-singular-vectors of ``A`` and ``T = Q^T A Q`` is the rank-``k``
  compression.  The approximation ``A v ~= Q (T (Q^T v))`` costs
  ``O(d * k)`` per matvec after setup.

  Setup cost is ``(1 + num_power_iters) * (k + oversample) + k`` JVPs
  plus a QR on a ``(d, k)`` tall-skinny matrix.  For the Halko-Martinsson
  -Tropp sketch, one power iteration is typically enough to get
  meaningful accuracy on well-conditioned operators.

  If this is still too expensive, a pure-random sketch without power
  iteration approximates the prior covariance rather than the
  Jacobian; see :class:`RandomProjectionPosteriorCovarianceFn` (not
  currently implemented).

  Shapes: ``example`` is ``(B, *spatial)``; Q and T are per-batch
  tensors shaped ``(B, d, k)`` and ``(B, k, k)`` where
  ``d = prod(spatial)``.  The batch dim is preserved so each particle
  gets its own low-rank factorisation.
  """
  batch = example.shape[0]
  spatial_shape = example.shape[1:]
  d = math.prod(spatial_shape)  # static Python computation; survives tracing
  k_eff = min(num_components, d)
  k_sketch = min(k_eff + oversample, d)

  # Draw one (d, k_sketch) random matrix per batch element.
  omega = jax.random.normal(
      key, (batch, d, k_sketch), dtype=example.dtype,
  )

  def apply_to_column_stack(stacked: jax.Array) -> jax.Array:
    """Apply ``jvp_fn`` to each of ``stacked``'s trailing columns.

    ``stacked`` has shape ``(B, d, k)`` for any ``k``; reshape each
    column back to ``(B, *spatial)``, apply JVP, flatten, stack.  Uses
    ``jax.vmap`` over the column axis so the JVPs run in parallel.
    """
    k = stacked.shape[-1]
    as_particles = stacked.reshape(batch, *spatial_shape, k)
    as_particles = jnp.moveaxis(as_particles, -1, 0)  # (k, B, *spatial)
    out = jax.vmap(jvp_fn)(as_particles)              # (k, B, *spatial)
    out = jnp.moveaxis(out, 0, -1)                    # (B, *spatial, k)
    return out.reshape(batch, d, k)

  y = apply_to_column_stack(omega)
  for _ in range(num_power_iters):
    # Power iteration: Y <- A (A^T (A Omega)) sharpens the top singular
    # directions.  We only have a forward JVP; skip the adjoint pass
    # and settle for A A^T = I approximation (valid when the operator
    # is already close to symmetric, e.g. posterior covariance).
    y = apply_to_column_stack(y)

  # Batched QR of ``(B, d, k_sketch)`` then truncate to k_eff columns.
  # When ``d <= k_sketch``, ``qr`` returns a (d, d) Q; truncate accordingly.
  q_full, _ = jnp.linalg.qr(y)                        # (B, d, min(d, k_sketch))
  q = q_full[..., :k_eff]                             # (B, d, k_eff)

  # Compute T = Q^T A Q: apply A to Q then project.
  aq = apply_to_column_stack(q)                       # (B, d, k_eff)
  t = jnp.einsum('bdi,bdj->bij', q, aq)               # (B, k_eff, k_eff)
  return q, t
