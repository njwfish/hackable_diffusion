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

Two primitives used by :class:`PiGDMCorrectionFn` and its extensions:

- :func:`batch_inner`: per-particle dot product summed over all
  non-batch axes.
- :func:`batched_cg`: conjugate-gradient solver running one independent
  system per batch element.  Every per-particle inner-product and scalar
  update is taken elementwise along the batch axis, so B sub-problems
  are solved in parallel within a single ``lax.while_loop``.
- :func:`linear_adjoint`: return the adjoint map ``A^T`` of a linear
  ``ForwardFn`` at a given input via ``jax.vjp``.
"""

from __future__ import annotations

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
