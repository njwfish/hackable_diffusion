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
  are solved in parallel within a single ``lax.while_loop``.  Requires
  the operator to be symmetric *positive-definite*.
- :func:`batched_minres`: minimum-residual solver for symmetric
  *indefinite* systems (Paige-Saunders 1975).  Same batched-while-loop
  pattern as ``batched_cg``; use it when the Kalman matrix is symmetric
  but its eigenvalues may be negative -- e.g. full Tweedie posterior
  covariance on a non-Bayes-optimal denoiser.
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


def batched_minres(
    matvec: Callable[[jax.Array], jax.Array],
    residual: jax.Array,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> jax.Array:
  """MINRES for symmetric (possibly indefinite) operators, batched per particle.

  Solves ``M_i z_i = residual_i`` for every ``i`` in the leading batch
  axis.  ``M`` is assumed *symmetric* -- if it is also positive-definite
  use :func:`batched_cg` (cheaper per iteration); MINRES is the right
  choice when ``M`` has negative eigenvalues.

  Algorithm follows the Paige-Saunders / scipy formulation: Lanczos
  tridiagonalisation interleaved with a Givens-rotation QR
  factorisation of the resulting tridiagonal.  All per-particle scalars
  and vectors are batched across the leading axis inside a single
  ``lax.while_loop``.  Each iteration costs one matvec plus a constant
  number of inner products and elementwise updates.

  Convergence stops per-particle when ``||r_k|| <= tol * ||r_0||`` or
  after ``max_iter`` iterations; inactive particles freeze.
  """
  ref_ndim = residual.ndim

  def _b(s):
    return _broadcast_scalars(s, ref_ndim)

  zero_v = jnp.zeros_like(residual)
  zero_s = jnp.zeros(residual.shape[0], dtype=residual.dtype)

  beta_init = jnp.sqrt(jnp.maximum(batch_inner(residual, residual), 0.0))
  beta_safe = jnp.maximum(beta_init, 1e-30)

  # Lanczos: r1 = old residual vector (used to build v_{k-1}); r2 = current
  # residual vector (built into v_k via division by beta).
  r1 = residual
  r2 = residual
  init_state = dict(
      x=zero_v,
      r1=r1, r2=r2,
      beta_old=zero_s, beta=beta_init,
      cs=-jnp.ones_like(beta_init), sn=zero_s,
      dbar=zero_s, eps=zero_s,
      phibar=beta_init,
      w=zero_v, w_old=zero_v,
      i=jnp.int32(0),
  )

  rs_init = jnp.maximum(beta_init ** 2, 1e-30)
  tol_sq = tol ** 2

  def cond(state):
    return (state['i'] < max_iter) & jnp.any(
        state['phibar'] ** 2 > tol_sq * rs_init,
    )

  def body(state):
    beta_safe = jnp.maximum(state['beta'], 1e-30)
    v = state['r2'] / _b(beta_safe)
    Av = matvec(v)
    # y = A v - (beta / beta_old) r1, valid only when k > 0; on the first
    # iteration beta_old = 0 so the second term must drop -- handled by
    # the safe-divide below.
    beta_old_safe = jnp.maximum(state['beta_old'], 1e-30)
    coef = jnp.where(state['i'] >= 1, state['beta'] / beta_old_safe, 0.0)
    y = Av - _b(coef) * state['r1']

    alfa = batch_inner(v, y)
    y = y - _b(alfa / beta_safe) * state['r2']

    r1 = state['r2']
    r2 = y
    beta_old = state['beta']
    beta_new = jnp.sqrt(jnp.maximum(batch_inner(r2, r2), 0.0))

    # Apply previous Givens rotation Q_{k-1} to the new tridiagonal column.
    eps_old = state['eps']  # epsilon_{k} from previous iter ("oldeps")
    delta = state['cs'] * state['dbar'] + state['sn'] * alfa
    gbar = state['sn'] * state['dbar'] - state['cs'] * alfa
    eps = state['sn'] * beta_new
    dbar = -state['cs'] * beta_new

    # Compute the new rotation Q_k.
    gamma = jnp.sqrt(jnp.maximum(gbar ** 2 + beta_new ** 2, 0.0))
    gamma_safe = jnp.maximum(gamma, 1e-30)
    cs = gbar / gamma_safe
    sn = beta_new / gamma_safe
    phi = cs * state['phibar']
    phibar_new = sn * state['phibar']

    # Update solution.
    w_new = (v - _b(eps_old) * state['w_old'] - _b(delta) * state['w']) / _b(gamma_safe)
    x_new = state['x'] + _b(phi) * w_new

    # Freeze inactive particles.
    active = state['phibar'] ** 2 > tol_sq * rs_init
    where_v = _b(active.astype(residual.dtype))

    return dict(
        x=jnp.where(where_v > 0, x_new, state['x']),
        r1=r1, r2=r2,
        beta_old=beta_old, beta=beta_new,
        cs=jnp.where(active, cs, state['cs']),
        sn=jnp.where(active, sn, state['sn']),
        dbar=jnp.where(active, dbar, state['dbar']),
        eps=jnp.where(active, eps, state['eps']),
        phibar=jnp.where(active, phibar_new, state['phibar']),
        w=jnp.where(where_v > 0, w_new, state['w']),
        w_old=state['w'],
        i=state['i'] + 1,
    )

  out = jax.lax.while_loop(cond, body, init_state)
  return out['x']


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
