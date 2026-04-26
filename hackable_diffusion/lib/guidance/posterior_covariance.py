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

"""Posterior-covariance matvec factories for Pi-GDM / Kalman guidance.

Each :class:`PosteriorCovarianceFn` is called once per step with
``(xt, time, schedule, denoiser_fn)`` and returns a closure ``v -> Cov v``.
The factory pattern hoists any one-time setup (e.g. randomized-SVD
sketch of the denoiser Jacobian) out of CG's inner loop -- the closure
is reused across all CG iterations for a single step.

Five variants cover every published Pi-GDM / Kalman-guidance variant
and scale cleanly to high-dimensional data (images):

- :class:`IsotropicPosteriorCovarianceFn`: ``Cov = scale(alpha, sigma) I``.
  The simplest projection-style correction; ``scale = sigma^2/alpha``
  (Miyasawa) is the Gaussian-prior exact value.
- :class:`FixedPriorPosteriorCovarianceFn`: ``Cov = (sigma^2/alpha) C``
  for a known prior covariance ``C``.  Exact under a Gaussian prior with
  covariance ``C``; gives the "cov-aware" correction in the literature.
- :class:`PCAPosteriorCovarianceFn`: ``Cov = (sigma^2/alpha) (U diag(s^2)
  U^T + eps I)``.  Rank-``k`` approximation built from a precomputed
  factor ``U: (d, k)`` (typically PCA of a training-data sample).
  ``O(d k)`` matvec, no denoiser calls during guidance; the optimal
  choice when a fixed representative data covariance is available.
- :class:`TweediePosteriorCovarianceFn`: ``Cov v = (sigma^2/alpha) *
  JVP(denoiser_x0, xt, v)`` via the Miyasawa/Stein identity.  Exact
  under *any* prior the denoiser implicitly represents, at the cost of
  one denoiser-derivative per matvec.
- :class:`LowRankTweediePosteriorCovarianceFn`: randomized-SVD
  approximation of the denoiser Jacobian.  Pays ``~(2 + p)(k + os)``
  JVPs setup cost once per step (factory-time); each subsequent matvec
  inside CG is ``O(d k + k^2)`` with no further denoiser calls.  The
  right choice when the implicit denoiser posterior matters but
  full-rank JVP per CG iteration is too expensive.

All five satisfy :class:`PosteriorCovarianceFn` and so plug into
:class:`KalmanCorrectionFn` interchangeably.

A cheaper still fallback, not currently implemented, would be a
fixed random projection ``Omega: (d, k)`` with
``Cov = (sigma^2/alpha) Omega Omega^T / k``.  This has no power over
the true posterior covariance but costs only ``O(d k)`` per matvec
with no setup -- useful as a sanity baseline.  Add as a sibling class
if needed.

Modality compatibility
----------------------
All five assume ``x_0`` lives in a Euclidean space where ``Cov`` is a
well-defined linear operator against the canonical inner product.  That
covers Gaussian (ODE + SDE) and distributional diffusion.  For
simplicial diffusion the "covariance" would need to be defined in the
tangent space of the simplex -- use a simplex-aware correction instead.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp

from hackable_diffusion.lib.guidance.linalg import randomized_svd_jvp
from hackable_diffusion.lib.guidance.protocols import PosteriorCovarianceFn
from hackable_diffusion.lib.guidance.utils import scalar_alpha_sigma


# Signature of a scalar schedule -> scale callable used by the isotropic
# variant.  Returns a 0-d array.
ScaleFn = Callable[[jax.Array, jax.Array], jax.Array]


def miyasawa_scale(alpha: jax.Array, sigma: jax.Array) -> jax.Array:
  """Tweedie-identity scalar ``sigma^2 / alpha`` (Gaussian-prior value)."""
  return (sigma ** 2) / jnp.maximum(alpha, 1e-8)


@dataclasses.dataclass(kw_only=True, frozen=True)
class IsotropicPosteriorCovarianceFn(PosteriorCovarianceFn):
  """``Cov(x_0 | x_t) = scale(alpha, sigma) * I``.

  Default ``scale_fn`` is :func:`miyasawa_scale` (``sigma^2 / alpha``),
  which is the Gaussian-white-prior posterior variance under the
  Miyasawa identity.  Any callable ``(alpha, sigma) -> scalar`` can be
  plugged in -- e.g. a constant ``strength`` for the classic DPS
  projection.
  """

  scale_fn: ScaleFn = miyasawa_scale

  def __call__(self, *, xt, time, schedule, denoiser_fn=None):
    del xt, denoiser_fn  # state-independent
    alpha, sigma = scalar_alpha_sigma(schedule, time)
    scale = self.scale_fn(alpha, sigma)
    return lambda v: scale * v


@dataclasses.dataclass(kw_only=True, frozen=True)
class FixedPriorPosteriorCovarianceFn(PosteriorCovarianceFn):
  """``Cov(x_0 | x_t) = (sigma^2/alpha) * C`` for a known prior covariance.

  Exact under a zero-mean Gaussian prior with covariance ``C``.
  ``prior_covariance`` may be either the full dense matrix (``(n, n)``)
  or a linear operator callable -- pass ``apply_fn = lambda v: C @ v``
  to support structured priors (FFT-diagonal, block-Toeplitz, etc.)
  without materialising ``C``.

  Only one of ``prior_covariance`` or ``apply_fn`` should be set.  The
  operator is applied per-batch to the flattened spatial axes.
  """

  prior_covariance: jax.Array | None = None
  apply_fn: Callable[[jax.Array], jax.Array] | None = None
  scale_fn: ScaleFn = miyasawa_scale

  def __post_init__(self):
    if (self.prior_covariance is None) == (self.apply_fn is None):
      raise ValueError(
          "Exactly one of ``prior_covariance`` or ``apply_fn`` must be set."
      )

  def __call__(self, *, xt, time, schedule, denoiser_fn=None):
    del xt, denoiser_fn  # state-independent
    alpha, sigma = scalar_alpha_sigma(schedule, time)
    scale = self.scale_fn(alpha, sigma)
    apply_cov = (
        self.apply_fn if self.apply_fn is not None
        else (lambda v_flat: v_flat @ self.prior_covariance.T)
    )

    def matvec(v: jax.Array) -> jax.Array:
      v_flat = v.reshape(v.shape[0], -1)
      out_flat = apply_cov(v_flat)
      return scale * out_flat.reshape(v.shape)
    return matvec


@dataclasses.dataclass(kw_only=True, frozen=True)
class PCAPosteriorCovarianceFn(PosteriorCovarianceFn):
  """``Cov = (sigma^2/alpha) (U diag(s^2) U^T + eps I)``.

  Rank-``k`` data-prior covariance built from a precomputed factor
  ``u_factor: (d, k)`` (orthonormal columns) and optional singular
  values ``singular_values: (k,)``.  ``eps`` is an optional isotropic
  regulariser that prevents Cov from being singular in the null space
  of ``U`` -- essential when the Kalman solve's CG matrix
  ``A Cov A^T + sigma_y^2 I`` would otherwise be rank-deficient.

  Factory :meth:`from_covariance` takes a full prior covariance
  ``C: (d, d)`` and eigendecomposes it into the top-``k`` eigenvectors.
  For images, compute ``C`` from a representative sample of the
  training distribution and truncate to whatever ``k`` fits in memory.

  Shapes: the factor is flattened against the non-batch axes, so the
  adapter works unchanged on rank-2 (distributional) or rank-4
  (images, ``(B, H, W, C)``) inputs.  ``u_factor`` is stored as
  ``(d, k)`` and matvec uses ``v.reshape(B, -1)``.
  """

  u_factor: jax.Array  # (d, k)
  singular_values: jax.Array | None = None  # (k,) or None for unit weight
  regulariser: float = 0.0
  scale_fn: ScaleFn = miyasawa_scale

  def __post_init__(self):
    if self.u_factor.ndim != 2:
      raise ValueError(
          f'u_factor must be (d, k); got shape {self.u_factor.shape}.'
      )
    if (self.singular_values is not None
        and self.singular_values.shape[0] != self.u_factor.shape[1]):
      raise ValueError(
          f'singular_values length {self.singular_values.shape[0]} != '
          f'u_factor rank {self.u_factor.shape[1]}.'
      )

  @classmethod
  def from_covariance(
      cls,
      covariance: jax.Array,
      *,
      num_components: int,
      regulariser: float = 0.0,
      scale_fn: ScaleFn = miyasawa_scale,
  ) -> 'PCAPosteriorCovarianceFn':
    """Build from a full covariance via top-``k`` eigendecomposition.

    Sorts eigenvalues descending, takes the top ``num_components``, and
    stores ``(eigenvectors, sqrt(eigenvalues))`` so that
    ``U diag(s^2) U^T`` reconstructs the rank-``k`` approximation.
    Assumes ``covariance`` is symmetric positive semi-definite.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
    eigenvalues = eigenvalues[::-1][:num_components]
    eigenvectors = eigenvectors[:, ::-1][:, :num_components]
    singular_values = jnp.sqrt(jnp.clip(eigenvalues, 0.0, None))
    return cls(
        u_factor=eigenvectors,
        singular_values=singular_values,
        regulariser=regulariser,
        scale_fn=scale_fn,
    )

  def __call__(self, *, xt, time, schedule, denoiser_fn=None):
    del xt, denoiser_fn  # state-independent
    alpha, sigma = scalar_alpha_sigma(schedule, time)
    scale = self.scale_fn(alpha, sigma)
    u = self.u_factor
    sv = self.singular_values
    reg = self.regulariser

    def matvec(v: jax.Array) -> jax.Array:
      v_flat = v.reshape(v.shape[0], -1)
      coeffs = v_flat @ u                          # (B, k)
      if sv is not None:
        coeffs = coeffs * (sv ** 2)
      low_rank = coeffs @ u.T                      # (B, d)
      out_flat = low_rank + reg * v_flat
      return scale * out_flat.reshape(v.shape)
    return matvec


@dataclasses.dataclass(kw_only=True, frozen=True)
class TweediePosteriorCovarianceFn(PosteriorCovarianceFn):
  """``Cov(x_0 | x_t) v = (sigma^2 / alpha) * (sym J) v`` via JVP+VJP.

  The Miyasawa/Stein identity gives ``Cov(x_0 | x_t) = (sigma^2 / alpha)
  d(xhat_0)/d(xt)``.  For a Bayes-optimal denoiser ``J`` is symmetric
  PSD; for a learned one it generally is *not*, which makes the
  Kalman CG matrix indefinite and the inpainting reconstruction
  divergent.

  ``symmetrize = True`` (the default) computes ``(J v + J^T v) / 2``
  via one JVP and one VJP -- twice the cost but valid as a
  symmetric operator on any denoiser.  PSD-ness is not enforced
  (would require an eigendecomposition every matvec); when the
  symmetrised Jacobian still has negative eigenvalues, prefer
  :class:`LowRankTweediePosteriorCovarianceFn` with
  ``project_psd=True``.
  """

  symmetrize: bool = True
  scale_fn: ScaleFn = miyasawa_scale

  def __call__(self, *, xt, time, schedule, denoiser_fn=None):
    if denoiser_fn is None:
      raise ValueError(
          "TweediePosteriorCovarianceFn requires ``denoiser_fn`` (a "
          "DenoiserFn closure evaluated at the current time); callers "
          "through KalmanCorrectionFn wire this in automatically."
      )
    alpha, sigma = scalar_alpha_sigma(schedule, time)
    scale = self.scale_fn(alpha, sigma)

    if self.symmetrize:
      _, vjp_fn = jax.vjp(denoiser_fn, xt)

      def matvec(v: jax.Array) -> jax.Array:
        _, jv = jax.jvp(denoiser_fn, (xt,), (v,))
        (jtv,) = vjp_fn(v)
        return scale * 0.5 * (jv + jtv)
    else:
      def matvec(v: jax.Array) -> jax.Array:
        _, jv = jax.jvp(denoiser_fn, (xt,), (v,))
        return scale * jv
    return matvec


@dataclasses.dataclass(kw_only=True, frozen=True)
class LowRankTweediePosteriorCovarianceFn(PosteriorCovarianceFn):
  """``Cov v ~= (sigma^2 / alpha) * (Q T Q^T) v`` from randomized-SVD Tweedie.

  Builds a rank-``num_components`` Halko-Martinsson-Tropp sketch of the
  denoiser Jacobian ``J = d(xhat_0)/d(xt)`` at the current ``xt``,
  caching ``(Q, T)`` in the returned closure.  Each subsequent matvec
  inside CG is then ``O(d k + k^2)`` with no further denoiser calls --
  the whole point of the factory pattern.

  ``symmetrize = True`` (default) replaces the small ``T = Q^T J Q``
  matrix with ``(T + T^T)/2`` -- cheap, k x k -- so the operator is
  symmetric on any denoiser.  ``project_psd = True`` (default) goes
  one step further and clips negative eigenvalues of ``T_sym`` to
  zero (or a small floor), projecting onto the PSD cone.  Both cost
  ``O(k^3)`` once at setup and are *essential* on real trained
  denoisers whose Jacobians are neither symmetric nor PSD.

  Setup cost (one-time, factory-time): ``(1 + num_power_iters) *
  (num_components + oversample) + num_components`` JVPs plus an
  ``O(k^3)`` symmetric eigendecomposition when ``project_psd``.

  Compared to :class:`TweediePosteriorCovarianceFn` (full-rank JVP+VJP
  every matvec), this amortises the denoiser over a one-time setup;
  the break-even point is ``cg_max_iter ~ k + 2 oversample`` for
  ``num_power_iters=1``.

  If even this is too expensive -- e.g. many-particle SMC on very
  large images -- a data-agnostic fallback is to replace the
  randomized-SVD sketch with a plain fixed random projection
  ``Omega: (d, k)``: ``Cov v = (sigma^2/alpha) * Omega (Omega^T v) / k``.
  That has no Jacobian information but skips the JVP setup entirely.
  Add a sibling class if you need it.
  """

  num_components: int
  oversample: int = 5
  num_power_iters: int = 1
  symmetrize: bool = True
  project_psd: bool = True
  psd_floor: float = 0.0
  scale_fn: ScaleFn = miyasawa_scale
  rng_key: jax.Array = dataclasses.field(
      default_factory=lambda: jax.random.PRNGKey(0),
  )

  def __call__(self, *, xt, time, schedule, denoiser_fn=None):
    if denoiser_fn is None:
      raise ValueError(
          'LowRankTweediePosteriorCovarianceFn requires ``denoiser_fn``; '
          'KalmanCorrectionFn threads it in automatically.'
      )
    alpha, sigma = scalar_alpha_sigma(schedule, time)
    scale = self.scale_fn(alpha, sigma)

    def jvp_fn(w):
      _, out = jax.jvp(denoiser_fn, (xt,), (w,))
      return out

    # Sketch built ONCE, captured in the closure below.
    q, t = randomized_svd_jvp(
        jvp_fn, xt,
        num_components=self.num_components,
        key=self.rng_key,
        oversample=self.oversample,
        num_power_iters=self.num_power_iters,
    )

    if self.symmetrize:
      t = 0.5 * (t + jnp.swapaxes(t, -1, -2))
    if self.project_psd:
      # Symmetric eigendecompose, clip negatives to ``psd_floor``,
      # rebuild.  k is small (typically <= 256) so eigh is cheap.
      eigvals, eigvecs = jnp.linalg.eigh(t)
      eigvals = jnp.maximum(eigvals, self.psd_floor)
      t = jnp.einsum(
          'bij,bj,bkj->bik', eigvecs, eigvals, eigvecs,
      )

    def matvec(v: jax.Array) -> jax.Array:
      v_flat = v.reshape(v.shape[0], -1)              # (B, d)
      coeffs = jnp.einsum('bdk,bd->bk', q, v_flat)    # (B, k)
      coeffs = jnp.einsum('bij,bj->bi', t, coeffs)    # (B, k)
      out_flat = jnp.einsum('bdk,bk->bd', q, coeffs)  # (B, d)
      return scale * out_flat.reshape(v.shape)
    return matvec
