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

"""Posterior-covariance operators ``v -> Cov(x_0 | x_t) v`` for Pi-GDM.

Five choices of ``Cov(x_0 | x_t)`` that cover every published Pi-GDM /
Kalman-guidance variant and scale cleanly to high-dimensional data
(images):

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
  an extra denoiser evaluation per matvec.
- :class:`LowRankTweediePosteriorCovarianceFn`: randomized-SVD
  approximation of the denoiser Jacobian.  Pays ``~(2k + 1) * denoiser``
  setup cost once; then each matvec is ``O(d k + k^2)``.  The right
  choice when the implicit denoiser posterior matters but full-rank
  JVP per CG iteration is too expensive (e.g. many-particle SMC on
  large images).

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
All three assume ``x_0`` lives in a Euclidean space where ``Cov`` is a
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

  def __call__(self, v, *, xt, time, schedule, denoiser_fn=None):
    del xt, denoiser_fn  # state-independent
    alpha, sigma = scalar_alpha_sigma(schedule, time)
    return self.scale_fn(alpha, sigma) * v


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

  def __post_init__(self):
    if (self.prior_covariance is None) == (self.apply_fn is None):
      raise ValueError(
          "Exactly one of ``prior_covariance`` or ``apply_fn`` must be set."
      )

  def _apply_cov(self, v_flat: jax.Array) -> jax.Array:
    if self.apply_fn is not None:
      return self.apply_fn(v_flat)
    return v_flat @ self.prior_covariance.T  # (B, n) @ (n, n)

  def __call__(self, v, *, xt, time, schedule, denoiser_fn=None):
    del xt, denoiser_fn  # state-independent
    alpha, sigma = scalar_alpha_sigma(schedule, time)
    scale = (sigma ** 2) / jnp.maximum(alpha, 1e-8)
    v_flat = v.reshape(v.shape[0], -1)
    out_flat = self._apply_cov(v_flat)
    return scale * out_flat.reshape(v.shape)


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
    # eigh returns ascending eigenvalues; reverse and truncate.
    eigenvalues = eigenvalues[::-1][:num_components]
    eigenvectors = eigenvectors[:, ::-1][:, :num_components]
    singular_values = jnp.sqrt(jnp.clip(eigenvalues, 0.0, None))
    return cls(
        u_factor=eigenvectors,
        singular_values=singular_values,
        regulariser=regulariser,
        scale_fn=scale_fn,
    )

  def __call__(self, v, *, xt, time, schedule, denoiser_fn=None):
    del xt, denoiser_fn  # state-independent
    alpha, sigma = scalar_alpha_sigma(schedule, time)
    scale = self.scale_fn(alpha, sigma)
    v_flat = v.reshape(v.shape[0], -1)  # (B, d)
    coeffs = v_flat @ self.u_factor                  # (B, k)
    if self.singular_values is not None:
      coeffs = coeffs * (self.singular_values ** 2)  # (B, k)
    low_rank = coeffs @ self.u_factor.T              # (B, d)
    out_flat = low_rank + self.regulariser * v_flat
    return scale * out_flat.reshape(v.shape)


@dataclasses.dataclass(kw_only=True, frozen=True)
class TweediePosteriorCovarianceFn(PosteriorCovarianceFn):
  """``Cov(x_0 | x_t) v = (sigma^2 / alpha) * JVP(denoiser_fn, xt, v)``.

  The Miyasawa/Stein identity gives ``Cov(x_0 | x_t) = (sigma^2 / alpha)
  d(xhat_0)/d(xt)``; the JVP computes the product with ``v`` in forward
  mode without materialising the Jacobian.  Exact for any prior the
  denoiser implicitly represents -- Gaussian, mixture, or learned.

  Requires ``denoiser_fn`` to be passed by the caller
  (:class:`KalmanCorrectionFn` threads it through from the sampler).
  """

  def __call__(self, v, *, xt, time, schedule, denoiser_fn=None):
    if denoiser_fn is None:
      raise ValueError(
          "TweediePosteriorCovarianceFn requires ``denoiser_fn`` (a "
          "DenoiserFn closure evaluated at the current time); callers "
          "through KalmanCorrectionFn wire this in automatically."
      )
    alpha, sigma = scalar_alpha_sigma(schedule, time)
    _, jvp = jax.jvp(denoiser_fn, (xt,), (v,))
    return (sigma ** 2 / jnp.maximum(alpha, 1e-8)) * jvp


@dataclasses.dataclass(kw_only=True, frozen=True)
class LowRankTweediePosteriorCovarianceFn(PosteriorCovarianceFn):
  """``Cov v ~= (sigma^2 / alpha) * (Q T Q^T) v`` from randomized-SVD Tweedie.

  Builds a rank-``num_components`` Halko-Martinsson-Tropp sketch of the
  denoiser Jacobian ``J = d(xhat_0)/d(xt)`` at the current ``xt`` and
  applies the resulting ``Q T Q^T`` to ``v``.  Setup cost is
  ``(1 + num_power_iters) * (num_components + oversample) + k`` JVPs;
  each matvec thereafter is ``O(d k + k^2)``.

  Compared to :class:`TweediePosteriorCovarianceFn` (full-rank JVP
  every matvec, ~20x denoiser cost inside a 20-iter CG solve), this
  amortises the denoiser over a one-time setup and leaves CG cheap.
  The accuracy tradeoff is captured by ``num_components``: set it
  large enough to cover the dominant directions of the Jacobian.

  If even this is too expensive -- e.g. many-particle SMC on very
  large images -- a data-agnostic fallback is to replace the
  randomized-SVD sketch with a plain fixed random projection
  ``Omega: (d, k)``: ``Cov v = (sigma^2/alpha) * Omega (Omega^T v) / k``.
  That has no Jacobian information but costs one matvec instead of
  ``O(k)`` denoiser evals.  Add a sibling class if you need it.

  Setup is re-run per call; this means each matvec inside a CG solve
  regenerates the sketch.  If you know the denoiser is locally linear
  over the CG trajectory, precompute ``(Q, T)`` in the calling
  correction instead -- but most denoisers are not, so the default is
  the safer per-call sketch.
  """

  num_components: int
  oversample: int = 5
  num_power_iters: int = 1
  rng_key: jax.Array = dataclasses.field(
      default_factory=lambda: jax.random.PRNGKey(0),
  )

  def __call__(self, v, *, xt, time, schedule, denoiser_fn=None):
    if denoiser_fn is None:
      raise ValueError(
          'LowRankTweediePosteriorCovarianceFn requires ``denoiser_fn``; '
          'KalmanCorrectionFn threads it in automatically.'
      )
    alpha, sigma = scalar_alpha_sigma(schedule, time)
    scale = (sigma ** 2) / jnp.maximum(alpha, 1e-8)

    def jvp_fn(w):
      _, out = jax.jvp(denoiser_fn, (xt,), (w,))
      return out

    # ``rng_key`` is a concrete array stored in the dataclass; the sketch
    # is deterministic and reproducible across calls (same key every
    # matvec inside a CG solve, which is fine -- the sketch averages
    # noise across components so a single fixed key is sufficient).
    q, t = randomized_svd_jvp(
        jvp_fn, xt,
        num_components=self.num_components,
        key=self.rng_key,
        oversample=self.oversample,
        num_power_iters=self.num_power_iters,
    )
    # q: (B, d, k); t: (B, k, k); v: (B, *spatial).
    v_flat = v.reshape(v.shape[0], -1)                 # (B, d)
    coeffs = jnp.einsum('bdk,bd->bk', q, v_flat)       # (B, k)
    coeffs = jnp.einsum('bij,bj->bi', t, coeffs)       # (B, k)
    out_flat = jnp.einsum('bdk,bk->bd', q, coeffs)     # (B, d)
    return scale * out_flat.reshape(v.shape)
