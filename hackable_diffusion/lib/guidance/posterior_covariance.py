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

Three choices of ``Cov(x_0 | x_t)`` that cover every published Pi-GDM /
Kalman-guidance variant:

- :class:`IsotropicPosteriorCovarianceFn`: ``Cov = scale(alpha, sigma) I``.
  The simplest projection-style correction; ``scale = sigma^2/alpha``
  (Miyasawa) is the Gaussian-prior exact value.
- :class:`FixedPriorPosteriorCovarianceFn`: ``Cov = (sigma^2/alpha) C``
  for a known prior covariance ``C``.  Exact under a Gaussian prior with
  covariance ``C``; gives the "cov-aware" correction in the literature.
- :class:`TweediePosteriorCovarianceFn`: ``Cov v = (sigma^2/alpha) *
  JVP(denoiser_x0, xt, v)`` via the Miyasawa/Stein identity.  Exact
  under *any* prior the denoiser implicitly represents, at the cost of
  an extra denoiser evaluation per matvec.

All three satisfy :class:`PosteriorCovarianceFn` and so plug into
:class:`PiGDMCorrectionFn` interchangeably.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp

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

  def __call__(self, v, *, xt, time, schedule, denoiser_x0=None):
    del xt, denoiser_x0  # state-independent
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

  def __call__(self, v, *, xt, time, schedule, denoiser_x0=None):
    del xt, denoiser_x0  # state-independent
    alpha, sigma = scalar_alpha_sigma(schedule, time)
    scale = (sigma ** 2) / jnp.maximum(alpha, 1e-8)
    v_flat = v.reshape(v.shape[0], -1)
    out_flat = self._apply_cov(v_flat)
    return scale * out_flat.reshape(v.shape)


@dataclasses.dataclass(kw_only=True, frozen=True)
class TweediePosteriorCovarianceFn(PosteriorCovarianceFn):
  """``Cov(x_0 | x_t) v = (sigma^2 / alpha) * JVP(denoiser_x0, xt, v)``.

  The Miyasawa/Stein identity gives ``Cov(x_0 | x_t) = (sigma^2 / alpha)
  d(xhat_0)/d(xt)``; the JVP computes the product with ``v`` in forward
  mode without materialising the Jacobian.  Exact for any prior the
  denoiser implicitly represents -- Gaussian, mixture, or learned.

  Requires ``denoiser_x0`` to be passed by the caller (the PiGDM
  correction threads it through from the sampler).
  """

  def __call__(self, v, *, xt, time, schedule, denoiser_x0=None):
    if denoiser_x0 is None:
      raise ValueError(
          "TweediePosteriorCovarianceFn requires ``denoiser_x0`` (an xt -> "
          "xhat_0 closure evaluated at the current time); callers through "
          "PiGDMCorrectionFn wire this in automatically."
      )
    alpha, sigma = scalar_alpha_sigma(schedule, time)
    _, jvp = jax.jvp(denoiser_x0, (xt,), (v,))
    return (sigma ** 2 / jnp.maximum(alpha, 1e-8)) * jvp
