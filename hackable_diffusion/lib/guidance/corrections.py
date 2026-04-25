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

"""Observation-driven ``x_0 -> x_0_new`` corrections.

Three primitives cover every posterior-sampling method:

- :class:`KalmanCorrectionFn`: closed-form Kalman update
  ``x_0 + Sigma A^T (A Sigma A^T + sigma_y^2 I)^{-1} (y - A x_0)``
  parameterised over a :class:`ForwardFn` and a
  :class:`PosteriorCovarianceFn`.  Pi-GDM family.
- :class:`GradientCorrectionFn`: first-order shift
  ``x_0 + prefactor * grad_{xt} log psi``.  DPS family.
- :class:`IteratedCorrectionFn`: wrap any base correction to apply it
  ``num_iters`` times with denoiser re-evaluation at each shifted xt.

All three satisfy :class:`CorrectionFn`: they take ``(x0, xt, time)`` and
a :class:`DenoiserFn` + schedule, and return the new ``x0``.  No
``corruption_process``, no raw ``inference_fn``, no ``rng``/``conditioning``
plumbing -- those are captured inside ``denoiser_fn``.

Modality compatibility
----------------------
- ``KalmanCorrectionFn``: Euclidean-x0 only (Gaussian ODE / SDE /
  distributional).  Kalman math assumes a standard inner product.
- ``GradientCorrectionFn``: Euclidean-x0 in principle (the gradient
  step is a Euclidean shift).  On a simplex the Euclidean step leaves
  the constraint set; use a simplex-aware correction.
- ``IteratedCorrectionFn``: inherits modality from its ``base``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import jax
import jax.numpy as jnp

from hackable_diffusion.lib.guidance.linalg import batched_cg, linear_adjoint
from hackable_diffusion.lib.guidance.protocols import (
    CorrectionFn,
    DenoiserFn,
    ForwardFn,
    PosteriorCovarianceFn,
    TwistFn,
)
from hackable_diffusion.lib.guidance.utils import (
    scalar_alpha,
    scalar_alpha_sigma,
)


################################################################################
# MARK: Prefactors for GradientCorrectionFn
################################################################################


def miyasawa_prefactor(
    *,
    alpha: jax.Array,
    sigma: jax.Array,
    xt: jax.Array,
    x0: jax.Array,
) -> jax.Array:
  """Tweedie-identity prefactor ``sigma^2 / alpha``.

  Exact in the single-Gaussian-prior limit via Miyasawa:
  ``Cov(x0 | xt) = (sigma^2 / alpha) d xhat_0/d xt``.
  """
  del xt, x0
  return (sigma ** 2) / jnp.maximum(alpha, 1e-8)


def dps_prefactor(
    *,
    alpha: jax.Array,
    sigma: jax.Array,
    xt: jax.Array,
    x0: jax.Array,
) -> jax.Array:
  """Canonical DPS step ``1 / ||residual||``."""
  del sigma
  flat = x0 - xt / jnp.maximum(alpha, 1e-8)
  norm = jnp.linalg.norm(
      flat.reshape(flat.shape[0], -1), axis=-1, keepdims=True,
  ) + 1e-8
  return 1.0 / jnp.maximum(norm, 1e-8)


# Signature of a step-size prefactor callable.
PrefactorFn = Callable[..., jax.Array]


################################################################################
# MARK: Kalman correction (Pi-GDM family)
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class KalmanCorrectionFn(CorrectionFn):
  """Closed-form Kalman update on ``x_0``.

      x_0_new = x_0 + Sigma A^T (A Sigma A^T + sigma_y^2 I)^{-1} (y - A x_0)

  over any linear :class:`ForwardFn` ``A`` and any
  :class:`PosteriorCovarianceFn` ``Sigma``.  Choice of covariance picks
  the Pi-GDM variant:

  - :class:`IsotropicPosteriorCovarianceFn`: DPS-style projection.
  - :class:`FixedPriorPosteriorCovarianceFn`: "cov-aware" (known prior).
  - :class:`TweediePosteriorCovarianceFn`: exact under any prior via
    Miyasawa JVP through the denoiser.

  The ``(A Sigma A^T + sigma_y^2 I) z = r`` solve is batched CG --
  posterior covariance only needs a matvec; the Jacobian is never
  materialised.  The adjoint of ``forward_fn`` comes from ``jax.vjp``.
  """

  observation: jax.Array
  forward_fn: ForwardFn
  posterior_covariance_fn: PosteriorCovarianceFn
  observation_noise: float = 0.1
  cg_max_iter: int = 20
  cg_tol: float = 1e-6

  def __call__(
      self,
      x0: jax.Array,
      xt: jax.Array,
      time: jax.Array,
      *,
      denoiser_fn: DenoiserFn,
      schedule: Any,
  ) -> jax.Array:
    adjoint = linear_adjoint(self.forward_fn, x0)
    sigma_y2 = max(float(self.observation_noise) ** 2, 1e-30)

    # Build the Cov matvec ONCE per Kalman call; reuse across CG iterations
    # and the final ``x0 + apply_cov(w)`` step.  For LowRankTweedie this
    # hoists the randomized-SVD sketch out of the inner loop.
    cov_matvec = self.posterior_covariance_fn(
        xt=xt, time=time, schedule=schedule, denoiser_fn=denoiser_fn,
    )

    def apply_cov(w):
      return cov_matvec(adjoint(w))

    def matvec(w):  # (A Sigma A^T + sigma_y^2 I) w
      return self.forward_fn.forward(apply_cov(w)) + sigma_y2 * w

    residual = self.observation - self.forward_fn.forward(x0)
    w = batched_cg(
        matvec, residual, max_iter=self.cg_max_iter, tol=self.cg_tol,
    )
    return x0 + apply_cov(w)


################################################################################
# MARK: Gradient correction (DPS family)
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class GradientCorrectionFn(CorrectionFn):
  """First-order shift from a differentiable :class:`TwistFn`.

      x_0_new = x_0 + strength * prefactor * grad_{xt} log psi(y | xt)

  Implements DPS-style guidance generically: any differentiable twist
  plugs in.  Step-size prefactor is an injected callable
  (:func:`miyasawa_prefactor` exact on Gaussian priors;
  :func:`dps_prefactor` canonical DPS ``1/||residual||``).
  """

  twist: TwistFn
  strength: float = 1.0
  prefactor_fn: PrefactorFn = miyasawa_prefactor

  def __call__(
      self,
      x0: jax.Array,
      xt: jax.Array,
      time: jax.Array,
      *,
      denoiser_fn: DenoiserFn,
      schedule: Any,
  ) -> jax.Array:
    alpha, sigma = scalar_alpha_sigma(schedule, time)

    def scalar_twist(xt_inner: jax.Array) -> jax.Array:
      return jnp.sum(self.twist(xt_inner, time, denoiser_fn=denoiser_fn))

    grad_xt = jax.grad(scalar_twist)(xt)
    prefactor = self.prefactor_fn(alpha=alpha, sigma=sigma, xt=xt, x0=x0)
    return x0 + float(self.strength) * prefactor * grad_xt


################################################################################
# MARK: Iterated correction (meta)
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class IteratedCorrectionFn(CorrectionFn):
  """Apply ``base`` ``num_iters`` times with denoiser re-evaluation.

  At each inner iteration:

    1. Apply ``base`` to the current ``x_0`` -> updated ``x_0_new``.
    2. Shift ``xt`` by ``alpha_t * (x_0_new - x_0_old)`` so the noise
       realisation is preserved.
    3. Re-evaluate the denoiser at the shifted xt -> new ``x_0``.
    4. Repeat for ``num_iters`` total iterations; the final call to
       ``base`` produces the returned ``x_0_new`` without a further
       shift (the outer sampler advances from the actual xt).

  For a Pi-GDM-family ``base`` on a mixture prior, the inner loop lets
  the denoiser's implicit mixture weights respond to the shifted xt,
  closing the intermediate-H bump on non-Gaussian posteriors.  K=3 is
  the empirical sweet spot.
  """

  base: CorrectionFn
  num_iters: int = 3

  def __call__(
      self,
      x0: jax.Array,
      xt: jax.Array,
      time: jax.Array,
      *,
      denoiser_fn: DenoiserFn,
      schedule: Any,
  ) -> jax.Array:
    alpha = scalar_alpha(schedule, time)
    xt_curr, x0_curr = xt, x0
    for _ in range(int(self.num_iters) - 1):
      x0_new = self.base(
          x0_curr, xt_curr, time,
          denoiser_fn=denoiser_fn, schedule=schedule,
      )
      xt_curr = xt_curr + alpha * (x0_new - x0_curr)
      x0_curr = denoiser_fn(xt_curr)
    return self.base(
        x0_curr, xt_curr, time,
        denoiser_fn=denoiser_fn, schedule=schedule,
    )
