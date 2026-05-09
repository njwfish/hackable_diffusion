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
  posterior-sampler).  Kalman math assumes a standard inner product.
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

from hackable_diffusion.lib.guidance.linalg import (
    batched_cg,
    batched_minres,
    linear_adjoint,
)
from hackable_diffusion.lib.guidance.posterior_covariance import (
    LowRankTweediePosteriorCovarianceFn,
    TweediePosteriorCovarianceFn,
)
from hackable_diffusion.lib.guidance.protocols import (
    CorrectionFn,
    DenoiserFn,
    ForwardFn,
    PosteriorCloudFn,
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


SOLVERS: dict[str, Callable] = {
    "cg": batched_cg,
    "minres": batched_minres,
}


def _default_solver_for(cov_fn: PosteriorCovarianceFn) -> str:
  """Pick CG vs MINRES based on whether ``cov_fn`` may be indefinite.

  Full ``TweediePosteriorCovarianceFn`` only symmetrises -- the
  Jacobian of a non-Bayes-optimal denoiser still has negative
  eigenvalues, so the Kalman matrix is symmetric indefinite and CG
  cannot be relied on.  ``LowRankTweediePosteriorCovarianceFn`` with
  ``project_psd=False`` has the same issue.  All other built-in cov fns
  produce PSD ``Sigma`` and CG is cheaper.
  """
  if isinstance(cov_fn, TweediePosteriorCovarianceFn):
    return "minres"
  if (isinstance(cov_fn, LowRankTweediePosteriorCovarianceFn)
      and not cov_fn.project_psd):
    return "minres"
  return "cg"


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

  The ``(A Sigma A^T + sigma_y^2 I) z = r`` solve is iterative;
  posterior covariance only needs a matvec.  Two solvers ship in
  ``SOLVERS``:

  - ``'cg'``: conjugate gradient.  Cheaper per iteration but requires
    the Kalman matrix to be symmetric *positive-definite*.  Right
    choice when ``Sigma`` is PSD -- :class:`IsotropicPosteriorCovarianceFn`,
    :class:`FixedPriorPosteriorCovarianceFn`,
    :class:`PCAPosteriorCovarianceFn`,
    :class:`LowRankTweediePosteriorCovarianceFn` with ``project_psd=True``.
  - ``'minres'``: minimum-residual.  Handles symmetric *indefinite*
    operators correctly; pick this when ``Sigma`` is symmetric but may
    have negative eigenvalues -- :class:`TweediePosteriorCovarianceFn`
    on a non-Bayes-optimal denoiser.

  ``solver=None`` (default) auto-selects via :func:`_default_solver_for`:
  MINRES for full-Tweedie / non-PSD-projected low-rank Tweedie, CG
  otherwise.  Pass an explicit string to override.
  """

  observation: jax.Array
  forward_fn: ForwardFn
  posterior_covariance_fn: PosteriorCovarianceFn
  observation_noise: float = 0.1
  cg_max_iter: int = 20
  cg_tol: float = 1e-6
  solver: str | None = None

  def __call__(
      self,
      x0: jax.Array,
      xt: jax.Array,
      time: jax.Array,
      *,
      denoiser_fn: DenoiserFn,
      schedule: Any,
      cloud_fn: PosteriorCloudFn | None = None,
      rng: jax.Array | None = None,
  ) -> jax.Array:
    del cloud_fn, rng  # single-point correction; cloud / rng unused here.
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
    solver = (
        self.solver if self.solver is not None
        else _default_solver_for(self.posterior_covariance_fn)
    )
    if solver not in SOLVERS:
      raise ValueError(
          f"Unknown solver {solver!r}; choose from {sorted(SOLVERS)}."
      )
    w = SOLVERS[solver](
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
      cloud_fn: PosteriorCloudFn | None = None,
      rng: jax.Array | None = None,
  ) -> jax.Array:
    del cloud_fn, rng  # single-point correction; cloud / rng unused here.
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
      cloud_fn: PosteriorCloudFn | None = None,
      rng: jax.Array | None = None,
  ) -> jax.Array:
    del cloud_fn, rng  # single-point correction; cloud / rng unused here.
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


################################################################################
# MARK: Categorical projection correction (discrete / simplicial state)
################################################################################


# Per-position log-likelihood: ``(x0_soft, time) -> [B, *sites, K]``.
# Each entry is ``log L(x_0_i = k | y)`` -- the unnormalised log-likelihood
# of category ``k`` at site ``i`` under the observation ``y``.  Closed-form
# for site-factorised observations; the closure can also depend on the
# soft-prior ``x0_soft`` if the likelihood is non-factorised.
PerSiteLogLikelihoodFn = Callable[..., jax.Array]


@dataclasses.dataclass(kw_only=True, frozen=True)
class CategoricalProjectionCorrectionFn(CorrectionFn):
  """Per-site Bayes update of the soft-categorical ``x_0`` prediction.

  For categorical / simplex state the denoiser exposes ``x_0`` as the
  per-site soft probability ``p_theta(x_0_i = k | x_t)``.  When the
  observation ``y`` factorises across sites with per-site log-likelihood
  ``log L(x_0_i = k | y)``, the posterior is closed-form:

    ``log p(x_0_i = k | x_t, y)`` =
        ``log p_theta(x_0_i = k | x_t) + log L(x_0_i = k | y) + const``.

  Renormalising per site produces the corrected soft-categorical
  ``x_0`` that gets fed back into the simplex / discrete sampler.  This
  is the discrete analogue of the (Pi-GDM / DPS) Gaussian projection
  corrections above -- exact under per-site factorisation, no
  finite-step approximation needed.

  The correction is independent of ``xt`` and ``time``: the soft
  ``x_0`` already encodes the posterior over ``x_t``.  Composing this
  with :class:`IteratedCorrectionFn` re-evaluates the denoiser at a
  shifted ``xt``, which is useful when the per-site likelihood is only
  approximate.

  Attributes:
    log_likelihood_fn: ``(x0_soft, time) -> [B, *sites, K]`` returning
      the per-site, per-category log-likelihood.  Most observations
      depend only on the site index and the category, so the closure
      typically ignores ``x0_soft`` and ``time``; we expose them so
      that adaptive (denoiser-aware) likelihoods can also be used.
    eps: clamp on the prior probability before taking ``log`` to
      avoid ``-inf`` propagation.
  """

  log_likelihood_fn: PerSiteLogLikelihoodFn
  eps: float = 1.0e-12

  def __call__(
      self,
      x0: jax.Array,
      xt: jax.Array,
      time: jax.Array,
      *,
      denoiser_fn: DenoiserFn,
      schedule: Any,
      **kwargs,                                                  # cloud_fn / rng
  ) -> jax.Array:
    del xt, denoiser_fn, schedule, kwargs
    log_prior = jnp.log(jnp.clip(x0, self.eps, 1.0))
    log_lik = self.log_likelihood_fn(x0, time)
    log_post = log_prior + log_lik
    return jax.nn.softmax(log_post, axis=-1)


################################################################################
# MARK: Cloud-aware projection (posterior-bridges Algorithm 2)
################################################################################


# Signature of an x_0-space projection map.  Takes a single sample
# ``[B, *x0_shape]`` and returns the projected sample of the same shape.
ProjectionFn = Callable[[jax.Array], jax.Array]

# Signature of an optional importance weight ``w_{t, y}`` for the
# projected proposal.  Takes ``(\\tilde x_0, x_t, time)`` -- typically
# ``\\tilde x_0`` and ``x_t`` are ``[B, *data]`` and ``time`` is ``[B,
# ...]`` -- and returns log weights of shape ``[B,]``.
LogImportanceFn = Callable[
    [jax.Array, jax.Array, jax.Array], jax.Array,
]


@dataclasses.dataclass(kw_only=True, frozen=True)
class ProjectionCloudCorrectionFn(CorrectionFn):
  """Per-step projection guidance with optional importance weighting.

  Implements manuscript Algorithm 2 (Projected endpoint local move):

    1. Draw an ``R``-sample posterior cloud ``x_0^r ~ \\hat p_{0|t}(.
       | x_t)`` (provided by the sampler via ``cloud_fn``).
    2. Project each member onto the constraint set:
       ``\\tilde x_0^r = T_{t, y}(x_0^r)``.
    3. Compute optional log-importance weights
       ``log w_{t, y}(\\tilde x_0^r, x_t)`` to correct for the change
       of measure under the projection.  Without a weight, the cloud is
       weighted uniformly (the projection is treated as exact, as in
       the manuscript's single-Gaussian example).
    4. Categorical-sample one index ``I`` per particle from
       ``softmax(log w)``.
    5. Return the selected projected sample ``\\tilde x_0^I`` as the
       corrected ``x_0``.  The bridge step ``K_{s|0,t}(. |
       \\tilde x_0^I, x_t)`` runs in ``stepper.update``.

  When the projection is exact for the unconstrained posterior (e.g.
  the single-isotropic-Gaussian case in Proposition
  ``gaussian-affine-example``), no importance weight is needed and the
  uniform-weighted random pick is unbiased.  When the projection is a
  proposal that produces wrong responsibilities (Proposition
  ``gaussian-mixture-example``), supplying ``log_importance_fn`` =
  ``log A_{+/-}(x_t)`` removes the bias floor.

  Inputs / outputs:
    - The base x_0 from the single-denoiser path is *ignored* -- this
      correction overrides x_0 entirely with a projected cloud sample.
    - The sampler's stepper sees ``{"x0": \\tilde x_0^I}`` and advances
      via ``K_{s|0,t}``.

  Attributes:
    projection_fn: ``T_{t, y}(x_0) -> \\tilde x_0`` applied per cloud
      member.  Called as ``jax.vmap(projection_fn, in_axes=1,
      out_axes=1)`` so a function that handles one batched sample
      ``[B, *x0_shape] -> [B, *x0_shape]`` works unchanged.
    log_importance_fn: Optional ``log w_{t, y}(\\tilde x_0, xt, t) ->
      [B,]`` weight evaluator.  If ``None``, uniform weights are used
      (the manuscript's single-Gaussian regime).
  """

  projection_fn: ProjectionFn
  log_importance_fn: LogImportanceFn | None = None

  def __call__(
      self,
      x0: jax.Array,
      xt: jax.Array,
      time: jax.Array,
      *,
      denoiser_fn: DenoiserFn,
      schedule: Any,
      cloud_fn: PosteriorCloudFn | None = None,
      rng: jax.Array | None = None,
  ) -> jax.Array:
    del x0, denoiser_fn, schedule  # cloud-only correction
    if cloud_fn is None:
      raise ValueError(
          "ProjectionCloudCorrectionFn requires cloud_fn; set "
          "ConditionalDiffusionSampler.posterior_cloud_size > 0 so the "
          "sampler builds a posterior cloud at each step."
      )
    if rng is None:
      raise ValueError(
          "ProjectionCloudCorrectionFn requires rng for the categorical "
          "selection of one projected sample per particle."
      )
    cloud = cloud_fn(xt)                                       # [B, R, *data]
    projected = jax.vmap(
        self.projection_fn, in_axes=1, out_axes=1,
    )(cloud)                                                    # [B, R, *data]

    bsz, R = projected.shape[:2]
    if self.log_importance_fn is None:
      log_w = jnp.zeros((bsz, R), dtype=projected.dtype)
    else:
      def _per_sample_log_w(proj_r):
        return self.log_importance_fn(proj_r, xt, time)
      log_w = jax.vmap(_per_sample_log_w, in_axes=1, out_axes=1)(projected)
      if log_w.shape != (bsz, R):
        raise ValueError(
            f"log_importance_fn must produce shape [B, R]; got {log_w.shape}, "
            f"expected {(bsz, R)}."
        )

    # Per-particle categorical sample over the R proposals.
    indices = jax.random.categorical(rng, log_w, axis=-1)      # [B]
    return jnp.take_along_axis(
        projected,
        indices.reshape((bsz, 1) + (1,) * (projected.ndim - 2)),
        axis=1,
    ).squeeze(axis=1)                                           # [B, *data]
