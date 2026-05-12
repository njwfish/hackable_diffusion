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

Four primitives:

- :class:`KalmanCorrectionFn`: closed-form Kalman update
  ``x_0 + Cov A^T (A Cov A^T + sigma_y^2 I)^{-1} (y - A x_0)``.  One
  class with three solver paths -- direct pseudo-inverse for hard
  observations / low-dim ``y``, CG and MINRES for high-dim ``y``.
- :class:`GradientCorrectionFn`: Tweedie-scaled gradient correction
  ``x_0 + strength * (sigma^2 / alpha) * grad_{xt} log psi``.  For
  *non-Gaussian* twists (classifier, energy); for Gaussian-likelihood +
  linear forward, ``KalmanCorrectionFn`` is the closed-form route.
- :class:`IteratedCorrectionFn`: re-evaluate the denoiser between
  corrections.  Wraps any base.
- :class:`CategoricalProjectionCorrectionFn`: per-site Bayes update of
  a soft-categorical ``x_0`` prediction (discrete / simplicial state).

All four satisfy :class:`CorrectionFn`: they take ``(x0, xt, time)``
plus a :class:`DenoiserFn` + schedule, and return the new ``x0``.

Modality compatibility
----------------------
- :class:`KalmanCorrectionFn`, :class:`GradientCorrectionFn`:
  Euclidean-x0 only (Gaussian ODE / SDE / distributional).
- :class:`CategoricalProjectionCorrectionFn`: simplicial / discrete.
- :class:`IteratedCorrectionFn`: inherits from its ``base``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import jax
import jax.numpy as jnp

from hackable_diffusion.lib.guidance.gaussian_conditioning import (
    _broadcast_observation,
    _flatten_observation,
    _materialize_observation_covariance,
    psd_pinv_solve,
)
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
)
from hackable_diffusion.lib.guidance.utils import (
    scalar_alpha,
    scalar_alpha_sigma,
)


################################################################################
# MARK: Kalman correction (Pi-GDM family) -- unified pinv / CG / MINRES
################################################################################


_ITERATIVE_SOLVERS = {"cg": batched_cg, "minres": batched_minres}
_VALID_SOLVERS = ("pinv",) + tuple(_ITERATIVE_SOLVERS)


def _default_iterative_solver_for(cov_fn: PosteriorCovarianceFn) -> str:
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

      x_0_new = x_0 + Cov A^T (A Cov A^T + sigma_y^2 I)^{-1} (y - A x_0)

  over any linear :class:`ForwardFn` ``A`` and any
  :class:`PosteriorCovarianceFn` ``Cov``.  Three solver paths:

  - ``solver="pinv"`` (default): materialise ``A Cov A^T`` as an
    ``M x M`` matrix (``M`` = observation dim) and solve via a
    symmetric eigendecomposition pseudo-inverse.  Handles
    ``observation_noise = 0`` correctly (rank-deficient hard
    observations), but ``O(M^3)`` per step -- right for low-dim ``y``
    (anti-diagonal constraints, scalar reward observations).

  - ``solver="cg"``: conjugate gradient on the matvec
    ``w -> (A Cov A^T + sigma_y^2 I) w``.  Never materialises the
    matrix; scales to high-dim ``y`` (image inpainting, super-
    resolution).  Requires the operator to be symmetric positive
    *definite* -- pair with PSD covariances
    (:class:`IsotropicPosteriorCovarianceFn`,
    :class:`FixedPriorPosteriorCovarianceFn`,
    :class:`PCAPosteriorCovarianceFn`,
    :class:`LowRankTweediePosteriorCovarianceFn` with
    ``project_psd=True``).

  - ``solver="minres"``: minimum-residual.  Same scaling as CG but
    handles symmetric *indefinite* operators -- the case for full
    :class:`TweediePosteriorCovarianceFn` on a non-Bayes-optimal
    denoiser or low-rank Tweedie with ``project_psd=False``.

  Picking the covariance picks the Pi-GDM variant: Isotropic gives the
  DPS-style projection, FixedPrior gives the "cov-aware" update with a
  known prior covariance, Tweedie gives the Miyasawa-exact update via
  JVP through the denoiser.

  CG and MINRES need ``observation_noise > 0`` to keep the Kalman
  matrix full-rank; ``"pinv"`` is the right solver for hard
  observations.  ``solver=None`` is *not* supported -- the user should
  explicitly choose; see ``solver`` argument.
  """

  observation: jax.Array
  forward_fn: ForwardFn
  posterior_covariance_fn: PosteriorCovarianceFn
  observation_noise: float = 0.0
  solver: str = "pinv"
  # Pinv knobs (used when ``solver == "pinv"``).
  pinv_rtol: float = 1e-5
  pinv_atol: float = 1e-8
  # Iterative-solver knobs (used when ``solver in {"cg", "minres"}``).
  cg_max_iter: int = 20
  cg_tol: float = 1e-6

  def __post_init__(self) -> None:
    if self.solver not in _VALID_SOLVERS:
      raise ValueError(
          f"KalmanCorrectionFn.solver must be one of {_VALID_SOLVERS}; "
          f"got {self.solver!r}.",
      )

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
    cov_matvec = self.posterior_covariance_fn(
        xt=xt, time=time, schedule=schedule, denoiser_fn=denoiser_fn,
    )

    if self.solver == "pinv":
      predicted = self.forward_fn.forward(x0)
      target = _broadcast_observation(self.observation, predicted)
      residual = _flatten_observation(target - predicted)
      system, adjoint, observation_shape = _materialize_observation_covariance(
          forward_fn=self.forward_fn,
          cov_matvec=cov_matvec,
          x0=x0,
      )
      sigma_y2 = float(self.observation_noise) ** 2
      if sigma_y2 > 0.0:
        eye = jnp.eye(system.shape[-1], dtype=system.dtype)
        system = system + sigma_y2 * eye[None, :, :]
      weights = psd_pinv_solve(
          system, residual,
          rtol=float(self.pinv_rtol), atol=float(self.pinv_atol),
      )
      weights_obs = weights.reshape(
          (weights.shape[0],) + observation_shape,
      )
      return x0 + cov_matvec(adjoint(weights_obs))

    # Iterative path (CG / MINRES) -- never materialises ``A Cov A^T``.
    adjoint = linear_adjoint(self.forward_fn, x0)
    sigma_y2 = max(float(self.observation_noise) ** 2, 1e-30)

    def apply_cov(w):
      return cov_matvec(adjoint(w))

    def matvec(w):
      return self.forward_fn.forward(apply_cov(w)) + sigma_y2 * w

    residual = self.observation - self.forward_fn.forward(x0)
    w = _ITERATIVE_SOLVERS[self.solver](
        matvec, residual,
        max_iter=int(self.cg_max_iter), tol=float(self.cg_tol),
    )
    return x0 + apply_cov(w)


################################################################################
# MARK: Gradient correction (Tweedie-scaled DPS-style)
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class GradientCorrectionFn(CorrectionFn):
  """Tweedie-scaled first-order shift from a differentiable :class:`TwistFn`.

      x_0_new = x_0 + strength * (sigma^2 / alpha) * grad_{xt} log psi(xt)

  ``sigma^2 / alpha`` is the Tweedie-identity scalar (Miyasawa);
  exact in the single-Gaussian-prior limit via
  ``Cov(x_0 | x_t) = (sigma^2 / alpha) d xhat_0 / d xt``.

  Intended for *non-Gaussian* twists -- :class:`ClassifierTwistFn`,
  :class:`EnergyTwistFn`, custom twists -- where ``log psi`` has no
  closed-form gradient with respect to ``x_0``.

  For Gaussian-likelihood + linear forward (e.g.
  :class:`GaussianLikelihoodTwistFn`,
  :class:`PosteriorPredictiveGaussianTwistFn`,
  :class:`NormResidualTwistFn`) :class:`KalmanCorrectionFn` is the
  closed-form route -- equivalent Cov-weighted update without going
  through the chain-rule split, which has a removable ``inf * 0``
  singularity at ``alpha -> 0`` and is FP-unstable in practice.
  """

  twist: 'TwistFn'                          # noqa: F821 - forward ref
  strength: float = 1.0

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
    alpha, sigma = scalar_alpha_sigma(schedule, time)

    def scalar_twist(xt_inner: jax.Array) -> jax.Array:
      return jnp.sum(self.twist(xt_inner, time, denoiser_fn=denoiser_fn))

    grad_xt = jax.grad(scalar_twist)(xt)
    prefactor = (sigma ** 2) / jnp.maximum(alpha, 1e-8)
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
from typing import Callable
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
      cloud_fn: PosteriorCloudFn | None = None,
      rng: jax.Array | None = None,
  ) -> jax.Array:
    del xt, denoiser_fn, schedule
    log_prior = jnp.log(jnp.clip(x0, self.eps, 1.0))
    log_lik = self.log_likelihood_fn(x0, time)
    log_post = log_prior + log_lik
    return jax.nn.softmax(log_post, axis=-1)


################################################################################
# MARK: Projection cloud correction (posterior-sample SMC)
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
