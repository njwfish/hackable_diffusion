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

"""Spectral Riesz energy scores for Riemannian endpoint posteriors.

Implements the manuscript's Section "Riemannian Endpoint Scoring Rules"
(Proposition manifold-riesz-energy-loss).  The spectral Riesz distance
on a compact connected ``d``-dimensional Riemannian manifold is

    rho_beta(x, y) = sum_{j >= 1} lambda_j^{-(d + beta)/2}
                                  (phi_j(x) - phi_j(y))^2,

excluding the constant Laplace--Beltrami mode.  The generalized Riesz
energy score is

    ell_{beta, lam}(Q, y)
        = E_{U ~ Q} rho_beta(U, y)
        - (lam / 2) E_{U, U' ~ Q} rho_beta(U, U'),

strictly proper on ``P(manifold)`` for ``beta in (0, 2)`` and ``lam = 1``.
By the U-statistic identity (same argument as the Euclidean energy
score) the empirical loss with an ``M``-sample posterior cloud is

    ell_emp = (1/M) sum_m rho_beta(U^m, y)
              - (lam / (M (M - 1))) sum_{m != m'} rho_beta(U^m, U^m').

All training pieces -- including the population-level kernel
expectations -- only need a callable ``rho_beta(x, y)`` that returns a
scalar per leading-axis batch element.  The two canonical Riemannian
specializations from the manuscript are:

- **Torus** ``T^d``: Fourier modes, eigenvalues ``||n||^2``.  The
  closed-form distance ``rho_{beta, T^d}(theta, theta') = 2 sum_n
  ||n||^{-(d + beta)} (1 - cos(n . (theta - theta')))`` follows from the
  manuscript and is what :func:`make_torus_riesz_distance_fn` returns.

- **Sphere** ``S^m``: spherical harmonics, eigenvalues
  ``ell (ell + m - 1)``.  By the harmonic addition theorem,
  ``sum_r (Y_{ell, r}(x) - Y_{ell, r}(y))^2 = N(m, ell)
  (1 - C_ell^{(m-1)/2}(<x, y>) / C_ell^{(m-1)/2}(1))``, so the distance
  reduces to a sum of Gegenbauer polynomials evaluated at the inner
  product ``<x, y>``.  :func:`make_sphere_riesz_distance_fn` builds the
  truncated kernel without ever materialising individual ``Y_{ell, r}``.

Both factories return a ``RiemannianDistanceFn`` -- a pure callable
``(x: [..., ambient], y: [..., ambient]) -> [...]`` -- that
:func:`compute_riesz_energy_score_loss` averages over an ``M``-sample
posterior cloud.

Strict propriety (manuscript Proposition manifold-riesz-energy-loss).
For ``beta in (0, 2)`` and ``lam = 1``, the loss equals the squared
``ell^2``-distance between the spectral feature embeddings of ``P``
and ``Q`` and is zero iff ``P = Q``.  At ``lam = 0`` the interaction
term vanishes and the optimum collapses to the pointwise Bayes act
``argmin_a E_Y rho_beta(a, Y)`` (manuscript
eq:manifold-riesz-collapsed-bayes-act).
"""

import dataclasses
import math
from typing import Callable

import jax
import jax.numpy as jnp
import kauldron.ktyping as kt

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.training import base


################################################################################
# MARK: Type aliases
################################################################################

DataArray = hd_typing.DataArray
LossOutput = hd_typing.LossOutput
TargetInfo = hd_typing.TargetInfo
TimeArray = hd_typing.TimeArray

GaussianSchedule = schedules.GaussianSchedule


# A pairwise distance on a Riemannian manifold ``M``: takes two arrays
# with identical shape ``[..., ambient]`` (so leading axes broadcast)
# and returns ``[...]`` -- one scalar per leading-axis element.
RiemannianDistanceFn = Callable[[DataArray, DataArray], jax.Array]


################################################################################
# MARK: Generic Riesz / energy score U-statistic
################################################################################


@kt.typechecked
def compute_riesz_energy_score_loss(
    preds: TargetInfo,
    targets: TargetInfo,
    time: TimeArray,
    *,
    distance_fn: RiemannianDistanceFn,
    lam: float = 1.0,
    prediction_type: str = "x0",
    schedule: GaussianSchedule | None = None,
    weight_fn: base.WeightFn | None = None,
) -> jax.Array:
  """U-statistic energy score with a configurable manifold distance.

  Mirrors :func:`compute_energy_score_loss` but with a user-supplied
  ``distance_fn`` instead of the Euclidean ``||.||^beta``.  The strict-
  propriety regime is ``lam = 1`` together with a ``distance_fn`` that
  is conditionally negative definite (CND) on the manifold; spectral
  Riesz distances built by :func:`make_torus_riesz_distance_fn` and
  :func:`make_sphere_riesz_distance_fn` are CND by construction.

  Expected shapes:
    preds[prediction_type]:    [B, M, *data]  with M >= 2
    targets[prediction_type]:  [B, *data]

  Args:
    preds: Dict of predictions; ``preds[prediction_type]`` carries an
      ``M``-sample posterior cloud.
    targets: Dict of targets; ``targets[prediction_type]`` is the clean
      endpoint.
    time: Time array ``[B, ...]`` (only consumed by an optional
      ``weight_fn``).
    distance_fn: Pairwise Riemannian distance.  Must accept arrays of
      shape ``[..., ambient]`` and return ``[...]``.  See the module
      docstring for the contract.
    lam: Interaction-term weight.  ``lam = 1`` (default) is strictly
      proper for any CND manifold distance; ``lam = 0`` collapses to
      the pointwise loss.
    prediction_type: Key into ``preds`` and ``targets``.  Default
      ``'x0'`` per the energy-score convention.
    schedule: Optional schedule forwarded to ``weight_fn``.
    weight_fn: Optional per-time weight; multiplies the per-sample
      loss after the U-statistic.

  Returns:
    Per-sample loss of shape ``[B,]``.
  """
  cloud = preds[prediction_type]
  target = targets[prediction_type]
  if cloud.ndim < 2:
    raise ValueError(
        f"preds[{prediction_type!r}] must have a leading [batch, population] "
        f"pair, got shape {cloud.shape}."
    )
  bsz, pop = cloud.shape[0], cloud.shape[1]
  if pop < 2:
    raise ValueError(
        f"Riesz energy score requires population size M >= 2, got M={pop}."
    )
  if target.shape[0] != bsz:
    raise ValueError(
        f"Batch dim mismatch: preds has {bsz}, targets has {target.shape[0]}."
    )
  if cloud.shape[2:] != target.shape[1:]:
    raise ValueError(
        "preds[..., 2:] and targets[..., 1:] must share the data shape: "
        f"{cloud.shape=} vs {target.shape=}."
    )

  # Data term: distance from each cloud member to the target.
  # Broadcast target across the M axis.
  target_bm = jnp.broadcast_to(
      target[:, None, ...], (bsz, pop) + target.shape[1:],
  )                                                          # [B, M, *data]
  data_distances = distance_fn(cloud, target_bm)              # [B, M]
  if data_distances.shape != (bsz, pop):
    raise ValueError(
        "distance_fn must reduce the data-axis tail to a scalar per "
        f"leading-axis element; got {data_distances.shape}, expected "
        f"{(bsz, pop)}."
    )
  data_term = jnp.mean(data_distances, axis=1)                # [B]

  # Interaction term: pairwise distances within the cloud.
  cloud_left = cloud[:, :, None, ...]                         # [B, M, 1, *data]
  cloud_right = cloud[:, None, :, ...]                        # [B, 1, M, *data]
  cloud_left_b = jnp.broadcast_to(
      cloud_left, (bsz, pop, pop) + cloud.shape[2:],
  )
  cloud_right_b = jnp.broadcast_to(
      cloud_right, (bsz, pop, pop) + cloud.shape[2:],
  )
  pair_distances = distance_fn(cloud_left_b, cloud_right_b)   # [B, M, M]
  if pair_distances.shape != (bsz, pop, pop):
    raise ValueError(
        f"pair distance_fn output shape {pair_distances.shape} != "
        f"{(bsz, pop, pop)}."
    )
  # U-statistic denominator 2 * M * (M-1) (paper's per-(i, j) sum-over-j form).
  mask = 1.0 - jnp.eye(pop, dtype=pair_distances.dtype)
  interaction = jnp.sum(pair_distances * mask, axis=(1, 2)) / (
      2.0 * pop * (pop - 1)
  )                                                            # [B]

  per_sample = data_term - float(lam) * interaction            # [B]

  if weight_fn is not None:
    weight = weight_fn(
        schedule=schedule, preds=preds, targets=targets, time=time,
    )
    weight = jnp.asarray(weight).reshape(bsz, -1).mean(axis=-1)
    per_sample = per_sample * weight

  return per_sample


@dataclasses.dataclass(frozen=True, kw_only=True)
class RieszEnergyScoreLoss(base.DiffusionLoss):
  """Generic Riesz energy score with a user-supplied manifold distance.

  See :func:`compute_riesz_energy_score_loss` for the U-statistic
  construction and the strict-propriety regime.

  Attributes:
    distance_fn: ``RiemannianDistanceFn`` from one of the factories
      below or a user-defined CND distance.
    lam: Interaction-term weight.
    prediction_type: Key for the cloud in ``preds`` / ``targets``.
    schedule: Optional, forwarded to ``weight_fn``.
    weight_fn: Optional time weighting.
  """

  distance_fn: RiemannianDistanceFn
  lam: float = 1.0
  prediction_type: str = "x0"
  schedule: GaussianSchedule | None = None
  weight_fn: base.WeightFn | None = None

  @kt.typechecked
  def __call__(
      self,
      preds: TargetInfo,
      targets: TargetInfo,
      time: TimeArray,
  ) -> LossOutput:
    return compute_riesz_energy_score_loss(
        preds=preds,
        targets=targets,
        time=time,
        distance_fn=self.distance_fn,
        lam=self.lam,
        prediction_type=self.prediction_type,
        schedule=self.schedule,
        weight_fn=self.weight_fn,
    )


################################################################################
# MARK: Torus T^d -- closed-form Fourier truncation
################################################################################


def make_torus_modes(modes_per_dim: int, dim: int) -> jax.Array:
  """Symmetric grid of integer modes ``n in Z^d`` with ``|n_i| <= modes_per_dim``.

  Excludes the zero mode and applies a sign canonicalisation so each
  mode-pair ``{n, -n}`` contributes once: we keep ``n`` whose first
  nonzero coordinate is positive.  This matches the manuscript's
  per-mode contribution ``2 ||n||^{-(d + beta)} (1 - cos(n . (theta -
  theta')))``, which already double-counts the cosine term across the
  ``n``/``-n`` pair via the ``2`` prefactor.

  Wait -- inspection: the manuscript form

      rho_{beta, T^d}(theta, theta')
          = sum_{n in Z^d \\ 0} ||n||^{-(d + beta)} |e^{i n.theta}
                                                    - e^{i n.theta'}|^2
          = 2 sum_{n in Z^d \\ 0} ||n||^{-(d + beta)}
                                  (1 - cos(n . (theta - theta')))

  sums over the **full** non-zero lattice -- both ``n`` and ``-n``.  For
  efficiency we sum over a half-lattice and double the result; the sign
  canonicalisation here returns that half-lattice.  The factor of 2 is
  applied inside :func:`make_torus_riesz_distance_fn`.

  Args:
    modes_per_dim: ``M``.  Includes ``M`` modes per axis in each
      direction; total mode count is roughly ``(2M+1)^d / 2``.
    dim: ``d``, the torus dimension.

  Returns:
    Integer ``[N, d]`` array of distinct mode vectors with ``n != 0``
    and the sign convention above.
  """
  if dim < 1:
    raise ValueError(f"dim must be >= 1; got {dim}.")
  if modes_per_dim < 1:
    raise ValueError(f"modes_per_dim must be >= 1; got {modes_per_dim}.")
  axis = jnp.arange(-modes_per_dim, modes_per_dim + 1, dtype=jnp.int32)
  grids = jnp.meshgrid(*[axis] * dim, indexing="ij")
  modes = jnp.stack([g.reshape(-1) for g in grids], axis=-1)   # [(2M+1)^d, d]
  # Drop the zero mode and keep only those whose first nonzero coord is
  # positive (so we represent each {n, -n} pair once).
  nonzero = jnp.any(modes != 0, axis=-1)
  # First nonzero coordinate sign.
  is_first_nonzero = (
      jnp.cumsum(modes != 0, axis=-1) == 1
  ) & (modes != 0)
  first_nonzero_value = jnp.sum(
      jnp.where(is_first_nonzero, modes, 0), axis=-1,
  )
  keep = nonzero & (first_nonzero_value > 0)
  return modes[keep]


def make_torus_riesz_distance_fn(
    *,
    modes: jax.Array,
    dim: int,
    beta: float = 1.0,
    eps: float = 1e-12,
) -> RiemannianDistanceFn:
  """Build the truncated Fourier-Riesz distance on ``T^d``.

  Returns a callable that, for ``theta, theta' in [0, 2*pi)^d``,
  evaluates

      rho_beta(theta, theta') = 4 sum_n ||n||^{-(d + beta)}
                                       (1 - cos(n . (theta - theta'))),

  where the sum is over the half-lattice from :func:`make_torus_modes`
  (the factor of 4 is ``2 * 2``: one ``2`` from the manuscript's full-
  lattice prefactor and one from the half-lattice symmetry).

  Args:
    modes: ``[N, d]`` integer mode vectors, e.g. from
      :func:`make_torus_modes`.  Caller controls truncation; for
      training, ``modes_per_dim ~ 8-32`` is typical for low-dim tori.
    dim: ``d``, used in the spectral weight ``||n||^{-(d + beta)}``.
    beta: Energy-score exponent ``beta in (0, 2)``.  ``beta = 1`` is the
      direct analogue of the Euclidean energy score.
    eps: Floor under the spectral weight to avoid division by zero
      (cannot trigger in practice since modes are nonzero, but used as
      a safety guard).
  """
  if dim < 1:
    raise ValueError(f"dim must be >= 1; got {dim}.")
  if not 0.0 < beta < 2.0:
    raise ValueError(
        f"beta must be in (0, 2) for strict propriety; got beta={beta}."
    )
  modes_arr = jnp.asarray(modes, dtype=jnp.float64)
  if modes_arr.ndim != 2 or modes_arr.shape[-1] != dim:
    raise ValueError(
        f"modes must be [N, dim={dim}]; got {modes_arr.shape}."
    )
  norms_sq = jnp.sum(modes_arr ** 2, axis=-1)                  # [N]
  norms = jnp.sqrt(jnp.maximum(norms_sq, eps))
  weights = jnp.power(norms, -(dim + beta))                    # [N]

  def distance_fn(theta: jax.Array, theta_prime: jax.Array) -> jax.Array:
    if theta.shape != theta_prime.shape:
      raise ValueError(
          f"theta / theta_prime shape mismatch: {theta.shape=} vs "
          f"{theta_prime.shape=}."
      )
    if theta.shape[-1] != dim:
      raise ValueError(
          f"theta last axis must equal dim={dim}; got {theta.shape[-1]}."
      )
    diff = theta - theta_prime                                 # [..., d]
    inner = jnp.tensordot(diff, modes_arr, axes=[[-1], [-1]])  # [..., N]
    # 4 = 2 (manuscript) * 2 (half-lattice).
    return 4.0 * jnp.sum(weights * (1.0 - jnp.cos(inner)), axis=-1)

  return distance_fn


################################################################################
# MARK: Sphere S^m -- Gegenbauer kernel via the addition theorem
################################################################################


def _gegenbauer_polynomials(
    t: jax.Array, max_degree: int, alpha: float,
) -> jax.Array:
  """Evaluate Gegenbauer polynomials ``C_ell^alpha(t)`` for ``ell = 0..L``.

  Uses the standard three-term recurrence
  ``ell C_ell^alpha(t) = 2 t (ell + alpha - 1) C_{ell-1}^alpha(t)
                          - (ell + 2 alpha - 2) C_{ell-2}^alpha(t)``
  with seeds ``C_0^alpha = 1`` and ``C_1^alpha = 2 alpha t``.

  Returns a stacked array of shape ``t.shape + (L+1,)``.

  Notes for ``alpha = 0`` (i.e. ``m = 1``, the circle):  the standard
  Gegenbauer recurrence breaks down at ``alpha = 0``; for the circle
  use the torus path with ``dim = 1`` instead.
  """
  if max_degree < 0:
    raise ValueError(f"max_degree must be >= 0; got {max_degree}.")
  if alpha == 0.0:
    raise ValueError(
        "Gegenbauer recurrence requires alpha > 0.  For S^1 use "
        "make_torus_riesz_distance_fn with dim=1 instead."
    )
  c0 = jnp.ones_like(t)
  if max_degree == 0:
    return c0[..., None]
  c1 = 2.0 * float(alpha) * t
  if max_degree == 1:
    return jnp.stack([c0, c1], axis=-1)

  # Iterative recurrence; small max_degree so a Python for-loop is
  # cleaner than scan for Gegenbauer.
  cs = [c0, c1]
  for ell in range(2, max_degree + 1):
    c_ell = (
        2.0 * t * (ell + alpha - 1.0) * cs[-1]
        - (ell + 2.0 * alpha - 2.0) * cs[-2]
    ) / float(ell)
    cs.append(c_ell)
  return jnp.stack(cs, axis=-1)                                # [..., L+1]


def make_sphere_riesz_distance_fn(
    *,
    max_degree: int,
    ambient_dim: int,
    beta: float = 1.0,
    eps: float = 1e-12,
) -> RiemannianDistanceFn:
  """Spectral Riesz distance on ``S^m`` truncated at degree ``L``.

  Manuscript Section "Sphere" (eq:sphere-riesz-energy-distance):

      rho_{beta, S^m}(x, y)
          = sum_{ell=1}^{infty} {ell (ell + m - 1)}^{-(m + beta)/2}
              sum_{r=1}^{N(m, ell)} (Y_{ell, r}(x) - Y_{ell, r}(y))^2.

  Use the harmonic addition theorem
  ``sum_r Y_{ell, r}(x) Y_{ell, r}(y) = (N(m, ell) / |S^m|)
   C_ell^{(m-1)/2}(<x, y>) / C_ell^{(m-1)/2}(1)``
  with orthonormal harmonics, so

      sum_r (Y_{ell, r}(x) - Y_{ell, r}(y))^2
          = (2 N(m, ell) / |S^m|) (1 - C_ell^alpha(<x, y>) / C_ell^alpha(1))

  where ``alpha = (m - 1)/2`` and ``N(m, ell) = (2 ell + m - 1) /
  (m - 1) * binom(ell + m - 2, m - 2)`` is the dimension of the
  degree-``ell`` harmonic space.  Each truncation level ``L`` evaluates
  this kernel at the inner product ``<x, y>`` -- no individual
  ``Y_{ell, r}`` is materialised.

  We absorb the ``|S^m|`` denominator into a global scale (it is the
  same constant across degrees) and drop additive constants that
  cancel in the energy-score U-statistic; the strict-propriety regime
  is preserved up to a positive scaling.

  Args:
    max_degree: Spectral cutoff ``L``.  Sum runs ``ell = 1..L``.
      Larger ``L`` better resolves the posterior; cost is
      ``O(L * batch)`` per pairwise distance.
    ambient_dim: ``m + 1``.  The sphere ``S^m`` is embedded in
      ``R^{m+1}``; we index by the ambient dimension to avoid
      confusion.
    beta: Energy-score exponent ``beta in (0, 2)``.
    eps: Floor for the kernel normalisation at ``t = 1``.
  """
  if ambient_dim < 3:
    raise ValueError(
        "ambient_dim must be >= 3 (i.e. S^m with m >= 2).  For S^1 / "
        "circular endpoints use make_torus_riesz_distance_fn with "
        "dim=1; for S^0 the loss is degenerate."
    )
  m = ambient_dim - 1
  alpha = (m - 1.0) / 2.0
  if max_degree < 1:
    raise ValueError(f"max_degree must be >= 1; got {max_degree}.")
  if not 0.0 < beta < 2.0:
    raise ValueError(
        f"beta must be in (0, 2) for strict propriety; got beta={beta}."
    )

  # Per-degree dimensions N(m, ell) and spectral weights.
  ells = jnp.arange(1, max_degree + 1, dtype=jnp.float64)      # [L]
  # Eigenvalue lambda_ell = ell (ell + m - 1).
  eigvals = ells * (ells + m - 1.0)                            # [L]
  # Spectral weight lambda^{-(m + beta)/2}.
  spectral_weight = jnp.power(eigvals, -(m + beta) / 2.0)      # [L]
  # Harmonic-space dimension N(m, ell) = (2 ell + m - 1) / (m - 1) *
  # binom(ell + m - 2, m - 2).
  if m == 2:
    n_dim = 2.0 * ells + 1.0                                   # standard S^2: 2L+1
  else:
    # Vectorise the binomial via Stirling-stable lgamma.
    binom_log = (
        jax.scipy.special.gammaln(ells + m - 1.0)
        - jax.scipy.special.gammaln(ells + 1.0)
        - jax.scipy.special.gammaln(jnp.asarray(m - 1.0))
    )
    binom = jnp.exp(binom_log)                                 # binom(ell+m-2, m-2)
    n_dim = (2.0 * ells + m - 1.0) / (m - 1.0) * binom         # [L]

  # Kernel normalisation at t = 1: C_ell^alpha(1) = binom(ell + 2 alpha
  # - 1, ell) for alpha != 0.  For alpha = (m - 1)/2 this is just
  # binom(ell + m - 2, ell).
  c_at_one_log = (
      jax.scipy.special.gammaln(ells + 2.0 * alpha)
      - jax.scipy.special.gammaln(ells + 1.0)
      - jax.scipy.special.gammaln(jnp.asarray(2.0 * alpha))
  )
  c_at_one = jnp.exp(c_at_one_log)                             # [L]
  c_at_one_safe = jnp.maximum(c_at_one, eps)

  # Per-degree coefficient: 2 * N(m, ell) * weight / C_ell^alpha(1).
  # The 2 is from (sum (a-b)^2 = 2 K(1) - 2 K(t)) and we factor the K(t)
  # term out below.
  per_degree_coeff = 2.0 * n_dim * spectral_weight / c_at_one_safe   # [L]

  def distance_fn(x: jax.Array, y: jax.Array) -> jax.Array:
    if x.shape != y.shape:
      raise ValueError(
          f"x / y shape mismatch: {x.shape=} vs {y.shape=}."
      )
    if x.shape[-1] != ambient_dim:
      raise ValueError(
          f"x last axis must equal ambient_dim={ambient_dim}; got "
          f"{x.shape[-1]}."
      )
    inner = jnp.sum(x * y, axis=-1)                            # [...] in [-1, 1]
    # Clip slightly inside [-1, 1] to be defensive against fp drift.
    inner_clipped = jnp.clip(inner, -1.0 + eps, 1.0 - eps)
    # Gegenbauer polynomials C_ell^alpha(<x, y>) for ell = 0..L.
    poly = _gegenbauer_polynomials(inner_clipped, max_degree, alpha)
    # Use ell = 1..L only (drop the constant mode).
    poly_pos = poly[..., 1:]                                   # [..., L]
    # Per-degree contribution: per_degree_coeff_ell * (C_ell(1) - C_ell(t))
    # = per_degree_coeff_ell * C_ell(1) - per_degree_coeff_ell * C_ell(t).
    # We can absorb the per_degree_coeff * C_ell(1) into a global
    # constant, but it varies per ell so we sum it explicitly.
    # rho = sum_ell (2 N w / C(1)) * (C(1) - C(t))
    #     = sum_ell 2 N w * (1 - C(t) / C(1))
    contributions = per_degree_coeff * (
        c_at_one - poly_pos
    )                                                          # [..., L]
    return jnp.sum(contributions, axis=-1)                     # [...]

  return distance_fn
