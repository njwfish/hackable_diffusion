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

"""Refinement-theory diagnostics for the posterior-bridge framework.

Closed-form quantities from manuscript Appendix "Canonical Contractive
Bridges" (Section ``canonical-bridges``) plus the iterative-refinement
recursion of Theorem ``iterative-refinement``.  None of these enter the
sampler hot path; they are *diagnostics* that quantify the contraction
and Bayes-risk bounds that the refinement theorem composes:

  - Bridge endpoint Lipschitz constants ``beta_{s, t}``: per Proposition
    ``gaussian-bridge-lipschitz`` and Proposition
    ``masked-bridge-compression``.
  - Bayes-risk upper bounds ``B_t(ell)`` for canonical losses: per
    Proposition ``gaussian-bayes-risk`` (squared endpoint MSE) and
    Proposition ``masked-bayes-risk`` (normalised sequence log loss).
  - Iterative-refinement recursion ``D_{k-1} = beta_k delta_k + kappa_k
    D_k`` and endpoint-certificate ``E_k = delta_k + L_k D_k`` from
    Theorem ``iterative-refinement``.

Use these to:

  * Plot the endpoint-error compression coefficient as a function of
    ``(s, t)`` for a given schedule (recovering Figure
    ``fig:gaussian-endpoint-compression`` in the manuscript).
  * Bound the local-prediction Bayes risk before training a model so
    the loss-floor is known.
  * Plug measured ``delta_k`` (per-step posterior error from a trained
    model) into the recursion and predict the endpoint certificate
    ``E_k``.

Modality.  ``vp_gaussian_*`` functions assume a Gaussian (variance-
preserving) bridge with an ``alpha`` schedule satisfying ``1 = alpha(0)
> alpha(t) > 0``.  ``masked_*`` functions assume an absorbing-mask
discrete bridge with an ``alpha`` schedule on the same convention
(``alpha(t)`` = expected fraction of unmasked tokens at time ``t``).
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp


# Floor for ``alpha`` denominators.  Schedules with ``alpha(0) = 1`` and
# ``alpha(1) = 0`` go to zero at the boundary; clipping prevents
# ``1 / 0`` and ``log 0`` blowups in the analytic formulas.  The
# diagnostics are not meaningful in the boundary limit anyway.
_ALPHA_FLOOR = 1e-12
_ONE_MINUS_ALPHA_FLOOR = 1e-12


def _alpha_at(schedule, t) -> jax.Array:
  """Call ``schedule.alpha`` with a 1-D wrapper, return a 0-d array.

  ``Schedule.alpha`` is type-checked to require a leading batch axis;
  the diagnostics here are scalar in time, so we wrap to ``[1]`` and
  squeeze the output.
  """
  t1 = jnp.atleast_1d(jnp.asarray(t))
  return schedule.alpha(t1).reshape(())


################################################################################
# MARK: Gaussian (variance-preserving) bridge
################################################################################


def vp_gaussian_endpoint_lipschitz(
    schedule, s: jax.Array, t: jax.Array,
) -> jax.Array:
  """Endpoint-coordinate Lipschitz constant of the VP Gaussian bridge.

  Manuscript Proposition ``gaussian-bridge-lipschitz``,
  eq:vp-endpoint-attenuation:

      A_{s, t} = (alpha_s^2 - alpha_t^2) / [alpha_s (1 - alpha_t^2)],
      beta_{s, t} = ||A_{s, t}||_op.

  For 0 < t < s < 1 with the variance-preserving conditional bridge
  ``X_t = alpha_t X_0 + sqrt(1 - alpha_t^2) Z_t``, this is the scalar
  endpoint coefficient at fixed current ``x_t``.  Lies in ``[0, 1)``;
  vanishes as ``s -> t`` (no time elapsed, no contraction); approaches
  ``1`` as ``s -> 0`` (bridge step almost reveals the endpoint).

  Args:
    schedule: GaussianSchedule exposing ``schedule.alpha(time)``.
    s: target (cleaner) time.  Scalar or 0-d array.
    t: source (noisier) time, with ``s < t``.

  Returns:
    A scalar ``jax.Array`` -- the endpoint Lipschitz constant
    ``beta_{s, t}``.
  """
  alpha_s = _alpha_at(schedule, s)
  alpha_t = _alpha_at(schedule, t)
  num = alpha_s ** 2 - alpha_t ** 2
  denom = alpha_s * jnp.maximum(1.0 - alpha_t ** 2, _ONE_MINUS_ALPHA_FLOOR)
  # Clip to [0, 1] to suppress any small negative drift from fp arithmetic
  # near the boundary; mathematically the ratio lives in [0, 1).
  return jnp.clip(num / denom, 0.0, 1.0)


def gaussian_endpoint_mse_bayes_risk_bound(
    schedule, t: jax.Array, dim: int,
    norm_sq_z: jax.Array | float = 1.0,
) -> jax.Array:
  """Squared-error Bayes risk bound for the Gaussian bridge.

  Manuscript Proposition ``gaussian-bayes-risk``, eq:gaussian-bayes-risk:
  for ``X_t = alpha_t X_0 + sigma_t Z`` with ``Z`` independent of
  ``X_0`` and squared loss ``||x_hat - x_0||^2``,

      B_t(loss) <= (sigma_t^2 / alpha_t^2) * E ||Z||^2.

  ``E ||Z||^2 = dim`` for ``Z ~ N(0, I_d)`` -- the default.  Returns the
  numeric value of the bound.  This is an *upper bound* on the
  intrinsic Bayes risk via the candidate estimator ``a_t(x) = x /
  alpha_t``; the true Bayes risk (posterior-mean estimator) can be
  smaller for prior-aware estimators.

  Args:
    schedule: GaussianSchedule with ``alpha(time)`` and either
      ``sigma(time)`` or such that ``sigma_t^2 = 1 - alpha_t^2`` (the
      VP convention used here -- we compute it directly).
    t: time scalar / 0-d array.
    dim: data dimension ``d`` (used as ``E ||Z||^2`` for ``Z ~ N(0,
      I_d)``).
    norm_sq_z: optionally override the noise norm if the source is not
      isotropic standard normal.  Default ``1.0`` (matching unit
      variance).  Pass ``E ||Z||^2`` directly for a non-standard prior.

  Returns:
    Scalar bound ``(sigma_t^2 / alpha_t^2) * dim * norm_sq_z``.
  """
  alpha_t = _alpha_at(schedule, t)
  alpha_t_sq = jnp.maximum(alpha_t ** 2, _ALPHA_FLOOR)
  sigma_t_sq = jnp.maximum(1.0 - alpha_t ** 2, 0.0)
  return (sigma_t_sq / alpha_t_sq) * float(dim) * float(norm_sq_z)


################################################################################
# MARK: Absorbing-mask discrete bridge
################################################################################


def masked_endpoint_lipschitz(
    schedule, s: jax.Array, t: jax.Array,
) -> jax.Array:
  """Reveal probability ``r_{s, t}`` of the masked bridge.

  Manuscript Proposition ``masked-bridge-compression``,
  eq:masked-reveal-probability:

      r_{s, t} = (alpha_s - alpha_t) / (1 - alpha_t),
      beta_{s, t} = r_{s, t}.

  For an absorbing-mask discrete bridge with ``X_t^i = X_0^i`` w.p.
  ``alpha_t``, ``= mask`` w.p. ``1 - alpha_t``, the bridge from ``t``
  back to ``s < t`` reveals each masked coordinate to its endpoint
  value with probability ``r_{s, t}``.  This is also the endpoint
  Hamming-Lipschitz constant ``beta_{s, t}``.

  Args:
    schedule: Schedule exposing ``schedule.alpha(time)``.
    s: target (cleaner) time.
    t: source (more masked) time.

  Returns:
    Scalar reveal probability ``r_{s, t}``.
  """
  alpha_s = _alpha_at(schedule, s)
  alpha_t = _alpha_at(schedule, t)
  num = alpha_s - alpha_t
  denom = jnp.maximum(1.0 - alpha_t, _ONE_MINUS_ALPHA_FLOOR)
  return jnp.clip(num / denom, 0.0, 1.0)


def masked_endpoint_entropy_bayes_risk_bound(
    schedule, t: jax.Array, vocab_size: int,
) -> jax.Array:
  """Normalised cross-entropy Bayes risk bound for the masked bridge.

  Manuscript Proposition ``masked-bayes-risk``, eq:masked-bayes-risk:
  for the per-coordinate normalised log loss
  ``ell(p_hat, x_0) = -(1/n) log p_hat(x_0)``,

      B_t(ell) = (1/n) H(X_0 | X_t) <= (1 - alpha_t) * log |V|.

  ``1 - alpha_t`` is the expected fraction of masked tokens at time
  ``t`` (each contributes at most ``log |V|`` to the conditional
  entropy); unmasked tokens contribute zero because they pin ``X_0^i``.

  Args:
    schedule: Schedule exposing ``schedule.alpha(time)``.
    t: time scalar / 0-d array.
    vocab_size: ``|V|``.

  Returns:
    Scalar bound ``(1 - alpha_t) * log |V|``.
  """
  alpha_t = _alpha_at(schedule, t)
  return jnp.clip(1.0 - alpha_t, 0.0, 1.0) * jnp.log(
      jnp.asarray(float(vocab_size))
  )


################################################################################
# MARK: Iterative-refinement recursion (manuscript Theorem 5.7)
################################################################################


def simulate_refinement_recursion(
    *,
    deltas: Sequence[float] | jax.Array,
    betas: Sequence[float] | jax.Array,
    kappas: Sequence[float] | jax.Array,
    Ls: Sequence[float] | jax.Array,
    D_init: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
  """Unroll the refinement recursion of Theorem ``iterative-refinement``.

  At each grid step ``k = N, N-1, ..., 1`` we have:
    ``D_{k-1} <= beta_k delta_k + kappa_k D_k``           (eq:posterior-
                                                          error-bridge-
                                                          recursion)
    ``E_k    <= delta_k + L_k D_k``                       (eq:posterior-
                                                          error-endpoint-
                                                          certificate)

  ``deltas[k]`` is the local posterior error at level ``k`` evaluated
  under the *sampler*'s law, ``betas[k]`` the bridge contraction
  coefficient (e.g. the output of
  :func:`vp_gaussian_endpoint_lipschitz` or
  :func:`masked_endpoint_lipschitz`), ``kappas[k]`` the marginally-
  exact-bridge propagation coefficient, and ``Ls[k]`` the current-law
  prediction-stability constant.  ``D_init`` is the noisy-end mismatch
  ``D_N`` (typically zero or small if the chain is initialised exactly
  from ``p_N``).

  Indexing: pass arrays of length ``N`` containing entries for ``k = N,
  N-1, ..., 1`` in that order.  Returns ``D_seq`` of length ``N + 1``
  in the same order (``D_init`` first, then descending), and ``E_seq``
  of length ``N`` aligned with the input.

  Args:
    deltas: ``[N]`` sequence of local posterior errors.
    betas: ``[N]`` sequence of endpoint compression coefficients.
    kappas: ``[N]`` sequence of bridge-step propagation coefficients.
    Ls: ``[N]`` sequence of prediction-stability constants.
    D_init: ``D_N`` -- noisy-end mismatch (default 0).

  Returns:
    ``(D_seq, E_seq)`` where ``D_seq`` is ``[N + 1]`` (the recursion
    starts at ``D_init`` and unrolls down to ``D_0``) and ``E_seq`` is
    ``[N]`` (one certificate per step).  Both ordered descending in
    ``k`` to match the input.
  """
  deltas = jnp.asarray(deltas, dtype=jnp.float64)
  betas = jnp.asarray(betas, dtype=jnp.float64)
  kappas = jnp.asarray(kappas, dtype=jnp.float64)
  Ls = jnp.asarray(Ls, dtype=jnp.float64)
  if not (deltas.shape == betas.shape == kappas.shape == Ls.shape):
    raise ValueError(
        f"deltas / betas / kappas / Ls must share shape; got "
        f"{deltas.shape=}, {betas.shape=}, {kappas.shape=}, {Ls.shape=}."
    )
  if deltas.ndim != 1:
    raise ValueError(f"sequences must be 1-D; got ndim {deltas.ndim}.")

  def step(D_k, k):
    delta_k = deltas[k]
    beta_k = betas[k]
    kappa_k = kappas[k]
    L_k = Ls[k]
    E_k = delta_k + L_k * D_k
    D_km1 = beta_k * delta_k + kappa_k * D_k
    return D_km1, (D_km1, E_k)

  N = deltas.shape[0]
  _, (D_after, E_seq) = jax.lax.scan(
      step, jnp.asarray(float(D_init), dtype=jnp.float64),
      jnp.arange(N),
  )
  D_seq = jnp.concatenate([
      jnp.asarray([float(D_init)], dtype=jnp.float64),
      D_after,
  ])
  return D_seq, E_seq
