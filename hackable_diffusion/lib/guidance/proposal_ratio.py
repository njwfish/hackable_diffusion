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

"""Closed-form ``log p_theta(x_t | x_{t+1}) - log q(x_t | x_{t+1})`` per step.

When a :class:`CorrectionFn` shifts the denoiser output, SMC importance
weights require this ratio to stay unbiased.  For common sampler steps
the ratio admits a closed form (no Monte Carlo needed).

Every registered implementation has the uniform signature

    ``(stepper, corruption_process, outputs_uncorrected, outputs_corrected,
       xt_prev, xt_next, time_prev, time_next) -> (K,) log-ratio``

so :func:`proposal_log_ratio` can dispatch by ``isinstance`` without
signature adapters.  Add a new stepper by registering its ratio via
:func:`register_proposal_ratio`.

Built-in implementations:

- :func:`ddim_proposal_log_ratio` for Gaussian :class:`DDIMStep`
  (linear-mean-shift formula; returns 0 at ``eta=0``).
- :func:`simplicial_ddim_proposal_log_ratio` for
  :class:`SimplicialDDIMStep` with ``churn=0`` (categorical log-prob
  ratio on the sampled token).
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from hackable_diffusion.lib.sampling.gaussian_step_sampler import (
    DDIMStep,
    SdeStep,
)
from hackable_diffusion.lib.sampling.simplicial_step_sampler import (
    SimplicialDDIMStep,
)
from hackable_diffusion.lib.guidance.utils import scalar_alpha_sigma


################################################################################
# MARK: Gaussian DDIM
################################################################################


def ddim_proposal_log_ratio(
    *,
    stepper: DDIMStep,
    corruption_process: Any,
    outputs_uncorrected: dict[str, jax.Array],
    outputs_corrected: dict[str, jax.Array],
    xt_prev: jax.Array,
    xt_next: jax.Array,
    time_prev: jax.Array,
    time_next: jax.Array,
) -> jax.Array:
  """Exact ``log p_theta - log q`` for a Gaussian DDIM step.

  The DDIM step has linear-in-xhat_0 mean

      mu(xhat_0) = B xhat_0 + C x_r,
      B = alpha_s - alpha_r sigma_s sqrt(1-eta^2) / sigma_r,
      C = sigma_s sqrt(1-eta^2) / sigma_r,

  with ``r = prev, s = next`` and step variance ``(sigma_s eta)^2``.  A
  correction that shifts ``xhat_0`` by ``Delta`` shifts the proposal mean
  by ``B Delta`` with variance unchanged, giving

      log p - log q = (||x - mu_q||^2 - ||x - mu_p||^2) / (2 sigma_step^2).

  At ``eta = 0`` the proposal is a Dirac and the ratio is ill-defined
  (importance sampling degenerates).  We return zero there -- consistent
  with standard TDS-literature practice for deterministic proposals.
  """
  schedule = corruption_process.schedule
  alpha_r, sigma_r = scalar_alpha_sigma(schedule, time_prev)
  alpha_s, sigma_s = scalar_alpha_sigma(schedule, time_next)
  eta = float(getattr(stepper, "stoch_coeff", 0.0))

  if eta == 0.0:
    return jnp.zeros(xt_next.shape[0], dtype=xt_next.dtype)

  x0_uncorrected = corruption_process.convert_predictions(
      outputs_uncorrected, xt_prev, time_prev,
  )["x0"]
  x0_corrected = corruption_process.convert_predictions(
      outputs_corrected, xt_prev, time_prev,
  )["x0"]

  det_factor = jnp.sqrt(jnp.maximum(1.0 - eta ** 2, 0.0))
  coeff_b = alpha_s - alpha_r * sigma_s * det_factor / jnp.maximum(sigma_r, 1e-12)
  coeff_c = sigma_s * det_factor / jnp.maximum(sigma_r, 1e-12)
  sigma_step = sigma_s * eta

  mu_p = coeff_b * x0_uncorrected + coeff_c * xt_prev
  mu_q = coeff_b * x0_corrected + coeff_c * xt_prev

  sq_p = jnp.sum((xt_next - mu_p).reshape(xt_next.shape[0], -1) ** 2, axis=-1)
  sq_q = jnp.sum((xt_next - mu_q).reshape(xt_next.shape[0], -1) ** 2, axis=-1)
  return (sq_q - sq_p) / (2.0 * sigma_step ** 2)


################################################################################
# MARK: SDE / DDPM ancestral
################################################################################


def sde_proposal_log_ratio(
    *,
    stepper: SdeStep,
    corruption_process: Any,
    outputs_uncorrected: dict[str, jax.Array],
    outputs_corrected: dict[str, jax.Array],
    xt_prev: jax.Array,
    xt_next: jax.Array,
    time_prev: jax.Array,
    time_next: jax.Array,
) -> jax.Array:
  """Exact ``log p_theta - log q`` for a score-SDE Euler-Maruyama step.

  The SDE step has mean

      mu(xhat_0) = xt + dt * [-f(t) xt + 0.5 g(t)^2 (1 + churn^2) score(xhat_0)]

  where ``score = (alpha xhat_0 - xt) / sigma^2``.  The mean is linear in
  ``xhat_0`` with coefficient

      B = 0.5 * g^2 * (1 + churn^2) * dt * alpha / sigma^2,

  and the proposal has standard deviation ``sqrt(dt) g churn``.  A
  correction that shifts ``xhat_0`` by ``Delta`` shifts the mean by
  ``B Delta``; the Gaussian log-ratio is the usual

      log p - log q = (||x - mu_q||^2 - ||x - mu_p||^2) / (2 sigma_step^2).

  At ``churn = 0`` the proposal collapses to a deterministic Dirac and
  importance sampling degenerates; we return zero there.
  """
  churn = float(getattr(stepper, "churn", 0.0))
  if churn == 0.0:
    return jnp.zeros(xt_next.shape[0], dtype=xt_next.dtype)

  schedule = corruption_process.schedule
  t_prev = jnp.atleast_1d(time_prev).reshape(-1)[0:1]
  t_next = jnp.atleast_1d(time_next).reshape(-1)[0:1]
  alpha = schedule.alpha(t_prev).reshape(())
  sigma = schedule.sigma(t_prev).reshape(())
  f_t = schedule.f(t_prev).reshape(())
  g_t = schedule.g(t_prev).reshape(())
  dt = (t_prev - t_next).reshape(())

  x0_unc = corruption_process.convert_predictions(
      outputs_uncorrected, xt_prev, time_prev,
  )["x0"]
  x0_cor = corruption_process.convert_predictions(
      outputs_corrected, xt_prev, time_prev,
  )["x0"]

  # mu(xhat_0) = xt + dt * (-f xt + 0.5 g^2 (1+churn^2) * score(xhat_0))
  # with score = (alpha xhat_0 - xt) / sigma^2.  Factor common terms:
  base = xt_prev * (1.0 - dt * f_t) \
      - 0.5 * g_t ** 2 * (1.0 + churn ** 2) * dt * xt_prev / jnp.maximum(sigma ** 2, 1e-12)
  coeff_b = 0.5 * g_t ** 2 * (1.0 + churn ** 2) * dt * alpha / jnp.maximum(sigma ** 2, 1e-12)
  mu_p = base + coeff_b * x0_unc
  mu_q = base + coeff_b * x0_cor
  sigma_step = jnp.sqrt(dt) * g_t * churn

  sq_p = jnp.sum((xt_next - mu_p).reshape(xt_next.shape[0], -1) ** 2, axis=-1)
  sq_q = jnp.sum((xt_next - mu_q).reshape(xt_next.shape[0], -1) ** 2, axis=-1)
  return (sq_q - sq_p) / (2.0 * sigma_step ** 2)


################################################################################
# MARK: Simplicial DDIM
################################################################################


def simplicial_ddim_proposal_log_ratio(
    *,
    stepper: SimplicialDDIMStep,
    corruption_process: Any,
    outputs_uncorrected: dict[str, jax.Array],
    outputs_corrected: dict[str, jax.Array],
    xt_prev: jax.Array,
    xt_next: jax.Array,
    time_prev: jax.Array,
    time_next: jax.Array,
) -> jax.Array:
  """Exact ``log p_theta - log q`` for the churn=0 simplicial DDIM step.

  The simplicial DDIM step at ``churn = 0`` draws a token
  ``i ~ Categorical(softmax(logits))`` and beta weights ``(w, 1-w)`` from
  a schedule-dependent Beta.  The beta weights are data-independent and
  cancel between numerator and denominator; only the token draw depends
  on the correction.  Hence

      log p_theta - log q = log softmax(logits_unc)[i]
                          - log softmax(logits_cor)[i].

  The sampled token is recovered site-wise from
  ``argmax(xt_next - xt_prev)``: at the sampled token the position picks
  up an extra ``log(1-w)`` term, while everywhere else the difference is
  just ``log w``.
  """
  del time_next  # Simplicial ratio depends only on time_prev logits.
  churn = float(getattr(stepper, "churn", 0.0))
  if churn != 0.0:
    raise NotImplementedError(
        "simplicial_ddim_proposal_log_ratio only supports churn=0; for "
        "churn>0 add a bootstrap (correction-free) proposal or derive the "
        "Dirichlet-shrinkage step ratio and register it here."
    )
  logits_unc = corruption_process.convert_predictions(
      outputs_uncorrected, xt_prev, time_prev,
  )["logits"]
  logits_cor = corruption_process.convert_predictions(
      outputs_corrected, xt_prev, time_prev,
  )["logits"]
  log_p_unc = jax.nn.log_softmax(logits_unc, axis=-1)
  log_p_cor = jax.nn.log_softmax(logits_cor, axis=-1)
  sampled_idx = jnp.argmax(xt_next - xt_prev, axis=-1)
  one_hot = jax.nn.one_hot(sampled_idx, logits_unc.shape[-1], dtype=log_p_unc.dtype)
  per_site = jnp.sum((log_p_unc - log_p_cor) * one_hot, axis=-1)
  return jnp.sum(per_site, axis=-1)


################################################################################
# MARK: Registry and dispatcher
################################################################################


ProposalRatioFn = Callable[..., jax.Array]

_PROPOSAL_RATIO_REGISTRY: dict[type, ProposalRatioFn] = {
    DDIMStep: ddim_proposal_log_ratio,
    SdeStep: sde_proposal_log_ratio,
    SimplicialDDIMStep: simplicial_ddim_proposal_log_ratio,
}


def register_proposal_ratio(stepper_cls: type, fn: ProposalRatioFn) -> None:
  """Register a closed-form proposal ratio for a stepper class.

  ``fn`` must match the uniform signature

      ``(stepper, corruption_process, outputs_uncorrected, outputs_corrected,
         xt_prev, xt_next, time_prev, time_next) -> (K,)``.
  """
  _PROPOSAL_RATIO_REGISTRY[stepper_cls] = fn


def proposal_log_ratio(
    *,
    stepper: Any,
    corruption_process: Any,
    outputs_uncorrected: dict[str, jax.Array],
    outputs_corrected: dict[str, jax.Array],
    xt_prev: jax.Array,
    xt_next: jax.Array,
    time_prev: jax.Array,
    time_next: jax.Array,
    correction_identity: bool,
) -> jax.Array:
  """Closed-form ``log p_theta - log q`` for a single sampler step.

  Dispatches by ``isinstance`` against the registry.  Returns zero when
  the correction is an identity (uncorrected = corrected).  Raises
  ``NotImplementedError`` for unknown steppers -- register one via
  :func:`register_proposal_ratio`, or drop the correction (bootstrap SMC
  is always exact).
  """
  if correction_identity:
    return jnp.zeros(xt_next.shape[0], dtype=xt_next.dtype)

  for stepper_cls, fn in _PROPOSAL_RATIO_REGISTRY.items():
    if isinstance(stepper, stepper_cls):
      return fn(
          stepper=stepper, corruption_process=corruption_process,
          outputs_uncorrected=outputs_uncorrected,
          outputs_corrected=outputs_corrected,
          xt_prev=xt_prev, xt_next=xt_next,
          time_prev=time_prev, time_next=time_next,
      )
  raise NotImplementedError(
      f"proposal_log_ratio: no registered handler for stepper type "
      f"{type(stepper).__name__}.  Register one via "
      "``register_proposal_ratio(stepper_cls, fn)``, or run SMC with no "
      "correction (bootstrap, always exact)."
  )
