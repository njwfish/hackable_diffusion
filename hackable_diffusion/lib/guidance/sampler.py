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

"""Conditional diffusion sampler composing CorrectionFn / TwistFn / ResamplerFn.

Wraps a :class:`DiffusionSampler` with three composable axes:

- ``correction_fn``: optional per-step :class:`CorrectionFn`
  (``x0 -> x0_new``).
- ``twist_fn``: optional :class:`TwistFn` (``log psi(y | xt)``) for
  SMC importance weights and DPS-style gradient guidance.
- ``resampler_fn``: optional :class:`ResamplerFn` (defaults to the
  identity :class:`NoResamplerFn`).

Per step the sampler builds a :class:`DenoiserFn` closure from the raw
``inference_fn`` + ``corruption_process`` + current ``(time,
conditioning, step_rng)``, calls it to get ``x0``, applies the
correction (if any), converts ``{"x0": x0_new}`` back to the stepper's
native prediction type, and advances.  Twists and corrections see only
``denoiser_fn``; they don't know about ``inference_fn``,
``corruption_process``, ``rng``, or ``conditioning``.

Weight formula: at each step we accumulate

    delta_log_w = log_proposal_ratio + (log psi_new - log psi_old)

The proposal ratio comes from ``stepper.kernel(...).log_density_ratio(...)``;
it is zero for an identity correction and for deterministic steppers.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import jax
import jax.numpy as jnp

from hackable_diffusion.lib.sampling.sampling import DiffusionSampler
from hackable_diffusion.lib.guidance.denoisers import make_denoiser_fn
from hackable_diffusion.lib.guidance.protocols import (
    CorrectionFn,
    DenoiserFn,
    ResamplerFn,
    TwistFn,
)
from hackable_diffusion.lib.guidance.proposal_ratio import proposal_log_ratio
from hackable_diffusion.lib.guidance.resamplers import NoResamplerFn
from hackable_diffusion.lib.guidance.utils import (
    accepts_rng_kwarg, call_inference_fn,
)


def _outputs_with_x0(
    outputs_native: dict[str, jax.Array], x0_new: jax.Array,
) -> dict[str, jax.Array]:
  """Write a corrected soft-x0 back into the stepper's native prediction type.

  Gaussian corruption accepts ``{"x0": ...}`` directly.  Simplicial
  corruption only accepts ``{"logits": ...}`` (its ``x0`` slot is
  the argmax, not usable as a soft correction target), so we supply
  ``logits = log(x0_new)`` -- softmax of that recovers the corrected
  simplex vector inside the stepper.
  """
  if "logits" in outputs_native:
    return {"logits": jnp.log(jnp.clip(x0_new, 1e-30, 1.0))}
  return {"x0": x0_new}


def _xt_time(step) -> tuple[jax.Array, jax.Array]:
  """Extract ``(xt, time)`` from a :class:`DiffusionStep`."""
  return step.xt, step.step_info.time


def _split_first_middle_last(tree):
  """Split a stacked step-info pytree into ``(first, middle, last)``."""
  return (
      jax.tree.map(lambda x: x[0], tree),
      jax.tree.map(lambda x: x[1:-1], tree),
      jax.tree.map(lambda x: x[-1], tree),
  )


def _gather_particle_leaves(carry, indices: jax.Array):
  """Gather every particle-indexed leaf by ``indices`` along axis 0.

  Leaves whose leading dim does not match ``indices.shape[0]`` are
  assumed shared across particles and pass through unchanged.
  """
  k = indices.shape[0]

  def gather_leaf(leaf):
    if not hasattr(leaf, "ndim") or leaf.ndim == 0:
      return leaf
    if leaf.shape[0] != k:
      return leaf
    return leaf[indices]

  return jax.tree.map(gather_leaf, carry)


def _resample_indices(
    resampler_fn: ResamplerFn,
    log_weights: jax.Array,
    *,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Return ``(gather_indices, new_log_weights)`` from a :class:`ResamplerFn`."""
  k = log_weights.shape[0]
  identity = jnp.arange(k)
  resampled_ids, new_log_weights = resampler_fn(
      identity[:, None].astype(log_weights.dtype),
      log_weights,
      rng=rng,
  )
  indices = resampled_ids[:, 0].astype(jnp.int32)
  return indices, new_log_weights


@dataclasses.dataclass(kw_only=True, frozen=True)
class ConditionalDiffusionSampler:
  """Conditional sampler composing CorrectionFn, TwistFn, and ResamplerFn.

  With no correction, no twist, and the identity resampler, delegates
  directly to ``base_sampler`` for bit-for-bit parity.

  Attributes:
    base_sampler: the underlying :class:`DiffusionSampler`.
    corruption_process: the corruption process; used to build per-step
      :class:`DenoiserFn` closures.
    correction_fn: optional ``x0 -> x0_new`` correction.
    twist_fn: optional ``log psi(y | xt)`` potential.
    resampler_fn: particle resampler (defaults to :class:`NoResamplerFn`).
    num_particles: SMC population size; ``1`` with no correction / twist
      / resampler short-circuits to the base sampler.
  """

  base_sampler: DiffusionSampler
  corruption_process: Any
  correction_fn: CorrectionFn | None = None
  twist_fn: TwistFn | None = None
  resampler_fn: ResamplerFn = dataclasses.field(default_factory=NoResamplerFn)
  num_particles: int = 1

  def __call__(
      self,
      *,
      inference_fn: Callable,
      rng: jax.Array,
      initial_noise: jax.Array,
      conditioning: Any = None,
  ):
    if (
        self.correction_fn is None
        and self.twist_fn is None
        and isinstance(self.resampler_fn, NoResamplerFn)
    ):
      return self.base_sampler(
          inference_fn=inference_fn,
          rng=rng,
          initial_noise=initial_noise,
          conditioning=conditioning,
      )
    return self._run_loop(
        inference_fn=inference_fn,
        rng=rng,
        initial_noise=initial_noise,
        conditioning=conditioning,
    )

  def _apply_correction(
      self,
      x0: jax.Array,
      xt: jax.Array,
      time: jax.Array,
      denoiser_fn: DenoiserFn,
      schedule: Any,
  ) -> jax.Array:
    """``x0_new = correction_fn(x0, xt, t, denoiser_fn, schedule)``, identity if None."""
    if self.correction_fn is None:
      return x0
    return self.correction_fn(
        x0, xt, time, denoiser_fn=denoiser_fn, schedule=schedule,
    )

  def _evaluate_twist(
      self,
      xt: jax.Array,
      time: jax.Array,
      denoiser_fn: DenoiserFn,
  ) -> jax.Array:
    """``twist_fn(xt, time, denoiser_fn=...)``, zero if None."""
    if self.twist_fn is None:
      return jnp.zeros(xt.shape[0], dtype=xt.dtype)
    return self.twist_fn(xt, time, denoiser_fn=denoiser_fn)

  def _run_loop(
      self,
      *,
      inference_fn: Callable,
      rng: jax.Array,
      initial_noise: jax.Array,
      conditioning: Any,
  ):
    corruption = self.corruption_process
    schedule = getattr(corruption, "schedule", None)
    stepper = self.base_sampler.stepper
    time_schedule = self.base_sampler.time_schedule
    num_steps = self.base_sampler.num_steps
    correction_identity = self.correction_fn is None
    uses_rng = accepts_rng_kwarg(inference_fn)

    all_infos = time_schedule.all_step_infos(rng, num_steps, initial_noise)
    first_info, next_infos, last_info = _split_first_middle_last(all_infos)
    first_step = stepper.initialize(initial_noise, first_info)

    # Initial twist evaluation at (xt_0, t_0).
    xt0, t0 = _xt_time(first_step)
    log_psi_prev = self._evaluate_twist(
        xt0, t0,
        denoiser_fn=make_denoiser_fn(
            inference_fn, corruption,
            time=t0, conditioning=conditioning, rng=None,
        ),
    )
    log_weights = jnp.zeros(initial_noise.shape[0], dtype=initial_noise.dtype)
    log_weights = log_weights + log_psi_prev

    def scan_body(carry, next_info):
      step_carry, log_w, log_psi_old, rng_state = carry
      xt, time = _xt_time(step_carry)

      rng_state, step_rng = jax.random.split(rng_state)
      step_rng_or_none = step_rng if uses_rng else None

      # One DenoiserFn per step, closed over the current (time, cond, rng).
      denoiser_fn = make_denoiser_fn(
          inference_fn, corruption,
          time=time, conditioning=conditioning, rng=step_rng_or_none,
      )

      # Raw native outputs -- kept as the uncorrected prediction so the
      # stepper can advance via its preferred parameterisation.
      raw_outputs = call_inference_fn(
          inference_fn, xt=xt, time=time,
          conditioning=conditioning, rng=step_rng_or_none,
      )

      x0_unc = denoiser_fn(xt)
      x0_cor = self._apply_correction(x0_unc, xt, time, denoiser_fn, schedule)

      outputs_uncorrected = raw_outputs
      outputs_corrected = _outputs_with_x0(raw_outputs, x0_cor)

      next_step = stepper.update(outputs_corrected, step_carry, next_info)
      xt_new, time_new = _xt_time(next_step)

      log_proposal_ratio = proposal_log_ratio(
          stepper=stepper,
          outputs_uncorrected=outputs_uncorrected,
          outputs_corrected=outputs_corrected,
          xt_prev=xt, xt_next=xt_new,
          time_prev=time, time_next=time_new,
          correction_identity=correction_identity,
      )

      # New twist at (xt_new, time_new); note: fresh denoiser_fn at new time.
      denoiser_fn_new = make_denoiser_fn(
          inference_fn, corruption,
          time=time_new, conditioning=conditioning, rng=step_rng_or_none,
      )
      log_psi_new = self._evaluate_twist(xt_new, time_new, denoiser_fn_new)

      log_w = log_w + log_proposal_ratio + (log_psi_new - log_psi_old)

      rng_state, resample_rng = jax.random.split(rng_state)
      indices, log_w_res = _resample_indices(
          self.resampler_fn, log_w, rng=resample_rng,
      )
      log_psi_new = log_psi_new[indices]
      next_step = _gather_particle_leaves(next_step, indices)

      return (next_step, log_w_res, log_psi_new, rng_state), None

    init_rng, scan_rng = jax.random.split(rng)
    carry_init = (first_step, log_weights, log_psi_prev, scan_rng)
    (last_before_carry, log_w_final, log_psi_final, _), _ = jax.lax.scan(
        scan_body, carry_init, next_infos,
    )

    # Final step (stepper.finalize advances to t=0).
    xt_last, time_last = _xt_time(last_before_carry)
    _, final_rng = jax.random.split(init_rng)
    final_rng_or_none = final_rng if uses_rng else None

    final_denoiser_fn = make_denoiser_fn(
        inference_fn, corruption,
        time=time_last, conditioning=conditioning, rng=final_rng_or_none,
    )
    final_raw_outputs = call_inference_fn(
        inference_fn, xt=xt_last, time=time_last,
        conditioning=conditioning, rng=final_rng_or_none,
    )
    x0_final_unc = final_denoiser_fn(xt_last)
    x0_final_cor = self._apply_correction(
        x0_final_unc, xt_last, time_last, final_denoiser_fn, schedule,
    )
    final_outputs_uncorrected = final_raw_outputs
    final_outputs_corrected = _outputs_with_x0(final_raw_outputs, x0_final_cor)
    final_step = stepper.finalize(
        final_outputs_corrected, last_before_carry, last_info,
    )

    if self.twist_fn is not None:
      xt_final, time_final = _xt_time(final_step)
      log_proposal_ratio_final = proposal_log_ratio(
          stepper=stepper,
          outputs_uncorrected=final_outputs_uncorrected,
          outputs_corrected=final_outputs_corrected,
          xt_prev=xt_last, xt_next=xt_final,
          time_prev=time_last, time_next=time_final,
          correction_identity=correction_identity,
      )
      denoiser_fn_final = make_denoiser_fn(
          inference_fn, corruption,
          time=time_final, conditioning=conditioning, rng=final_rng_or_none,
      )
      log_psi_at_final = self._evaluate_twist(
          xt_final, time_final, denoiser_fn_final,
      )
      log_w_final = log_w_final + log_proposal_ratio_final + (
          log_psi_at_final - log_psi_final
      )

    return final_step, None, log_w_final
