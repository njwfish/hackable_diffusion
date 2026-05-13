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

from hackable_diffusion.lib.sampling.sampling import (
    DiffusionSampler, _concat_pytree,
)
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


@dataclasses.dataclass(kw_only=True, frozen=True)
class ConditionalDiffusionSampler:
  """Conditional sampler composing CorrectionFn, TwistFn, and ResamplerFn.

  With no correction, no twist, and the identity resampler, delegates
  directly to ``base_sampler`` for bit-for-bit parity (returning the
  base sampler's ``(final_step, trajectory_or_none)`` 2-tuple).

  Otherwise returns ``(final_step, trajectory_or_none, log_w_final)``,
  where ``trajectory_or_none`` mirrors ``base_sampler.return_trajectory``:
  a stacked :class:`DiffusionStepTree` over all ``num_steps`` steps when
  trajectory return is on, and ``None`` otherwise.  The intermediate
  steps emitted into the stack are post-resample, so the trajectory
  reflects the particles actually carried forward at each step.

  Attributes:
    base_sampler: the underlying :class:`DiffusionSampler`.  Its
      ``return_trajectory`` flag controls whether this sampler emits a
      stacked trajectory (the SMC-loop path mirrors the base sampler's
      behaviour).
    corruption_process: the corruption process; used to build per-step
      :class:`DenoiserFn` closures.
    correction_fn: optional ``x0 -> x0_new`` correction.
    twist_fn: optional ``log psi(y | xt)`` potential.
    resampler_fn: particle resampler (defaults to :class:`NoResamplerFn`).
      The SMC population size is ``initial_noise.shape[0]``; with no
      correction / twist / non-identity resampler the call short-circuits
      to ``base_sampler``.
    resample_until_step_frac: fraction of the trajectory after which
      ``resampler_fn`` is replaced by the identity (``NoResamplerFn``).
      ``1.0`` (default) resamples for the entire trajectory; ``0.8``
      stops resampling for the last 20% of steps.  Late-stage steps have
      tiny step-noise, so resampled duplicates can no longer diverge --
      stopping the resampler there preserves population diversity at
      the price of slightly looser posterior weighting.  This is the
      TDS (Wu et al. 2023) recipe for diffusion SMC.
  """

  base_sampler: DiffusionSampler
  corruption_process: Any
  correction_fn: CorrectionFn | None = None
  twist_fn: TwistFn | None = None
  resampler_fn: ResamplerFn = dataclasses.field(default_factory=NoResamplerFn)
  resample_until_step_frac: float = 1.0

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
    return_trajectory = bool(self.base_sampler.return_trajectory)
    correction_identity = self.correction_fn is None
    uses_rng = accepts_rng_kwarg(inference_fn)

    all_infos = time_schedule.all_step_infos(rng, num_steps, initial_noise)
    first_info, next_infos, last_info = _split_first_middle_last(all_infos)
    first_step = stepper.initialize(initial_noise, first_info)

    # Resampling cutoff: stop after this many scan-body steps (i.e. middle
    # steps; the final stepper.finalize step never resamples).  Clamp to
    # [0, num_middle_steps] so a frac of 0 disables resampling entirely
    # and a frac of 1 keeps it on through the last middle step.
    num_middle_steps = max(num_steps - 2, 0)
    resample_cutoff = int(
        round(float(self.resample_until_step_frac) * num_middle_steps)
    )
    resample_cutoff = max(0, min(resample_cutoff, num_middle_steps))

    def _build_denoiser(time, rng_or_none):
      return make_denoiser_fn(
          inference_fn, corruption,
          time=time, conditioning=conditioning, rng=rng_or_none,
      )

    def _correct_and_advance(*, current_step, advance_fn, rng_or_none):
      """Build denoiser, correct ``x_0``, advance to the next step.

      Returns ``(next_step, outputs_uncorrected, outputs_corrected)``.
      Used by both the scan body (``advance_fn = stepper.update``) and the
      final-step block (``advance_fn = stepper.finalize``); the only
      difference is which ``stepper`` method advances the state.
      """
      xt, time = _xt_time(current_step)
      denoiser_fn = _build_denoiser(time, rng_or_none)
      raw_outputs = call_inference_fn(
          inference_fn, xt=xt, time=time,
          conditioning=conditioning, rng=rng_or_none,
      )
      x0_unc = denoiser_fn(xt)
      x0_cor = self._apply_correction(x0_unc, xt, time, denoiser_fn, schedule)
      outputs_corrected = _outputs_with_x0(raw_outputs, x0_cor)
      next_step = advance_fn(outputs_corrected, current_step)
      return next_step, raw_outputs, outputs_corrected

    # Initial twist evaluation at (xt_0, t_0).  Stochastic inference fns
    # need an rng to draw their per-call noise; deterministic inference
    # fns ignore it entirely.  Split only in the stochastic branch so the
    # rng stream feeding the scan body is bit-identical to the pre-fix
    # behaviour for every existing deterministic-inference-fn caller.
    if uses_rng:
      rng, initial_twist_rng = jax.random.split(rng)
    else:
      initial_twist_rng = None
    xt0, t0 = _xt_time(first_step)
    log_psi_prev = self._evaluate_twist(
        xt0, t0, denoiser_fn=_build_denoiser(t0, initial_twist_rng),
    )
    # Log weights must be floating-point.  ``initial_noise.dtype`` is
    # float for Gaussian / simplicial corruption but integer for the
    # MDM token state, where using the noise dtype would produce an
    # int32 weight array and trigger a ``scan`` carry dtype mismatch
    # the moment the proposal log-ratio (always float) is added.
    log_weights = jnp.zeros(initial_noise.shape[0], dtype=jnp.float32)
    log_weights = log_weights + log_psi_prev

    def scan_body(carry, scan_input):
      step_carry, log_w, log_psi_old, rng_state, step_idx = carry
      next_info = scan_input

      rng_state, step_rng = jax.random.split(rng_state)
      step_rng_or_none = step_rng if uses_rng else None

      xt, time = _xt_time(step_carry)
      next_step, outputs_uncorrected, outputs_corrected = _correct_and_advance(
          current_step=step_carry,
          advance_fn=lambda outputs, cur: stepper.update(outputs, cur, next_info),
          rng_or_none=step_rng_or_none,
      )
      xt_new, time_new = _xt_time(next_step)

      log_proposal_ratio = proposal_log_ratio(
          stepper=stepper,
          outputs_uncorrected=outputs_uncorrected,
          outputs_corrected=outputs_corrected,
          xt_prev=xt, xt_next=xt_new,
          time_prev=time, time_next=time_new,
          correction_identity=correction_identity,
      )
      log_psi_new = self._evaluate_twist(
          xt_new, time_new, _build_denoiser(time_new, step_rng_or_none),
      )
      log_w = log_w + log_proposal_ratio + (log_psi_new - log_psi_old)

      rng_state, resample_rng = jax.random.split(rng_state)
      indices, log_w_res = self.resampler_fn(log_w, rng=resample_rng)

      # Gate resampling on the step counter: past the cutoff we keep the
      # current particles and weights so the population can spread out
      # again under independent step noise.
      should_resample_step = step_idx < resample_cutoff
      log_psi_after = jnp.where(
          should_resample_step, log_psi_new[indices], log_psi_new,
      )
      log_w_after = jnp.where(should_resample_step, log_w_res, log_w)
      next_step_after = jax.lax.cond(
          should_resample_step,
          lambda: _gather_particle_leaves(next_step, indices),
          lambda: next_step,
      )

      new_carry = (
          next_step_after, log_w_after, log_psi_after, rng_state, step_idx + 1,
      )
      # Emit the post-resample step into the stacked trajectory so it
      # reflects the particles actually carried forward.
      scan_emit = next_step_after if return_trajectory else None
      return new_carry, scan_emit

    # Reserve a separate rng tree for the final stepper.finalize call;
    # taking ``finalize_seed`` from the *first* half of the split keeps
    # ``scan_rng`` (the second half) bit-identical to the pre-refactor
    # behaviour for every existing caller.
    finalize_seed, scan_rng = jax.random.split(rng)
    _, final_rng = jax.random.split(finalize_seed)
    carry_init = (first_step, log_weights, log_psi_prev, scan_rng, 0)
    (last_before_carry, log_w_final, log_psi_final, _, _), intermediate_steps = (
        jax.lax.scan(scan_body, carry_init, next_infos)
    )

    # Final step (stepper.finalize advances to t=0).
    xt_last, time_last = _xt_time(last_before_carry)
    final_rng_or_none = final_rng if uses_rng else None
    final_step, final_outputs_uncorrected, final_outputs_corrected = (
        _correct_and_advance(
            current_step=last_before_carry,
            advance_fn=lambda outputs, cur: stepper.finalize(
                outputs, cur, last_info,
            ),
            rng_or_none=final_rng_or_none,
        )
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
      log_psi_at_final = self._evaluate_twist(
          xt_final, time_final,
          _build_denoiser(time_final, final_rng_or_none),
      )
      log_w_final = log_w_final + log_proposal_ratio_final + (
          log_psi_at_final - log_psi_final
      )

    trajectory = (
        _concat_pytree(first_step, intermediate_steps, final_step)
        if return_trajectory else None
    )
    return final_step, trajectory, log_w_final
