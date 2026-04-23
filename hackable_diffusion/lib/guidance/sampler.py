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

- ``correction``: an optional :class:`CorrectionFn` applied to the
  denoiser outputs before each step.  Covers Pi-GDM, cov-aware,
  isotropic projection, iterated Pi-GDM, and DPS (via
  :class:`GradientCorrectionFn`).
- ``twist``: an optional :class:`TwistFn` whose log-potential
  ``log psi(y | xt)`` drives the SMC importance weight.
- ``resampler``: an optional :class:`ResamplerFn` fired at each
  step.  :class:`NoResamplerFn` (default) gives single-trajectory
  behaviour; ``SystematicResamplerFn`` / ``MultinomialResamplerFn``
  give standard SMC.

The ``num_particles`` axis represents the SMC population.  Setting
``num_particles=1`` with ``twist=None`` and
``resampler=NoResamplerFn()`` delegates to the base sampler for
bit-for-bit equivalence.

Weight formula: at each step we accumulate

    delta_log_w = log_proposal_ratio + (log psi_new - log psi_old)

The proposal ratio ``log p_theta - log q`` is computed in closed form by
:func:`proposal_log_ratio` (dispatches on stepper type); it is zero for
an identity correction and for deterministic DDIM.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Callable

import jax
import jax.numpy as jnp

from hackable_diffusion.lib.sampling.sampling import DiffusionSampler
from hackable_diffusion.lib.guidance.protocols import (
    CorrectionFn,
    ResamplerFn,
    TwistFn,
)
from hackable_diffusion.lib.guidance.proposal_ratio import proposal_log_ratio
from hackable_diffusion.lib.guidance.resamplers import NoResamplerFn
from hackable_diffusion.lib.guidance.utils import (
    accepts_rng_kwarg,
    call_inference_fn,
)


def _accepts_inference_fn_kwarg(correction: CorrectionFn) -> bool:
  """True iff ``correction`` accepts an ``inference_fn`` kwarg."""
  try:
    sig = inspect.signature(correction.__call__)
  except (TypeError, ValueError):
    return False
  return "inference_fn" in sig.parameters


def _xt_time(step) -> tuple[jax.Array, jax.Array]:
  """Extract ``(xt, time)`` from a :class:`DiffusionStep`."""
  return step.xt, step.step_info.time


def _split_pytree_first_middle_last(tree):
  """Split a stacked step-info pytree into ``(first, middle, last)``."""
  return (
      jax.tree.map(lambda x: x[0], tree),
      jax.tree.map(lambda x: x[1:-1], tree),
      jax.tree.map(lambda x: x[-1], tree),
  )


def _gather_particle_leaves(carry, indices: jax.Array):
  """Gather every particle-indexed leaf by ``indices`` along axis 0.

  Leaves whose leading dim does not equal ``indices.shape[0]`` are
  assumed shared across particles (scalars, schedule times) and pass
  through unchanged.
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
    resampler: ResamplerFn,
    log_weights: jax.Array,
    *,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Return ``(gather_indices, new_log_weights)`` for a :class:`ResamplerFn`.

  The ``ResamplerFn`` protocol returns resampled particles; for SMC on
  steppers with non-trivial ``aux`` we need the indices so we can gather
  across the full carry pytree.
  """
  k = log_weights.shape[0]
  identity = jnp.arange(k)
  resampled_ids, new_log_weights = resampler(
      identity[:, None].astype(log_weights.dtype),
      log_weights,
      rng=rng,
  )
  indices = resampled_ids[:, 0].astype(jnp.int32)
  return indices, new_log_weights


@dataclasses.dataclass(kw_only=True, frozen=True)
class ConditionalDiffusionSampler:
  """Conditional sampler composing CorrectionFn, TwistFn, and ResamplerFn.

  The sampler orchestrates one reverse-diffusion pass across ``K``
  particles and automatically maintains the exact SMC importance weight
  by combining the twist increment with the closed-form proposal ratio
  from :func:`proposal_log_ratio`.

  Attributes:
    base_sampler: the underlying :class:`DiffusionSampler`.
    corruption_process: the corruption process used by the base sampler.
    correction_fn: optional per-step correction on the denoiser outputs.
    twist_fn: optional SMC log-potential ``log psi(y | x_t)``.
    resampler_fn: particle resampler (defaults to :class:`NoResamplerFn`).
    num_particles: number of SMC particles; 1 with no correction / twist
      / resampler delegates to the base sampler for bit-for-bit parity.
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

  def _split_inference(
      self,
      inference_fn: Callable,
      conditioning: Any,
  ) -> tuple[Callable, Callable, Callable]:
    """Return ``(raw_call, apply_correction, wrapped)``.

    - ``raw_call(xt, time, rng)`` returns the uncorrected outputs.
    - ``apply_correction(outputs, xt, time, rng)`` returns the corrected
      outputs (identity if ``self.correction_fn is None``).
    - ``wrapped(xt, time, rng)`` composes the two and is what the
      :class:`TwistFn` sees as ``inference_fn``.
    """
    correction_fn = self.correction_fn
    correction_needs_fn = (
        correction_fn is not None
        and _accepts_inference_fn_kwarg(correction_fn)
    )
    schedule = getattr(self.corruption_process, "schedule", None)
    outer_conditioning = conditioning

    def raw_call(xt, time, conditioning=None, rng=None):
      active_conditioning = (
          conditioning if conditioning is not None else outer_conditioning
      )
      return call_inference_fn(
          inference_fn, xt=xt, time=time,
          conditioning=active_conditioning, rng=rng,
      )

    def apply_correction(outputs, xt, time, rng=None):
      if correction_fn is None:
        return outputs
      if correction_needs_fn:
        return correction_fn(
            outputs, xt, time,
            schedule=schedule,
            corruption_process=self.corruption_process,
            conditioning=outer_conditioning, rng=rng,
            inference_fn=raw_call,
        )
      return correction_fn(
          outputs, xt, time,
          schedule=schedule,
          corruption_process=self.corruption_process,
          conditioning=outer_conditioning, rng=rng,
      )

    def wrapped(xt, time, conditioning=None, rng=None):
      outputs = raw_call(xt, time, conditioning=conditioning, rng=rng)
      return apply_correction(outputs, xt, time, rng=rng)

    return raw_call, apply_correction, wrapped

  def _evaluate_twist(
      self,
      xt: jax.Array,
      time: jax.Array,
      inference_fn: Callable,
      rng: jax.Array | None,
      conditioning: Any,
  ) -> jax.Array:
    if self.twist_fn is None:
      return jnp.zeros(xt.shape[0], dtype=xt.dtype)
    schedule = getattr(self.corruption_process, "schedule", None)
    return self.twist_fn(
        xt=xt, time=time,
        inference_fn=inference_fn,
        schedule=schedule,
        corruption_process=self.corruption_process,
        conditioning=conditioning, rng=rng,
    )

  def _run_loop(
      self,
      *,
      inference_fn: Callable,
      rng: jax.Array,
      initial_noise: jax.Array,
      conditioning: Any,
  ):
    k = initial_noise.shape[0]
    log_weights = jnp.zeros(k, dtype=initial_noise.dtype)

    raw_call, apply_correction, wrapped_fn = self._split_inference(
        inference_fn, conditioning,
    )
    correction_identity = self.correction_fn is None

    time_schedule = self.base_sampler.time_schedule
    stepper = self.base_sampler.stepper
    num_steps = self.base_sampler.num_steps

    all_infos = time_schedule.all_step_infos(rng, num_steps, initial_noise)
    first_info, next_infos, last_info = _split_pytree_first_middle_last(all_infos)
    first_step = stepper.initialize(initial_noise, first_info)

    # Initial twist evaluation.
    xt0, t0 = _xt_time(first_step)
    log_psi_prev = self._evaluate_twist(
        xt0, t0, inference_fn=wrapped_fn, rng=None, conditioning=conditioning,
    )
    log_weights = log_weights + log_psi_prev

    uses_rng = accepts_rng_kwarg(inference_fn)

    def scan_body(carry, next_info):
      step_carry, log_w, log_psi_old, rng_state = carry
      xt, time = _xt_time(step_carry)

      rng_state, step_rng = jax.random.split(rng_state)
      denoiser_kwargs = {"rng": step_rng} if uses_rng else {}

      outputs_uncorrected = raw_call(xt, time, **denoiser_kwargs)
      outputs_corrected = apply_correction(
          outputs_uncorrected, xt, time, **denoiser_kwargs,
      )
      next_step = stepper.update(outputs_corrected, step_carry, next_info)
      xt_new, time_new = _xt_time(next_step)

      log_proposal_ratio = proposal_log_ratio(
          stepper=stepper,
          corruption_process=self.corruption_process,
          outputs_uncorrected=outputs_uncorrected,
          outputs_corrected=outputs_corrected,
          xt_prev=xt, xt_next=xt_new,
          time_prev=time, time_next=time_new,
          correction_identity=correction_identity,
      )

      log_psi_new = self._evaluate_twist(
          xt_new, time_new, inference_fn=wrapped_fn,
          rng=None, conditioning=conditioning,
      )

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
    final_kwargs = {"rng": final_rng} if uses_rng else {}

    final_outputs_uncorrected = raw_call(xt_last, time_last, **final_kwargs)
    final_outputs_corrected = apply_correction(
        final_outputs_uncorrected, xt_last, time_last, **final_kwargs,
    )
    final_step = stepper.finalize(
        final_outputs_corrected, last_before_carry, last_info,
    )

    if self.twist_fn is not None:
      xt_final, time_final = _xt_time(final_step)
      log_proposal_ratio_final = proposal_log_ratio(
          stepper=stepper,
          corruption_process=self.corruption_process,
          outputs_uncorrected=final_outputs_uncorrected,
          outputs_corrected=final_outputs_corrected,
          xt_prev=xt_last, xt_next=xt_final,
          time_prev=time_last, time_next=time_final,
          correction_identity=correction_identity,
      )
      log_psi_at_final = self._evaluate_twist(
          xt_final, time_final, inference_fn=wrapped_fn,
          rng=None, conditioning=conditioning,
      )
      log_w_final = log_w_final + log_proposal_ratio_final + (
          log_psi_at_final - log_psi_final
      )

    return final_step, None, log_w_final
