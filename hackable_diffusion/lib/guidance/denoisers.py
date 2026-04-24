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

"""Denoiser-level composition primitives.

- :func:`make_denoiser_fn`: the canonical constructor: raw
  ``inference_fn`` + corruption process -> :class:`DenoiserFn` closure
  at a fixed ``(time, conditioning, rng)``.
- :class:`LinearBlendDenoiserFn`: arbitrary linear combination of
  base denoisers.  Covers classifier-free guidance as a scalar special
  case (see :func:`cfg_denoiser_fn`); also supports multi-condition
  blends (positive + negative + null, denoiser ensembling, etc.).
- :func:`make_cfg_inference_fn`: the inference-level CFG composition.
  Takes two ``inference_fn``s (conditional + unconditional) and a
  :class:`GuidanceFn` from ``lib.inference.guidance``, returns a single
  ``inference_fn`` that blends their outputs before the sampler sees
  them.  This is the simplest CFG entry point -- no framework
  correction needed.

CFG is not a correction in the framework sense: it doesn't condition on
an observation, it composes two denoisers.  Treating it at the
denoiser / inference-fn layer keeps the ``CorrectionFn`` primitive
strictly reserved for observation-driven ``x_0`` shifts.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import jax
import jax.numpy as jnp

from hackable_diffusion.lib.guidance.protocols import DenoiserFn
from hackable_diffusion.lib.guidance.utils import call_inference_fn


def make_denoiser_fn(
    inference_fn: Callable,
    corruption_process: Any,
    *,
    time: jax.Array,
    conditioning: Any = None,
    rng: jax.Array | None = None,
) -> DenoiserFn:
  """Build a pure ``xt -> xhat_0(xt)`` closure at a fixed ``(time, cond, rng)``.

  Returns the *soft* ``xhat_0`` -- a differentiable, real-valued
  representation of the denoiser's prediction:

  - Gaussian corruption: the Tweedie ``x_0`` directly.
  - Simplicial corruption: ``softmax(logits)`` -- the simplex
    probability vector, not the hard ``argmax`` integer that
    ``convert_predictions`` exposes under the ``x_0`` key.  Twists
    and gradient corrections need the soft form to differentiate.

  Every primitive that needs to evaluate / differentiate the denoiser
  consumes the :class:`DenoiserFn` built here: JVP (Tweedie posterior
  covariance), grad (DPS), repeated invocation at shifted xt (iterated
  corrections).  Fixing ``rng`` inside the closure is essential for
  differentiability of stochastic denoisers.
  """

  def denoiser_fn(xt: jax.Array) -> jax.Array:
    outputs = call_inference_fn(
        inference_fn, xt=xt, time=time,
        conditioning=conditioning, rng=rng,
    )
    converted = corruption_process.convert_predictions(outputs, xt, time)
    x0 = converted["x0"]
    # Simplicial ``x_0`` is argmax(logits) (integer) -- unusable
    # downstream.  Fall back to softmax(logits) whenever the x_0 slot
    # comes back integer-typed.
    if jnp.issubdtype(x0.dtype, jnp.integer) and "logits" in converted:
      return jax.nn.softmax(converted["logits"], axis=-1)
    return x0

  return denoiser_fn


@dataclasses.dataclass(kw_only=True, frozen=True)
class LinearBlendDenoiserFn(DenoiserFn):
  """Linear combination of base denoisers at fixed scalar weights.

  ``xhat_0(xt) = sum_i weights[i] * denoisers[i](xt)``

  Scalar CFG is a two-element blend with weights ``(1 + w, -w)``.
  Multi-condition CFG (positive + negative + null) is three elements.
  Denoiser ensembling is uniform weights summing to one.  The blend
  operates in ``x_0`` space -- the sampler converts to the stepper's
  native prediction type at the boundary.
  """

  denoisers: tuple[DenoiserFn, ...]
  weights: tuple[float, ...]

  def __post_init__(self):
    if len(self.denoisers) != len(self.weights):
      raise ValueError(
          "denoisers and weights must have matching length; got "
          f"{len(self.denoisers)} vs {len(self.weights)}."
      )

  def __call__(self, xt: jax.Array) -> jax.Array:
    if not self.denoisers:
      raise ValueError("LinearBlendDenoiserFn needs at least one denoiser.")
    total = self.weights[0] * self.denoisers[0](xt)
    for w, d in zip(self.weights[1:], self.denoisers[1:]):
      total = total + w * d(xt)
    return total


def make_cfg_inference_fn(
    conditional_inference_fn: Callable,
    unconditional_inference_fn: Callable,
    guidance_fn: Any,
) -> Callable:
  """Build a CFG-blended ``inference_fn`` from a cond + uncond pair.

  Calls both at every step and combines their outputs via any
  :class:`lib.inference.guidance.GuidanceFn` (e.g.
  :class:`ScalarGuidanceFn`, :class:`LimitedIntervalGuidanceFn`).  The
  returned callable has the standard ``inference_fn`` signature and
  plugs directly into :class:`ConditionalDiffusionSampler` -- the
  sampler doesn't need to know CFG is happening.

  This is the preferred CFG entry point: CFG is denoiser composition,
  not a correction.  Build your blended model once, hand it to the
  sampler as ``inference_fn``, done.
  """

  def blended(xt, time, conditioning=None, rng=None):
    cond_outputs = call_inference_fn(
        conditional_inference_fn, xt=xt, time=time,
        conditioning=conditioning, rng=rng,
    )
    uncond_outputs = call_inference_fn(
        unconditional_inference_fn, xt=xt, time=time,
        conditioning=None, rng=rng,
    )
    return guidance_fn(
        xt=xt,
        conditioning=conditioning,
        time=time,
        cond_outputs=cond_outputs,
        uncond_outputs=uncond_outputs,
    )

  return blended


def cfg_denoiser_fn(
    conditional_denoiser: DenoiserFn,
    unconditional_denoiser: DenoiserFn,
    guidance: float,
) -> LinearBlendDenoiserFn:
  """Build a CFG-blended :class:`DenoiserFn` from two base denoisers.

  Scalar CFG with the standard formula
  ``(1 + guidance) * cond - guidance * uncond``.  Use this when you
  already have two :class:`DenoiserFn` closures (e.g. inside an
  :class:`IteratedCorrectionFn`'s inner loop); prefer
  :func:`make_cfg_inference_fn` when you have raw ``inference_fn``s
  and want a single blended one to hand to the sampler.
  """
  return LinearBlendDenoiserFn(
      denoisers=(conditional_denoiser, unconditional_denoiser),
      weights=(1.0 + guidance, -guidance),
  )
