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

"""Denoiser- and posterior-cloud-level composition primitives.

- :func:`make_denoiser_fn`: the canonical constructor: raw
  ``inference_fn`` + corruption process -> :class:`DenoiserFn` closure
  at a fixed ``(time, conditioning, rng)``.  Returns one prediction
  ``xt -> xhat_0(xt)``; for stochastic inference fns this is one
  posterior sample.
- :func:`make_posterior_cloud_fn`: the cloud-valued analogue.  Returns
  ``xt -> [B, R, *x0_shape]`` by splitting ``rng`` into ``R`` keys and
  vmap'ing :func:`make_denoiser_fn` over them.  Cloud-aware twists
  (``\\hat h_k^R = (1/R) \\sum L_y(x_0^r)``), projection guidance, and
  self-normalised endpoint MC consume the cloud closure.
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

from hackable_diffusion.lib.guidance.protocols import (
    DenoiserFn, PosteriorCloudFn,
)
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


def make_posterior_cloud_fn(
    inference_fn: Callable,
    corruption_process: Any,
    *,
    time: jax.Array,
    conditioning: Any = None,
    rng: jax.Array,
    population_size: int,
) -> PosteriorCloudFn:
  """Build a pure ``xt -> [B, R, *x0_shape]`` posterior-cloud closure.

  The cloud-valued analogue of :func:`make_denoiser_fn`.  Splits ``rng``
  into ``R = population_size`` independent keys and ``jax.vmap``s a
  per-sample :func:`make_denoiser_fn` over them, returning a closure that
  produces ``R`` posterior samples at any ``x_t``.

  For a :class:`PosteriorSamplerInferenceFn` (or any other stochastic
  inference fn that accepts ``rng``), the ``R`` calls draw ``R``
  independent posterior samples from ``\\hat p_{0|t}(. | x_t)`` -- the
  manuscript's clean-endpoint cloud.  For a deterministic inference fn,
  the ``R`` outputs are identical, which is the explicit mean-plug-in
  baseline used for ablations against full posterior MC.

  Output shape: ``(B, R, *x0_shape)`` -- the ``R`` axis is at position
  1, matching :func:`hackable_diffusion.lib.distributional.ensemble_apply`
  at training time so the same downstream code paths can read it.

  Args:
    inference_fn: The model.  Either a stochastic
      :class:`PosteriorSamplerInferenceFn` (R calls produce R distinct
      samples) or any deterministic inference fn (R calls produce R
      identical copies -- the mean-plug-in baseline).
    corruption_process: Used by the per-sample
      :func:`make_denoiser_fn` to convert raw network output to the
      soft ``x_0`` representation.
    time: The fixed time at which the cloud is built.
    conditioning: Optional conditioning passed to the inference fn.
    rng: PRNGKey to split into ``R`` per-sample subkeys.  Required --
      a cloud always needs splittable rng even if the per-sample
      inference fn happens to ignore it.
    population_size: The cloud size ``R``.  Must be ``>= 1``.

  Returns:
    A :class:`PosteriorCloudFn` closure that, given ``x_t``, returns a
    cloud of shape ``(B, R, *x0_shape)``.
  """
  if population_size < 1:
    raise ValueError(
        f"population_size must be >= 1, got {population_size}."
    )
  rngs = jax.random.split(rng, population_size)

  def cloud_fn(xt: jax.Array) -> jax.Array:
    def _one(rng_r: jax.Array) -> jax.Array:
      return make_denoiser_fn(
          inference_fn, corruption_process,
          time=time, conditioning=conditioning, rng=rng_r,
      )(xt)
    # vmap over the R rng axis; out_axes=1 puts R right after the batch
    # dim, matching ensemble_apply's [B, R, *data] training-time shape.
    return jax.vmap(_one, in_axes=0, out_axes=1)(rngs)

  return cloud_fn


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
