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

"""Particle resamplers for sequential Monte Carlo guidance.

All resamplers share the :class:`ResamplerFn` signature:

    ``(log_weights, *, rng) -> (indices, new_log_weights)``

The caller does the gather (``new_particles = old_particles[indices]``).
Contract: ``new_log_weights`` is set to ``log(mean(weights))`` for every
particle, so cumulative-weight estimators stay unbiased even after
resampling (Chopin and Papaspiliopoulos, Ch. 9).

- :class:`NoResamplerFn` is the identity (``indices = arange(K)``,
  weights unchanged) -- default for deterministic samplers.
- :class:`MultinomialResamplerFn` and :class:`SystematicResamplerFn` are
  the textbook choices.
- :class:`ESSThresholdedResamplerFn` wraps a base resampler and triggers
  only when the effective sample size falls below a fraction.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from hackable_diffusion.lib.guidance.protocols import ResamplerFn


def normalised_weights(log_weights: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Return ``(weights, log_mean_weight)`` from unnormalised log weights."""
  max_log = jnp.max(log_weights)
  shifted = log_weights - max_log
  weights = jnp.exp(shifted)
  total = jnp.sum(weights)
  mean = total / log_weights.shape[0]
  log_mean = jnp.log(mean) + max_log
  return weights / total, log_mean


@dataclasses.dataclass(kw_only=True, frozen=True)
class NoResamplerFn(ResamplerFn):
  """Identity resampler -- ``indices = arange(K)``, weights unchanged."""

  def __call__(
      self,
      log_weights: jax.Array,
      *,
      rng: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    del rng
    indices = jnp.arange(log_weights.shape[0], dtype=jnp.int32)
    return indices, log_weights


@dataclasses.dataclass(kw_only=True, frozen=True)
class MultinomialResamplerFn(ResamplerFn):
  """Draw indices i.i.d. from the normalised-weight categorical."""

  def __call__(
      self,
      log_weights: jax.Array,
      *,
      rng: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    weights, log_mean = normalised_weights(log_weights)
    k = log_weights.shape[0]
    indices = jax.random.categorical(
        rng, jnp.log(jnp.clip(weights, 1e-30, None)), shape=(k,),
    ).astype(jnp.int32)
    new_log_weights = jnp.full((k,), log_mean, dtype=log_weights.dtype)
    return indices, new_log_weights


@dataclasses.dataclass(kw_only=True, frozen=True)
class SystematicResamplerFn(ResamplerFn):
  """Stratified / systematic resampling (Kitagawa 1996)."""

  def __call__(
      self,
      log_weights: jax.Array,
      *,
      rng: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    weights, log_mean = normalised_weights(log_weights)
    k = log_weights.shape[0]
    u0 = jax.random.uniform(rng, (), minval=0.0, maxval=1.0 / k)
    grid = u0 + jnp.arange(k, dtype=weights.dtype) / k
    cumulative = jnp.cumsum(weights)
    indices = jnp.clip(jnp.searchsorted(cumulative, grid), 0, k - 1).astype(
        jnp.int32,
    )
    new_log_weights = jnp.full((k,), log_mean, dtype=log_weights.dtype)
    return indices, new_log_weights


@dataclasses.dataclass(kw_only=True, frozen=True)
class ESSThresholdedResamplerFn(ResamplerFn):
  """Apply ``base`` only when effective sample size drops below ``threshold``.

  ESS is ``(sum w_i)^2 / sum w_i^2``.  Normalised ESS in ``[0, 1]`` is
  ``ESS / K``.  When normalised ESS falls below ``threshold``, the wrapped
  resampler fires.
  """

  base: ResamplerFn
  threshold: float = 0.5

  def __call__(
      self,
      log_weights: jax.Array,
      *,
      rng: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    weights, _ = normalised_weights(log_weights)
    k = log_weights.shape[0]
    norm_ess = 1.0 / (k * jnp.sum(weights ** 2))
    should_resample = norm_ess < self.threshold

    indices, new_log_weights = self.base(log_weights, rng=rng)
    identity_indices = jnp.arange(k, dtype=indices.dtype)
    return (
        jnp.where(should_resample, indices, identity_indices),
        jnp.where(should_resample, new_log_weights, log_weights),
    )
