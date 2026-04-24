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

"""Twist functions: tractable surrogates for ``log p(y | xt)``.

A ``TwistFn`` evaluates ``log psi(y | xt)`` -- the SMC log-potential
used by TDS, MCGDiff, and (via :class:`GradientCorrectionFn`) DPS-style
score guidance.

Four published shapes of twist are covered:

- :class:`GaussianLikelihoodTwistFn`: linear-Gaussian observations
  ``y = A x_0 + N(0, sigma_y^2 I)``.  Canonical DPS / Pi-GDM choice.
- :class:`DiscreteCompositionTwistFn` and its multi-head variant:
  multinomial observations on a simplex (MCGDiff / cascade guidance).
- :class:`ClassifierTwistFn`: arbitrary external log-probability model
  ``log p(y | x_0)`` -- recovers classifier guidance when combined with
  :class:`GradientCorrectionFn`.
- :class:`EnergyTwistFn`: arbitrary scalar energy ``E(x_0)`` at an
  inverse temperature ``1/T`` -- ``log psi = -E(x_0) / T``.

All depend only on a :class:`ForwardFn` / user-supplied callable; none
hard-codes a state-space.

Modality compatibility
----------------------
- ``GaussianLikelihoodTwistFn``: Gaussian (ODE/SDE) and distributional.
  Requires a linear ``ForwardFn A`` whose range is Euclidean -- the
  twist is the log-density of a Gaussian at ``A xhat_0``.  Not
  applicable when ``x_0`` is on a simplex (use the discrete twists).
- ``DiscreteCompositionTwistFn`` /
  ``DiscreteMultiHeadCompositionTwistFn``: simplicial only.  The
  observation is a per-block multinomial / composition vector and
  ``forward_fn`` aggregates over sites.
- ``ClassifierTwistFn``: universal.  The ``log_prob_fn`` sees whatever
  ``x_0`` the corruption process produces -- simplex, Euclidean, or
  ensemble -- so the caller decides what's valid.
- ``EnergyTwistFn``: universal, same reasoning as classifier.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import jax
import jax.numpy as jnp

from hackable_diffusion.lib.guidance.protocols import ForwardFn, TwistFn
from hackable_diffusion.lib.guidance.utils import call_inference_fn


def _denoiser_x0(
    inference_fn: Callable,
    xt: jax.Array,
    time: jax.Array,
    *,
    corruption_process: Any,
    conditioning: Any = None,
    rng: jax.Array | None = None,
) -> jax.Array:
  """Evaluate ``inference_fn`` and convert its outputs to ``x0``."""
  outputs = call_inference_fn(
      inference_fn, xt=xt, time=time, conditioning=conditioning, rng=rng,
  )
  converted = corruption_process.convert_predictions(outputs, xt, time)
  return converted["x0"]


def _denoiser_log_probs(
    inference_fn: Callable,
    xt: jax.Array,
    time: jax.Array,
    *,
    corruption_process: Any,
    conditioning: Any = None,
    rng: jax.Array | None = None,
) -> jax.Array:
  """Evaluate ``inference_fn`` and return per-site categorical log-probs.

  Prefers converted ``logits`` when available; falls back to a clipped-log
  of the ``x0`` simplex prediction.
  """
  outputs = call_inference_fn(
      inference_fn, xt=xt, time=time, conditioning=conditioning, rng=rng,
  )
  converted = corruption_process.convert_predictions(outputs, xt, time)
  logits = converted.get("logits")
  if logits is None:
    logits = jnp.log(jnp.clip(converted["x0"], 1e-30, 1.0))
  return jax.nn.log_softmax(logits, axis=-1)


@dataclasses.dataclass(kw_only=True, frozen=True)
class GaussianLikelihoodTwistFn(TwistFn):
  """Twist for linear-Gaussian observations ``y = A x_0 + N(0, sigma_y^2 I)``.

  ``log psi(y | xt) = log N(y; A xhat_0(xt), sigma_y^2 I)`` computed via
  the denoiser's Tweedie output.  Shape-agnostic on xt; the
  :class:`ForwardFn` is applied to the per-particle x0 prediction.

  For a hard constraint (sigma_y -> 0) use a small positive
  ``observation_noise`` so the twist remains smooth.  Setting
  ``observation_noise`` to 0 gives a delta -- useful only at the final
  step of an inpainting-style pipeline.
  """

  observation: jax.Array
  forward_fn: ForwardFn
  observation_noise: float = 0.1

  def __call__(
      self,
      xt: jax.Array,
      time: jax.Array,
      *,
      inference_fn: Callable,
      schedule: Any,
      corruption_process: Any,
      conditioning: Any = None,
      rng: jax.Array | None = None,
  ) -> jax.Array:
    del schedule
    x0 = _denoiser_x0(
        inference_fn, xt, time,
        corruption_process=corruption_process,
        conditioning=conditioning, rng=rng,
    )
    residual = self.observation - self.forward_fn.forward(x0)
    flat = residual.reshape(residual.shape[0], -1)
    sigma2 = float(self.observation_noise) ** 2
    sigma2 = jnp.maximum(sigma2, 1e-30)
    return -0.5 * jnp.sum(flat ** 2, axis=-1) / sigma2


def _categorical_block_log_likelihood(
    log_p: jax.Array,
    forward_fn: ForwardFn,
    observation: jax.Array,
) -> jax.Array:
  """Compute ``sum_b y_b . log p_b`` where ``p_b = forward_fn(exp(log_p))``.

  ``log_p`` has shape ``(B, n_child, K)`` (per-site categorical log-probs).
  The forward map aggregates along the site axis; we temporarily swap the
  category axis out of the way so ``forward_fn.forward`` sees the sites as
  the last axis, matching :class:`ForwardFn`'s expected convention.
  """
  probs = jnp.exp(log_p)
  probs_af = jnp.swapaxes(probs, -1, -2)           # (B, K, n_child)
  p_block_af = forward_fn.forward(probs_af)         # (B, K, n_parent)
  p_block = jnp.swapaxes(p_block_af, -1, -2)        # (B, n_parent, K)
  log_p_block = jnp.log(jnp.clip(p_block, 1e-30, 1.0))
  y_block = jnp.broadcast_to(observation, p_block.shape)
  return jnp.sum(y_block * log_p_block, axis=(-2, -1))


@dataclasses.dataclass(kw_only=True, frozen=True)
class DiscreteCompositionTwistFn(TwistFn):
  """Twist for categorical composition observations on a child simplex.

  ``log psi(y | xt) = sum_b log Multinomial(y_b ; n_b, p_b(xt))``

  where ``p_b(xt)`` is the block-averaged categorical probability vector
  under the denoiser's predicted log-probs, and ``y_b`` is the observed
  per-block composition (counts or empirical-frequency vector).

  ``softness`` divides the log-likelihood by a scalar to effectively
  widen the twist -- useful for numerical stability when the composition
  is near-deterministic.
  """

  observation: jax.Array
  forward_fn: ForwardFn
  softness: float = 1.0

  def __call__(
      self,
      xt: jax.Array,
      time: jax.Array,
      *,
      inference_fn: Callable,
      schedule: Any,
      corruption_process: Any,
      conditioning: Any = None,
      rng: jax.Array | None = None,
  ) -> jax.Array:
    del schedule
    log_p = _denoiser_log_probs(
        inference_fn, xt, time,
        corruption_process=corruption_process,
        conditioning=conditioning, rng=rng,
    )
    log_lik = _categorical_block_log_likelihood(
        log_p, self.forward_fn, self.observation,
    )
    return log_lik / float(self.softness)


@dataclasses.dataclass(kw_only=True, frozen=True)
class DiscreteMultiHeadCompositionTwistFn(TwistFn):
  """Multi-head categorical composition twist on the child simplex.

  Discrete analog of :class:`GaussianLikelihoodTwistFn` with a stack of
  forward maps: for each head in ``forward_fns`` we aggregate the
  predicted per-site probabilities into block statistics and score the
  corresponding multinomial log-likelihood against the observed head.

  Reduces to :class:`DiscreteCompositionTwistFn` when the tuple has
  length one.  The per-head log-likelihood is

      log_psi_h = sum_{b} y_{b,h} * log p_{b,h}(xt)

  and the total twist is ``sum_h log_psi_h / softness``.
  """

  observations: tuple[jax.Array, ...]
  forward_fns: tuple[ForwardFn, ...]
  softness: float = 1.0

  def __post_init__(self):
    if len(self.observations) != len(self.forward_fns):
      raise ValueError(
          "observations and forward_fns must have matching length; got "
          f"{len(self.observations)} vs {len(self.forward_fns)}."
      )

  def __call__(
      self,
      xt: jax.Array,
      time: jax.Array,
      *,
      inference_fn: Callable,
      schedule: Any,
      corruption_process: Any,
      conditioning: Any = None,
      rng: jax.Array | None = None,
  ) -> jax.Array:
    del schedule
    log_p = _denoiser_log_probs(
        inference_fn, xt, time,
        corruption_process=corruption_process,
        conditioning=conditioning, rng=rng,
    )
    total = jnp.zeros(log_p.shape[0], dtype=log_p.dtype)
    for obs, forward_fn in zip(self.observations, self.forward_fns):
      total = total + _categorical_block_log_likelihood(log_p, forward_fn, obs)
    return total / float(self.softness)


# Signature of a log-prob callable used by ClassifierTwistFn: (x0,) -> (B,).
LogProbFn = Callable[[jax.Array], jax.Array]

# Signature of a scalar-energy callable used by EnergyTwistFn: (x0,) -> (B,).
EnergyFn = Callable[[jax.Array], jax.Array]


@dataclasses.dataclass(kw_only=True, frozen=True)
class ClassifierTwistFn(TwistFn):
  """``log psi(y | xt) = log_prob_fn(xhat_0(xt))``.

  Generic classifier / external likelihood twist: any callable that
  consumes ``x_0`` and returns a per-particle log-probability of the
  observation ``y`` slot into this twist.  Composed with
  :class:`GradientCorrectionFn` this reproduces classifier guidance
  (Dhariwal & Nichol 2021); composed with an SMC resampler it gives
  the classifier-guided TDS variant.

  ``log_prob_fn`` closes over the target ``y`` -- e.g.
  ``lambda x0: jax.nn.log_softmax(classifier(x0))[:, y_class]``.
  """

  log_prob_fn: LogProbFn

  def __call__(
      self,
      xt: jax.Array,
      time: jax.Array,
      *,
      inference_fn: Callable,
      schedule: Any,
      corruption_process: Any,
      conditioning: Any = None,
      rng: jax.Array | None = None,
  ) -> jax.Array:
    del schedule
    x0 = _denoiser_x0(
        inference_fn, xt, time,
        corruption_process=corruption_process,
        conditioning=conditioning, rng=rng,
    )
    return self.log_prob_fn(x0)


@dataclasses.dataclass(kw_only=True, frozen=True)
class EnergyTwistFn(TwistFn):
  """``log psi(y | xt) = -energy_fn(xhat_0(xt)) / temperature``.

  Generic unnormalised-density twist: any scalar per-particle energy
  ``E(x_0)`` defines a Boltzmann log-potential at inverse temperature
  ``1/T``.  Useful for constraint-style guidance where the target is
  expressed as "minimise this functional" rather than a Bayesian
  likelihood (hard clipping, sparsity penalties, physics-informed
  regularisers).
  """

  energy_fn: EnergyFn
  temperature: float = 1.0

  def __call__(
      self,
      xt: jax.Array,
      time: jax.Array,
      *,
      inference_fn: Callable,
      schedule: Any,
      corruption_process: Any,
      conditioning: Any = None,
      rng: jax.Array | None = None,
  ) -> jax.Array:
    del schedule
    x0 = _denoiser_x0(
        inference_fn, xt, time,
        corruption_process=corruption_process,
        conditioning=conditioning, rng=rng,
    )
    return -self.energy_fn(x0) / float(self.temperature)
