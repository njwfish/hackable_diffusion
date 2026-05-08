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

"""Twist functions: tractable log-potentials ``log psi(y | xt)``.

Every implementation consumes a :class:`DenoiserFn` and evaluates a
log-density at ``xhat_0(xt) = denoiser_fn(xt)``.  No raw
``inference_fn``, ``corruption_process``, ``rng``, or ``conditioning``
plumbing -- the closure lives inside ``denoiser_fn``.

- :class:`GaussianLikelihoodTwistFn`: linear-Gaussian observations
  ``y = A x_0 + N(0, sigma_y^2 I)``.
- :class:`DiscreteCompositionTwistFn` / its multi-head variant:
  multinomial observations on a simplex.
- :class:`ClassifierTwistFn`: arbitrary ``log p(y | x_0)``.
- :class:`EnergyTwistFn`: arbitrary scalar energy ``E(x_0)`` at
  inverse temperature ``1/T``.

Modality compatibility
----------------------
- ``GaussianLikelihoodTwistFn``: Euclidean-x0 (Gaussian ODE/SDE,
  posterior-sampler).  Not meaningful on a simplex.
- ``DiscreteCompositionTwistFn`` / multi-head: simplicial only.
- ``ClassifierTwistFn`` / ``EnergyTwistFn``: universal -- the caller
  decides what ``log_prob_fn`` / ``energy_fn`` accepts as input.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp

from hackable_diffusion.lib.guidance.protocols import (
    DenoiserFn, ForwardFn, PosteriorCloudFn, TwistFn,
)


################################################################################
# MARK: Gaussian likelihood
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class GaussianLikelihoodTwistFn(TwistFn):
  """Linear-Gaussian likelihood: ``y = A x_0 + N(0, sigma_y^2 I)``.

  ``log psi(y | xt) = log N(y; A xhat_0(xt), sigma_y^2 I)``

  For a hard constraint (``sigma_y -> 0``), use a small positive
  ``observation_noise`` to keep the twist smooth.  ``observation_noise = 0``
  gives a delta -- useful only at the final step of an inpainting-style
  pipeline.
  """

  observation: jax.Array
  forward_fn: ForwardFn
  observation_noise: float = 0.1

  def __call__(
      self,
      xt: jax.Array,
      time: jax.Array,
      *,
      denoiser_fn: DenoiserFn,
      cloud_fn: PosteriorCloudFn | None = None,
  ) -> jax.Array:
    del time, cloud_fn
    x0 = denoiser_fn(xt)
    residual = self.observation - self.forward_fn.forward(x0)
    flat = residual.reshape(residual.shape[0], -1)
    sigma2 = jnp.maximum(float(self.observation_noise) ** 2, 1e-30)
    return -0.5 * jnp.sum(flat ** 2, axis=-1) / sigma2


################################################################################
# MARK: Simplicial / discrete composition twists
################################################################################


def _categorical_block_log_likelihood(
    probs: jax.Array,
    forward_fn: ForwardFn,
    observation: jax.Array,
) -> jax.Array:
  """``sum_b y_b . log p_b`` where ``p_b`` is ``forward_fn`` applied to probs.

  ``probs`` has shape ``(B, n_child, K)`` (categories on the last axis).
  ``forward_fn`` aggregates along the site axis; we swap to expose
  sites as the last axis, aggregate, then swap back.
  """
  probs_sites_last = jnp.swapaxes(probs, -1, -2)        # (B, K, n_child)
  p_block_sites_last = forward_fn.forward(probs_sites_last)  # (B, K, n_parent)
  p_block = jnp.swapaxes(p_block_sites_last, -1, -2)    # (B, n_parent, K)
  log_p_block = jnp.log(jnp.clip(p_block, 1e-30, 1.0))
  y = jnp.broadcast_to(observation, p_block.shape)
  return jnp.sum(y * log_p_block, axis=(-2, -1))


@dataclasses.dataclass(kw_only=True, frozen=True)
class DiscreteCompositionTwistFn(TwistFn):
  """Multinomial log-likelihood of block-aggregated categorical probabilities.

  ``log psi(y | xt) = sum_b y_b . log p_b(xt)`` where ``p_b`` is the
  block-averaged simplex vector under the denoiser's prediction.
  ``softness`` scales the log-likelihood to widen the twist for
  numerical stability when the composition is near-deterministic.

  Assumes ``xhat_0(xt) = denoiser_fn(xt)`` is a simplex vector
  (``(B, n_child, K)``, softmax-normalised along the last axis).
  """

  observation: jax.Array
  forward_fn: ForwardFn
  softness: float = 1.0

  def __call__(
      self,
      xt: jax.Array,
      time: jax.Array,
      *,
      denoiser_fn: DenoiserFn,
      cloud_fn: PosteriorCloudFn | None = None,
  ) -> jax.Array:
    del time, cloud_fn
    probs = denoiser_fn(xt)
    return _categorical_block_log_likelihood(
        probs, self.forward_fn, self.observation,
    ) / float(self.softness)


@dataclasses.dataclass(kw_only=True, frozen=True)
class DiscreteMultiHeadCompositionTwistFn(TwistFn):
  """Multi-head simplicial composition twist.

  One observation + forward_fn per head; total twist is the sum of
  per-head multinomial log-likelihoods divided by ``softness``.
  """

  observations: tuple[jax.Array, ...]
  forward_fns: tuple[ForwardFn, ...]
  softness: float = 1.0

  def __post_init__(self):
    if len(self.observations) != len(self.forward_fns):
      raise ValueError(
          "observations and forward_fns must have matching length; "
          f"got {len(self.observations)} vs {len(self.forward_fns)}."
      )

  def __call__(
      self,
      xt: jax.Array,
      time: jax.Array,
      *,
      denoiser_fn: DenoiserFn,
      cloud_fn: PosteriorCloudFn | None = None,
  ) -> jax.Array:
    del time, cloud_fn
    probs = denoiser_fn(xt)
    total = jnp.zeros(probs.shape[0], dtype=probs.dtype)
    for obs, fwd in zip(self.observations, self.forward_fns):
      total = total + _categorical_block_log_likelihood(probs, fwd, obs)
    return total / float(self.softness)


################################################################################
# MARK: Classifier / energy twists
################################################################################


# Signatures of user-supplied callables for the generic twists.
LogProbFn = Callable[[jax.Array], jax.Array]   # (x0,) -> (B,)
EnergyFn = Callable[[jax.Array], jax.Array]    # (x0,) -> (B,)


@dataclasses.dataclass(kw_only=True, frozen=True)
class ClassifierTwistFn(TwistFn):
  """``log psi(y | xt) = log_prob_fn(xhat_0(xt))``.

  Generic external-log-probability twist.  Composed with
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
      denoiser_fn: DenoiserFn,
      cloud_fn: PosteriorCloudFn | None = None,
  ) -> jax.Array:
    del time, cloud_fn
    return self.log_prob_fn(denoiser_fn(xt))


@dataclasses.dataclass(kw_only=True, frozen=True)
class EnergyTwistFn(TwistFn):
  """``log psi(y | xt) = -energy_fn(xhat_0(xt)) / temperature``.

  Constraint-style guidance: sparsity / physics / smoothness penalties
  expressed as scalar energies.
  """

  energy_fn: EnergyFn
  temperature: float = 1.0

  def __call__(
      self,
      xt: jax.Array,
      time: jax.Array,
      *,
      denoiser_fn: DenoiserFn,
      cloud_fn: PosteriorCloudFn | None = None,
  ) -> jax.Array:
    del time, cloud_fn
    return -self.energy_fn(denoiser_fn(xt)) / float(self.temperature)


################################################################################
# MARK: Cloud-aware twists (posterior-bridges Algorithm 1)
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class EndpointTiltCloudTwistFn(TwistFn):
  """Posterior-MC estimator of ``log H_t(x_t) = log E[L_y(X_0) | x_t]``.

  Implements the manuscript's Algorithm 1 potential
  ``\\hat h_k^R(x) = (1/R) sum_r L_y(x_0^r)`` with
  ``x_0^r ~ \\hat p_{0|t}(. | x)`` from the cloud closure built by the
  sampler when ``posterior_cloud_size > 0``.  Returns ``log \\hat h_k``,
  so the existing scan body's ``log psi_new - log psi_old`` increment
  becomes ``log \\hat h_{k-1}(z) - log \\hat h_k(x)`` -- exactly the
  manuscript's incremental SMC weight ``a_k(z, x) = h_{k-1}(z) /
  h_k(x)`` once we exponentiate.

  Numerically: we evaluate ``log L_y(x_0^r)`` per cloud member and
  combine with ``logsumexp(log_L) - log R``, which is stable when the
  per-sample log-likelihoods span many orders of magnitude.

  Strict propriety regime: as ``R -> infty`` the estimator converges
  almost surely to ``H_t(x_t)`` (manuscript Corollary
  ``potential-consistency``); at finite ``R`` the SMC sampler is
  consistent in the large-particle limit.

  Attributes:
    log_L_y: ``[B, R, *x0_shape] -> [B, R]`` log-likelihood evaluator
      applied across the cloud's leading two axes.  Caller closes over
      the conditioning ``y``.  We ``jax.vmap`` over the ``R`` axis so a
      callable that takes ``[B, *x0_shape] -> [B]`` works without
      modification.
  """

  log_L_y: Callable[[jax.Array], jax.Array]

  def __call__(
      self,
      xt: jax.Array,
      time: jax.Array,
      *,
      denoiser_fn: DenoiserFn,
      cloud_fn: PosteriorCloudFn | None = None,
  ) -> jax.Array:
    del time, denoiser_fn  # cloud-only twist
    if cloud_fn is None:
      raise ValueError(
          "EndpointTiltCloudTwistFn requires cloud_fn; set "
          "ConditionalDiffusionSampler.posterior_cloud_size > 0 so the "
          "sampler builds an R-sample posterior cloud at each step."
      )
    cloud = cloud_fn(xt)                                      # [B, R, *data]
    # vmap log_L_y over the R axis so single-batch implementations work.
    log_L = jax.vmap(self.log_L_y, in_axes=1, out_axes=1)(cloud)  # [B, R]
    R = cloud.shape[1]
    log_R = jnp.log(jnp.asarray(R, dtype=log_L.dtype))
    return jax.nn.logsumexp(log_L, axis=-1) - log_R           # [B]


################################################################################
# MARK: Self-normalised endpoint Monte Carlo
################################################################################


def self_normalized_posterior_expectation(
    f: Callable[[jax.Array], jax.Array],
    log_L: Callable[[jax.Array], jax.Array],
    cloud: jax.Array,
) -> jax.Array:
  """Self-normalised MC estimator of ``E_{p_{0|t}^y}[f(X_0) | x_t]``.

  Manuscript Corollary ``self-normalized``:

      sum_r f(x_0^r) L_y(x_0^r) / sum_r L_y(x_0^r)
      ->  E[f(X_0) | x_t, Y = y]   a.s. as R -> infty,

  for any ``f`` integrable against the tilted posterior.  Implementation:
  vmap ``f`` and ``log_L`` over the ``R`` axis, normalise via
  ``softmax(log_L)``, and contract.

  Args:
    f: Per-sample functional ``[..., *x0_shape] -> [...]`` (or
      vectorisable via jax broadcasting).  We ``jax.vmap`` over the
      ``R`` axis.
    log_L: Per-sample log-likelihood ``[..., *x0_shape] -> [...]``.
    cloud: ``[B, R, *x0_shape]`` posterior cloud.

  Returns:
    ``[B, ...]`` estimator with the same trailing shape that ``f``
    produces per sample.
  """
  f_values = jax.vmap(f, in_axes=1, out_axes=1)(cloud)        # [B, R, ...]
  log_L_values = jax.vmap(log_L, in_axes=1, out_axes=1)(cloud)  # [B, R]
  weights = jax.nn.softmax(log_L_values, axis=-1)             # [B, R]
  # Broadcast weights over f's trailing axes.
  weight_shape = weights.shape + (1,) * (f_values.ndim - weights.ndim)
  weights_b = weights.reshape(weight_shape)
  return jnp.sum(weights_b * f_values, axis=1)                # [B, ...]
