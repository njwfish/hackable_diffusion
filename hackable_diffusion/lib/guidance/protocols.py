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

"""Core protocols for the composable guidance framework.

Five protocols cover every published guidance method:

- ``ForwardFn``: linear / non-linear measurement map ``y = A(x_0)``
  (shared by twists and the Pi-GDM correction).
- ``PosteriorCovarianceFn``: linear operator ``v -> Cov(x_0 | x_t) v``
  (plug-in point for Isotropic / Tweedie / fixed-prior Kalman variants
  of Pi-GDM).
- ``CorrectionFn``: modifies the denoiser's outputs dict before the
  sampler advances.  Covers Pi-GDM, covariance-aware, isotropic
  projection, iterated Pi-GDM, and DPS (via ``GradientCorrectionFn``).
- ``TwistFn``: evaluates the log-potential ``log psi(y | xt)``.  Used by
  SMC methods (TDS, MCGDiff) for importance weights and by DPS-style
  methods as a gradient source.
- ``ResamplerFn``: pure operation on a particle/weight pair.

The orchestrator :class:`ConditionalDiffusionSampler` (in ``sampler.py``)
composes the above around a ``DiffusionSampler`` to produce conditional
samples.  The K=1 case with ``correction=None, twist=None`` delegates to
the base sampler bit-for-bit.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import jax


class ForwardFn(Protocol):
  """Linear / non-linear forward map ``y = A(x)`` for inverse-problem twists.

  A minimal Protocol: a single ``forward(x)`` method that takes a batched
  input and returns the aggregated / measured output.  Any object with a
  matching ``forward`` method satisfies this -- no base-class inheritance
  required.  Used by :class:`GaussianLikelihoodTwistFn`, the discrete
  composition twists, and :class:`PiGDMCorrectionFn`, which together
  cover every first- and second-order guidance method in the framework.

  The adjoint ``A^T`` is obtained automatically via ``jax.vjp`` for
  linear ``forward`` implementations -- no need to supply it explicitly.
  """

  def forward(self, x: jax.Array) -> jax.Array:
    ...


class PosteriorCovarianceFn(Protocol):
  """Linear operator ``v -> Cov(x_0 | x_t) v`` at a given ``(xt, time)``.

  Used by :class:`PiGDMCorrectionFn` as the Kalman-gain covariance term.
  Three built-in variants cover the published choices:

  - :class:`IsotropicPosteriorCovarianceFn` (``Cov = scale(alpha, sigma) I``)
  - :class:`FixedPriorPosteriorCovarianceFn` (known prior covariance)
  - :class:`TweediePosteriorCovarianceFn` (via Miyasawa JVP through the
    denoiser -- exact under any prior the denoiser represents).

  Implementations may ignore any unused kwarg; the fully generic variant
  needs ``denoiser_x0`` (a pre-composed ``xt -> xhat_0`` closure),
  state-independent variants only use ``(time, schedule)``.
  """

  def __call__(
      self,
      v: jax.Array,
      *,
      xt: jax.Array,
      time: jax.Array,
      schedule: Any,
      denoiser_x0: Callable[[jax.Array], jax.Array] | None = None,
  ) -> jax.Array: ...


class CorrectionFn(Protocol):
  """Modify the denoiser's outputs dict before the step advances.

  ``outputs`` is the dict returned by ``inference_fn``, already converted
  to whatever prediction type the corruption process exposes.  A
  ``Correction`` returns a new outputs dict with the same keys; it should
  modify only the keys that represent the denoiser's x0/logits prediction
  (the remaining keys pass through unchanged, per the
  ``corruption_process.convert_predictions`` contract).

  Shape agnosticism: implementations must work for ``(B, ...)``
  (single-particle), ``(B, M, ...)`` (distributional ensemble), and
  folded ``(B*H, ...)`` (moment family) layouts.
  """

  def __call__(
      self,
      outputs: dict[str, jax.Array],
      xt: jax.Array,
      time: jax.Array,
      *,
      schedule: Any,
      corruption_process: Any,
      conditioning: Any = None,
      rng: jax.Array | None = None,
  ) -> dict[str, jax.Array]: ...


class TwistFn(Protocol):
  """Evaluate the SMC log-potential ``log psi(y | xt)``.

  Approximates the intractable ``log p(y | xt)`` by a tractable
  surrogate; the canonical choice is the plug-in
  ``log p(y | xhat_0(xt))`` where ``xhat_0`` is the Tweedie denoiser
  output.

  The twist is separate from the correction because the two axes
  compose independently:

  - DPS uses a twist as a gradient source (no correction, no SMC).
  - Pi-GDM uses a correction (no twist).
  - TDS uses a twist for SMC weighting (any correction for the proposal).

  Implementations typically call ``inference_fn`` internally to obtain
  ``xhat_0``; the RNG is threaded through in case the denoiser is
  stochastic (distributional diffusion).
  """

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
    """Return ``log psi`` shape ``(particle_batch,)``."""
    ...


class ResamplerFn(Protocol):
  """Resample particles according to log-weights.

  Pure operation: no knowledge of the diffusion state space, the
  schedule, or the twist.  Implementations include ``NoResamplerFn``
  (identity, for deterministic samplers), ``MultinomialResamplerFn``,
  ``SystematicResamplerFn``, and ``ESSThresholdedResamplerFn`` (resample
  only when effective sample size falls below a fraction).

  Contract: after resampling, ``log_weights`` is set to the log of the
  mean weight so that cumulative-weight estimators remain unbiased
  (Chopin and Papaspiliopoulos, "An Introduction to Sequential Monte
  Carlo", Ch. 9).
  """

  def __call__(
      self,
      particles: jax.Array,
      log_weights: jax.Array,
      *,
      rng: jax.Array,
  ) -> tuple[jax.Array, jax.Array]: ...
