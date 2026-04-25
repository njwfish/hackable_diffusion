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

Six callable Protocols cover every published guidance method as a
composition:

- ``DenoiserFn``: pure ``xt -> xhat_0`` at a fixed ``(time, conditioning,
  rng)``.  The single atomic learned object; every other primitive
  consumes or produces one.
- ``ForwardFn``: measurement map ``y = A(x_0)``.  Adjoint is picked up
  automatically by ``jax.vjp``.
- ``PosteriorCovarianceFn``: linear operator ``v -> Cov(x_0 | x_t) v``,
  the Kalman-gain covariance term.
- ``CorrectionFn``: ``(x_0, xt, t) -> x_0_new`` observation-driven shift.
- ``TwistFn``: ``(xt, t) -> log psi(y | xt)``.
- ``ResamplerFn``: particle resample.

The :class:`StepKernel` from ``lib.sampling.base`` is the seventh
design axis -- built into each ``SamplerStep`` via its ``kernel``
method.

Modality compatibility
----------------------
``DenoiserFn``, ``ForwardFn``, ``ResamplerFn`` are universal.
``CorrectionFn`` and ``TwistFn`` are universal in shape but individual
implementations may assume Euclidean or simplex ``x_0``; see each
class's docstring.  ``PosteriorCovarianceFn`` implementations are
Gaussian-modality.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import jax


class DenoiserFn(Protocol):
  """Pure ``xt -> xhat_0(xt)`` map at a fixed ``(time, conditioning, rng)``.

  The atomic object.  Every other primitive that needs to evaluate or
  differentiate the denoiser consumes a ``DenoiserFn``: twists invoke it
  to score at ``xhat_0``; Tweedie-covariance JVPs it to get the
  Jacobian-vector product; iterated corrections re-call it at shifted
  ``xt``; and linear-blend denoisers compose several into one.

  Build one with :func:`make_denoiser_fn` from a raw ``inference_fn`` +
  ``corruption_process``.  The returned closure captures ``time``,
  ``conditioning``, and ``rng``, so JVP/grad through it trace a fixed
  noise realisation -- essential for differentiability of stochastic
  denoisers (distributional diffusion).
  """

  def __call__(self, xt: jax.Array) -> jax.Array: ...


class ForwardFn(Protocol):
  """Measurement / aggregation map ``y = A(x_0)``.

  A minimal Protocol: one ``forward(x)`` method.  The adjoint ``A^T``
  is obtained automatically via ``jax.vjp`` for linear maps -- no need
  to expose it explicitly.  Works across modalities as long as the
  caller supplies a map consistent with the ``x_0`` space (Euclidean
  for Gaussian diffusion, simplex-valued for simplicial, etc.).
  """

  def forward(self, x: jax.Array) -> jax.Array: ...


class PosteriorCovarianceFn(Protocol):
  """Matvec factory: ``(xt, time, ...) -> (v -> Cov(x_0 | x_t) v)``.

  Calling a :class:`PosteriorCovarianceFn` at a given step builds and
  returns a JAX-traceable closure that maps ``v`` to ``Cov v``.  This
  factory pattern lets implementations that need expensive one-time
  setup (the randomized-SVD sketch in
  :class:`LowRankTweediePosteriorCovarianceFn`) hoist the setup out of
  CG's inner loop -- :class:`KalmanCorrectionFn` calls the factory
  once per step, then invokes the returned closure per CG iteration.

  Built-in variants:

  - :class:`IsotropicPosteriorCovarianceFn`: ``Cov = scale(alpha, sigma) I``.
  - :class:`FixedPriorPosteriorCovarianceFn`: ``Cov = (sigma^2/alpha) C``
    for a known prior covariance ``C``.
  - :class:`PCAPosteriorCovarianceFn`: rank-``k`` PCA factor.
  - :class:`TweediePosteriorCovarianceFn`: exact Miyasawa JVP through
    the denoiser.  Requires a non-None ``denoiser_fn``.
  - :class:`LowRankTweediePosteriorCovarianceFn`: randomized-SVD sketch
    of the denoiser Jacobian, computed once per call.

  State-independent variants ignore ``denoiser_fn``; Tweedie variants
  require it and raise when absent.

  Modality: assumes a Euclidean inner product on ``x_0``; use a
  dedicated simplex-aware operator for simplicial diffusion.
  """

  def __call__(
      self,
      *,
      xt: jax.Array,
      time: jax.Array,
      schedule: Any,
      denoiser_fn: DenoiserFn | None = None,
  ) -> Callable[[jax.Array], jax.Array]: ...


class CorrectionFn(Protocol):
  """Observation-driven ``x_0 -> x_0_new`` shift applied per step.

  Takes the denoiser's ``xhat_0`` estimate, returns a corrected one.
  Composes polymorphically over an underlying ``DenoiserFn`` (for
  corrections that re-evaluate at shifted xt -- iterated, Tweedie-cov
  -- or differentiate through it -- gradient corrections).

  Intentionally narrow signature: no ``outputs`` dict, no
  ``corruption_process``, no ``inference_fn``/``conditioning``/``rng``
  (all threaded via ``denoiser_fn``).  The sampler converts
  ``{"x0": x0_new}`` back to the stepper's native prediction type at
  the boundary -- once per step, not inside every correction.
  """

  def __call__(
      self,
      x0: jax.Array,
      xt: jax.Array,
      time: jax.Array,
      *,
      denoiser_fn: DenoiserFn,
      schedule: Any,
  ) -> jax.Array: ...


class TwistFn(Protocol):
  """SMC log-potential ``log psi(y | xt)``.

  The canonical plug-in ``log psi := log p(y | xhat_0(xt))`` evaluates
  the likelihood at the denoiser's Tweedie output.  Implementations
  consume only a :class:`DenoiserFn` -- no raw ``inference_fn``,
  ``corruption_process``, or ``rng`` plumbing.

  ``jax.grad(twist_fn, argnums=0)`` (at fixed ``time, denoiser_fn``)
  gives the DPS gradient: that's what :class:`GradientCorrectionFn`
  consumes.
  """

  def __call__(
      self,
      xt: jax.Array,
      time: jax.Array,
      *,
      denoiser_fn: DenoiserFn,
  ) -> jax.Array: ...


class ResamplerFn(Protocol):
  """Resample particles according to log-weights.

  Pure operation: no knowledge of the diffusion state space, schedule,
  or twist.  After resampling, ``new_log_weights`` is set to
  ``log(mean(weights))`` for every particle so cumulative-weight
  estimators stay unbiased (Chopin and Papaspiliopoulos, Ch. 9).
  """

  def __call__(
      self,
      particles: jax.Array,
      log_weights: jax.Array,
      *,
      rng: jax.Array,
  ) -> tuple[jax.Array, jax.Array]: ...
