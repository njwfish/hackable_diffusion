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

"""Small utilities shared across guidance-framework modules."""

from __future__ import annotations

import inspect
from typing import Any, Callable

import jax
import jax.numpy as jnp


def accepts_rng_kwarg(fn: Callable) -> bool:
  """True iff ``fn`` takes an ``rng`` parameter (directly or via ``**kwargs``)."""
  try:
    sig = inspect.signature(fn)
  except (TypeError, ValueError):
    return False
  params = sig.parameters
  if "rng" in params:
    return True
  return any(
      p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
  )


def call_inference_fn(
    inference_fn: Callable,
    *,
    xt: jax.Array,
    time: jax.Array,
    conditioning: Any = None,
    rng: jax.Array | None = None,
) -> dict[str, jax.Array]:
  """Call deterministic or stochastic inference functions uniformly."""
  if rng is not None and accepts_rng_kwarg(inference_fn):
    return inference_fn(xt=xt, time=time, conditioning=conditioning, rng=rng)
  return inference_fn(xt=xt, time=time, conditioning=conditioning)


def scalar_alpha(schedule: Any, time: jax.Array) -> jax.Array:
  """Return scalar ``alpha_t`` from a scalar-or-batch ``time`` input.

  Works for any schedule exposing ``.alpha(t)`` (Gaussian, simplicial,
  linear-discrete, ...).
  """
  t = jnp.atleast_1d(time).reshape(-1)[0:1]
  return schedule.alpha(t).reshape(())


def scalar_alpha_sigma(
    schedule: Any, time: jax.Array
) -> tuple[jax.Array, jax.Array]:
  """Return ``(alpha_t, sigma_t)`` for a scalar-or-batch ``time`` input.

  Requires the schedule to expose both ``.alpha`` and ``.sigma`` --
  i.e. a Gaussian-forward schedule.  Discrete schedules only have
  ``alpha``; use :func:`scalar_alpha` there.
  """
  t = jnp.atleast_1d(time).reshape(-1)[0:1]
  return schedule.alpha(t).reshape(()), schedule.sigma(t).reshape(())


def make_denoiser_fn(
    inference_fn: Callable,
    corruption_process: Any,
    *,
    time: jax.Array,
    conditioning: Any = None,
    rng: jax.Array | None = None,
) -> Callable[[jax.Array], jax.Array]:
  """Build a pure ``xt -> xhat_0(xt)`` closure at a fixed ``(time, cond, rng)``.

  Used by every correction / posterior-covariance implementation that
  needs to evaluate / differentiate the denoiser at a shifted xt: the
  closure is amenable to :func:`jax.jvp` (Tweedie covariance),
  :func:`jax.grad` (gradient corrections), and repeated invocation with
  different xt (iterated corrections).

  Fixing ``rng`` inside the closure is essential for differentiability
  of stochastic denoisers: the JVP traces a deterministic path at the
  captured noise realisation rather than re-sampling on every call.
  """

  def denoiser_fn(xt: jax.Array) -> jax.Array:
    outputs = call_inference_fn(
        inference_fn, xt=xt, time=time,
        conditioning=conditioning, rng=rng,
    )
    return corruption_process.convert_predictions(outputs, xt, time)["x0"]

  return denoiser_fn


def replace_x0(
    outputs: dict[str, jax.Array],
    x0_new: jax.Array,
    xt: jax.Array,
    time: jax.Array,
    corruption_process: Any,
) -> dict[str, jax.Array]:
  """Replace the x0 prediction in ``outputs`` and return the same key-set.

  Every correction produces an updated x0 but must hand back a dict in
  the denoiser's native prediction parameterisation (which may be
  ``x0``, ``eps``, ``velocity``, or something else).  This helper does
  the round-trip through ``convert_predictions`` once and returns the
  dict in the same shape as the input -- no ``next(iter(outputs.keys()))``
  guesswork at call sites.
  """
  converted = corruption_process.convert_predictions(
      {"x0": x0_new}, xt, time,
  )
  return {k: converted[k] for k in outputs.keys()}
