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
