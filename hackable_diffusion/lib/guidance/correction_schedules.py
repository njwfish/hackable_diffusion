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

"""Per-step time schedules for fading a :class:`CorrectionFn`.

A schedule is any ``Callable[[time], weight_in_[0, 1]]``.  Wrapping a
bound correction in :class:`TimeBlendedCorrectionFn` linearly blends the
corrected ``x_0`` with the uncorrected ``x_0`` according to the
schedule's weight at the current sampling time, letting the user fade
guidance along the trajectory without touching the underlying
correction.

Two built-in schedule shapes cover the common recipes:

- :class:`LinearRampSchedule` -- ramp from ``w_at_t_low`` to
  ``w_at_t_high`` between two times.  Standard "fade in"
  (``w(t=1)=0``, ``w(t=0)=1``) or "fade out" patterns.
- :class:`IntervalSchedule` -- indicator-style ``inside_weight`` /
  ``outside_weight`` over a time window.  Useful for "only apply
  correction near the clean endpoint" (``IntervalSchedule(t_low=0,
  t_high=0.3)``).

These are deliberately tiny -- the schedule is just a scalar function
of time; richer schedules can be passed directly as any callable.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class TimeBlendedCorrectionFn:
    """Wrap a bound correction so its output blends toward ``x0`` per step.

    The reverse-diffusion step calls ``correction(x0, xt, time, ...)``
    once per sampling step.  This wrapper computes
    ``x0_corr = base(...)`` and returns
    ``x0_corr * w(time) + x0 * (1 - w(time))`` so a scalar weight in
    ``[0, 1]`` lets the user fade the correction along the trajectory
    without touching the underlying correction code.
    """

    base: Any
    weight_fn: Callable[[jax.Array], jax.Array]

    def __call__(self, x0, xt, time, **kw):
        x0_corr = self.base(x0, xt, time, **kw)
        w = jnp.asarray(self.weight_fn(time), dtype=x0.dtype)
        while w.ndim < x0.ndim:
            w = w[..., None]
        return x0_corr * w + x0 * (1.0 - w)


@dataclasses.dataclass(frozen=True)
class LinearRampSchedule:
    """Linear ramp from ``w_at_t_low`` at ``t_low`` to ``w_at_t_high`` at ``t_high``.

    Clamps outside ``[min(t_low, t_high), max(t_low, t_high)]``.  Useful
    schedules:

    - ``LinearRampSchedule(t_low=0.0, t_high=1.0, w_at_t_low=1.0, w_at_t_high=0.0)``
      hard at clean (``t=0``), zero at pure noise (``t=1``).  Standard
      "fade in" pattern -- correct only late in sampling.
    - ``LinearRampSchedule(t_low=0.0, t_high=1.0, w_at_t_low=0.0, w_at_t_high=1.0)``
      zero at clean, hard at noise.  Fades out as sampling progresses
      so the model can refine details without being clamped at the end.
    """

    t_low: float = 0.0
    t_high: float = 1.0
    w_at_t_low: float = 1.0
    w_at_t_high: float = 0.0

    def __call__(self, time):
        t = jnp.asarray(time, dtype=jnp.float32)
        t_lo, t_hi = float(min(self.t_low, self.t_high)), float(max(self.t_low, self.t_high))
        w_lo, w_hi = float(self.w_at_t_low), float(self.w_at_t_high)
        denom = max(t_hi - t_lo, 1e-12)
        frac = jnp.clip((t - t_lo) / denom, 0.0, 1.0)
        if self.t_low > self.t_high:
            frac = 1.0 - frac
        return w_lo + frac * (w_hi - w_lo)


@dataclasses.dataclass(frozen=True)
class IntervalSchedule:
    """Indicator-style schedule: ``inside_weight`` on ``[t_low, t_high]``, ``outside_weight`` elsewhere."""

    t_low: float = 0.0
    t_high: float = 1.0
    inside_weight: float = 1.0
    outside_weight: float = 0.0

    def __call__(self, time):
        t = jnp.asarray(time, dtype=jnp.float32)
        is_inside = jnp.logical_and(t >= float(self.t_low), t <= float(self.t_high))
        return jnp.where(
            is_inside,
            jnp.asarray(self.inside_weight, dtype=jnp.float32),
            jnp.asarray(self.outside_weight, dtype=jnp.float32),
        )
