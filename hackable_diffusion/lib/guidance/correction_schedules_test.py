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

"""Unit tests for per-step correction schedules."""

from __future__ import annotations

import unittest

import jax.numpy as jnp
import numpy as np

from hackable_diffusion.lib.guidance.correction_schedules import (
    IntervalSchedule,
    LinearRampSchedule,
    TimeBlendedCorrectionFn,
)


class LinearRampScheduleTest(unittest.TestCase):
    def test_endpoints_and_midpoint_for_ascending_range(self):
        # w(t=0) = 1.0; w(t=1) = 0.0; w(t=0.5) = 0.5.
        s = LinearRampSchedule(t_low=0.0, t_high=1.0, w_at_t_low=1.0, w_at_t_high=0.0)
        self.assertAlmostEqual(float(s(0.0)), 1.0, places=6)
        self.assertAlmostEqual(float(s(0.5)), 0.5, places=6)
        self.assertAlmostEqual(float(s(1.0)), 0.0, places=6)

    def test_clamps_outside_interval(self):
        s = LinearRampSchedule(t_low=0.2, t_high=0.8, w_at_t_low=0.0, w_at_t_high=1.0)
        self.assertAlmostEqual(float(s(0.0)), 0.0, places=6)
        self.assertAlmostEqual(float(s(1.0)), 1.0, places=6)
        self.assertAlmostEqual(float(s(0.5)), 0.5, places=6)

    def test_descending_range_orientation(self):
        # Specifying t_low > t_high flips the orientation.
        s = LinearRampSchedule(t_low=1.0, t_high=0.0, w_at_t_low=1.0, w_at_t_high=0.0)
        self.assertAlmostEqual(float(s(1.0)), 1.0, places=6)
        self.assertAlmostEqual(float(s(0.0)), 0.0, places=6)


class IntervalScheduleTest(unittest.TestCase):
    def test_inside_versus_outside(self):
        s = IntervalSchedule(t_low=0.2, t_high=0.7, inside_weight=1.0, outside_weight=0.0)
        self.assertAlmostEqual(float(s(0.5)), 1.0, places=6)
        self.assertAlmostEqual(float(s(0.1)), 0.0, places=6)
        self.assertAlmostEqual(float(s(0.9)), 0.0, places=6)
        # Endpoints inclusive.
        self.assertAlmostEqual(float(s(0.2)), 1.0, places=6)
        self.assertAlmostEqual(float(s(0.7)), 1.0, places=6)


class TimeBlendedCorrectionFnTest(unittest.TestCase):
    def test_blend_full_weight_passes_through_base_output(self):
        def base(x0, xt, time, **kw):
            del xt, time, kw
            return jnp.full_like(x0, 7.0)

        wrapped = TimeBlendedCorrectionFn(base=base, weight_fn=lambda t: 1.0)
        x0 = jnp.zeros((2, 4))
        xt = jnp.zeros_like(x0)
        out = wrapped(x0, xt, jnp.asarray(0.5))
        np.testing.assert_allclose(np.asarray(out), 7.0)

    def test_blend_zero_weight_returns_x0(self):
        def base(x0, xt, time, **kw):
            del xt, time, kw
            return jnp.full_like(x0, 7.0)

        wrapped = TimeBlendedCorrectionFn(base=base, weight_fn=lambda t: 0.0)
        x0 = jnp.full((2, 4), 1.5)
        out = wrapped(x0, x0, jnp.asarray(0.5))
        np.testing.assert_allclose(np.asarray(out), 1.5)

    def test_blend_intermediate_weight_lerps(self):
        def base(x0, xt, time, **kw):
            del xt, time, kw
            return jnp.full_like(x0, 10.0)

        wrapped = TimeBlendedCorrectionFn(base=base, weight_fn=lambda t: 0.25)
        x0 = jnp.full((2, 4), 2.0)
        out = wrapped(x0, x0, jnp.asarray(0.5))
        # 0.25 * 10 + 0.75 * 2 = 4.0
        np.testing.assert_allclose(np.asarray(out), 4.0)

    def test_blend_with_per_sample_time(self):
        # weight_fn returns a per-sample weight (B,) that broadcasts.
        def base(x0, xt, time, **kw):
            del xt, time, kw
            return jnp.full_like(x0, 10.0)

        weights = jnp.asarray([1.0, 0.0])
        wrapped = TimeBlendedCorrectionFn(base=base, weight_fn=lambda t: weights)
        x0 = jnp.stack([jnp.full((4,), 2.0), jnp.full((4,), 3.0)], axis=0)
        out = wrapped(x0, x0, jnp.asarray([0.5, 0.5]))
        np.testing.assert_allclose(np.asarray(out)[0], 10.0)
        np.testing.assert_allclose(np.asarray(out)[1], 3.0)


if __name__ == "__main__":
    unittest.main()
