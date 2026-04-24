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

"""M5 tests for :class:`StochasticInterpolant`.

Covers:

- Degenerate case ``gamma = 0`` reduces to the linear interpolant
  ``(1-t) x_0 + t x_1`` mathematically (for any ``z`` drawn).
- Velocity matches closed-form ``alpha'(t) x_0 + beta'(t) x_1 +
  gamma'(t) z`` for the canonical schedule.
- ``gamma(0) != 0`` and ``gamma(1) != 0`` at construction raise
  ``ValueError``.
- ``eval`` with ``z = None`` raises (safety against
  misconfiguration).
- ``InterpolantProcess(SI, VelocityOnlyTargets)`` end-to-end:
  ``corrupt`` emits ``{x0, x1, velocity}`` with shapes matching the
  input batch.
- ``schedule`` property exposes ``{alpha, beta, gamma}`` via
  ``evaluate``; lets downstream samplers peek at ``gamma(t)`` without
  new plumbing.
"""

from __future__ import annotations

import unittest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from hackable_diffusion.lib.corruption import base
from hackable_diffusion.lib.corruption import couplings
from hackable_diffusion.lib.corruption import interpolants
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.corruption import targets


class StochasticInterpolantTest(unittest.TestCase):

  def test_gamma_zero_reduces_to_linear(self):
    # alpha = 1 - t, beta = t, gamma = 0  <=>  LinearInterpolant(RFSchedule).
    si = interpolants.StochasticInterpolant(
        alpha=lambda t: 1.0 - t,
        beta=lambda t: t,
        gamma=lambda t: jnp.zeros_like(t),
    )
    linear = interpolants.LinearInterpolant(schedule=schedules.RFSchedule())
    key = jax.random.PRNGKey(0)
    x0 = jax.random.normal(key, (8, 4), dtype=jnp.float64)
    x1 = jax.random.normal(jax.random.PRNGKey(1), (8, 4), dtype=jnp.float64)
    t = jnp.linspace(0.05, 0.95, 8, dtype=jnp.float64)
    z = jax.random.normal(jax.random.PRNGKey(2), x0.shape, dtype=jnp.float64)

    xt_si, v_si = si.eval(x0, x1, t, z)
    xt_lin, v_lin = linear.eval(x0, x1, t)
    # gamma = 0 => z drops out of both xt and velocity.
    np.testing.assert_allclose(xt_si, xt_lin, atol=1e-12)
    np.testing.assert_allclose(v_si, v_lin, atol=1e-12)

  def test_velocity_matches_closed_form(self):
    # Canonical SI: alpha = 1-t, beta = t, gamma = sqrt(t(1-t)).
    si = interpolants.StochasticInterpolant(
        alpha=lambda t: 1.0 - t,
        beta=lambda t: t,
        gamma=interpolants.canonical_gamma,
    )
    key = jax.random.PRNGKey(0)
    x0 = jax.random.normal(key, (4, 3), dtype=jnp.float64)
    x1 = jax.random.normal(jax.random.PRNGKey(1), (4, 3), dtype=jnp.float64)
    z = jax.random.normal(jax.random.PRNGKey(2), x0.shape, dtype=jnp.float64)
    t = jnp.asarray([0.2, 0.4, 0.6, 0.8], dtype=jnp.float64)

    xt, v = si.eval(x0, x1, t, z)

    # Closed-form: alpha'(t) = -1, beta'(t) = 1, gamma'(t) = (1-2t)/(2 sqrt(t(1-t))).
    t_b = t[:, None]
    gamma_der = (1.0 - 2.0 * t_b) / (2.0 * jnp.sqrt(t_b * (1.0 - t_b)))
    v_expected = -x0 + x1 + gamma_der * z
    np.testing.assert_allclose(v, v_expected, atol=1e-10)

    # xt also matches closed form.
    xt_expected = (1.0 - t_b) * x0 + t_b * x1 + jnp.sqrt(t_b * (1.0 - t_b)) * z
    np.testing.assert_allclose(xt, xt_expected, atol=1e-12)

  def test_gamma_boundary_violation_raises(self):
    with self.assertRaisesRegex(ValueError, 'gamma'):
      interpolants.StochasticInterpolant(
          alpha=lambda t: 1.0 - t,
          beta=lambda t: t,
          gamma=lambda t: 0.3 + 0.0 * t,   # gamma(0) = gamma(1) = 0.3 != 0
      )

  def test_eval_requires_z(self):
    si = interpolants.StochasticInterpolant(
        alpha=lambda t: 1.0 - t,
        beta=lambda t: t,
        gamma=interpolants.canonical_gamma,
    )
    x0 = jnp.zeros((2, 2), dtype=jnp.float64)
    x1 = jnp.zeros((2, 2), dtype=jnp.float64)
    t = jnp.asarray([0.3, 0.7], dtype=jnp.float64)
    with self.assertRaisesRegex(ValueError, 'requires'):
      si.eval(x0, x1, t, z=None)

  def test_schedule_exposes_alpha_beta_gamma(self):
    si = interpolants.StochasticInterpolant(
        alpha=lambda t: 1.0 - t,
        beta=lambda t: t,
        gamma=interpolants.canonical_gamma,
    )
    t = jnp.asarray(0.25, dtype=jnp.float64)
    info = si.schedule.evaluate(t)
    self.assertEqual(set(info.keys()), {'alpha', 'beta', 'gamma'})
    np.testing.assert_allclose(float(info['alpha']), 0.75)
    np.testing.assert_allclose(float(info['beta']), 0.25)
    np.testing.assert_allclose(
        float(info['gamma']), float(jnp.sqrt(0.25 * 0.75)),
    )
    # Schedule *is* self -- no wrapper needed.
    self.assertIs(si.schedule, si)

  def test_interpolant_process_end_to_end(self):
    # InterpolantProcess(SI, IndependentCoupling(StandardNormal),
    # VelocityOnlyTargets): corrupt emits {x0, x1, velocity}.
    process = base.InterpolantProcess(
        coupling=couplings.IndependentCoupling(
            source=couplings.StandardNormalSource(),
        ),
        interpolant=interpolants.StochasticInterpolant(
            alpha=lambda t: 1.0 - t,
            beta=lambda t: t,
            gamma=interpolants.canonical_gamma,
        ),
        targets=targets.VelocityOnlyTargets(),
    )
    key = jax.random.PRNGKey(0)
    x0 = jax.random.normal(key, (16, 5), dtype=jnp.float64)
    t = jax.random.uniform(
        jax.random.PRNGKey(1), (16,), dtype=jnp.float64,
        minval=0.05, maxval=0.95,
    )
    xt, target_info = process.corrupt(jax.random.PRNGKey(2), x0, t)
    self.assertEqual(xt.shape, x0.shape)
    self.assertEqual(set(target_info.keys()), {'x0', 'x1', 'velocity'})
    self.assertEqual(target_info['velocity'].shape, x0.shape)

  def test_process_schedule_forwards_to_interpolant(self):
    # InterpolantProcess.schedule forwards to interpolant.schedule,
    # which for SI is self -- so a downstream sampler can read gamma(t)
    # via ``corruption_process.schedule.gamma(t)``.
    si = interpolants.StochasticInterpolant(
        alpha=lambda t: 1.0 - t,
        beta=lambda t: t,
        gamma=interpolants.canonical_gamma,
    )
    process = base.InterpolantProcess(
        coupling=couplings.IndependentCoupling(
            source=couplings.StandardNormalSource(),
        ),
        interpolant=si,
        targets=targets.VelocityOnlyTargets(),
    )
    t = jnp.asarray(0.4, dtype=jnp.float64)
    self.assertIs(process.schedule, si)
    np.testing.assert_allclose(
        float(process.schedule.gamma(t)),
        float(jnp.sqrt(0.4 * 0.6)),
    )

  def test_diffusion_degenerate_case(self):
    # Gaussian-diffusion-like setup: beta = 0, gamma = sigma(t), alpha = 1-t.
    # x_t = alpha * x_0 + gamma * z -- standard VP form with x_1 unused.
    si = interpolants.StochasticInterpolant(
        alpha=lambda t: 1.0 - t,
        beta=lambda t: jnp.zeros_like(t),
        gamma=lambda t: jnp.sqrt(jnp.clip(t * (1.0 - t), 0.0, None)),
    )
    key = jax.random.PRNGKey(0)
    x0 = jax.random.normal(key, (4, 2), dtype=jnp.float64)
    x1 = jax.random.normal(jax.random.PRNGKey(1), (4, 2), dtype=jnp.float64)
    z = jax.random.normal(jax.random.PRNGKey(2), x0.shape, dtype=jnp.float64)
    t = jnp.asarray([0.3, 0.5, 0.7, 0.9], dtype=jnp.float64)
    xt, _ = si.eval(x0, x1, t, z)
    # beta = 0 => x_1 dropped from xt.
    t_b = t[:, None]
    xt_expected = (1.0 - t_b) * x0 + jnp.sqrt(t_b * (1.0 - t_b)) * z
    np.testing.assert_allclose(xt, xt_expected, atol=1e-12)


if __name__ == '__main__':
  unittest.main()
