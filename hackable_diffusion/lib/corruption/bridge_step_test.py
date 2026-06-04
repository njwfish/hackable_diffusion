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

"""Tests for the two-endpoint bridge step ``K_{s|0,t}`` on interpolants.

The defining property checked throughout is *endpoint recovery*: if a
state ``x_t`` was generated from endpoints ``(x_0, x_1)`` at time ``t``,
then for a deterministic bridge the step back to ``s`` must land on the
same path's ``x_s`` (``eval(x_0, x_1, s)``).  For the stochastic
(Brownian-bridge) interpolant we instead check the conditional mean /
variance against the closed form of Posterior Bridges eq. (5).
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from hackable_diffusion.lib import manifolds
from hackable_diffusion.lib.corruption import interpolants
from hackable_diffusion.lib.corruption import schedules

from absl.testing import absltest


class LinearBridgeStepTest(absltest.TestCase):

  def test_recovers_path_endpoint(self):
    # A deterministic linear bridge from (x0, xt) must reproduce the exact
    # x_s of the path that generated xt -- x1 is determined by (x0, xt).
    interp = interpolants.LinearInterpolant(schedule=schedules.CosineSchedule())
    x0 = jax.random.normal(jax.random.PRNGKey(0), (8, 4), dtype=jnp.float64)
    x1 = jax.random.normal(jax.random.PRNGKey(1), (8, 4), dtype=jnp.float64)
    t = jnp.full((8,), 0.7, dtype=jnp.float64)
    s = jnp.full((8,), 0.3, dtype=jnp.float64)

    xt, _ = interp.eval(x0, x1, t)
    xs_true, _ = interp.eval(x0, x1, s)
    xs = interp.bridge_step(x0=x0, xt=xt, t=t, s=s)

    np.testing.assert_allclose(xs, xs_true, atol=1e-10)

  def test_collapses_to_x0_at_zero(self):
    # s = 0 => K_{0|0,t} = delta_{x0}.
    interp = interpolants.LinearInterpolant(schedule=schedules.RFSchedule())
    x0 = jax.random.normal(jax.random.PRNGKey(0), (8, 4), dtype=jnp.float64)
    x1 = jax.random.normal(jax.random.PRNGKey(1), (8, 4), dtype=jnp.float64)
    t = jnp.full((8,), 0.5, dtype=jnp.float64)
    xt, _ = interp.eval(x0, x1, t)

    xs = interp.bridge_step(
        x0=x0, xt=xt, t=t, s=jnp.zeros((8,), dtype=jnp.float64)
    )
    np.testing.assert_allclose(xs, x0, atol=1e-10)


class StochasticBridgeStepTest(absltest.TestCase):

  def _canonical(self, sigma_br):
    return interpolants.StochasticInterpolant(
        alpha=lambda t: 1.0 - t,
        beta=lambda t: t,
        gamma=lambda t: sigma_br * interpolants.canonical_gamma(t),
    )

  def test_gamma_zero_matches_linear_deterministic(self):
    si = self._canonical(0.0)
    linear = interpolants.LinearInterpolant(schedule=schedules.RFSchedule())
    x0 = jax.random.normal(jax.random.PRNGKey(0), (8, 4), dtype=jnp.float64)
    x1 = jax.random.normal(jax.random.PRNGKey(1), (8, 4), dtype=jnp.float64)
    t = jnp.full((8,), 0.8, dtype=jnp.float64)
    s = jnp.full((8,), 0.2, dtype=jnp.float64)
    xt, _ = linear.eval(x0, x1, t)

    xs_si = si.bridge_step(x0=x0, xt=xt, t=t, s=s, key=jax.random.PRNGKey(7))
    xs_lin = linear.bridge_step(x0=x0, xt=xt, t=t, s=s)
    np.testing.assert_allclose(xs_si, xs_lin, atol=1e-10)

  def test_conditional_mean_and_variance(self):
    # Closed form (eq. 5): with rho = lambda_s / lambda_t = s / t,
    #   x_s = (1 - rho) x0 + rho xt + sigma_br sqrt(lambda_s (1 - rho)) Z.
    sigma_br = 1.3
    si = self._canonical(sigma_br)
    x0 = jnp.asarray([[1.0, -2.0, 0.5, 3.0]], dtype=jnp.float64)
    xt = jnp.asarray([[0.0, 1.0, -1.0, 2.0]], dtype=jnp.float64)
    t_val, s_val = 0.8, 0.3
    t = jnp.asarray([t_val], dtype=jnp.float64)
    s = jnp.asarray([s_val], dtype=jnp.float64)

    keys = jax.random.split(jax.random.PRNGKey(0), 20000)
    draws = jax.vmap(
        lambda k: si.bridge_step(x0=x0, xt=xt, t=t, s=s, key=k)
    )(keys)  # (N, 1, 4)
    draws = draws[:, 0, :]

    rho = s_val / t_val
    mean_true = (1.0 - rho) * x0[0] + rho * xt[0]
    std_true = sigma_br * np.sqrt(s_val * (1.0 - rho))

    np.testing.assert_allclose(draws.mean(0), mean_true, atol=2e-2)
    np.testing.assert_allclose(
        draws.std(0), np.full(4, std_true), rtol=3e-2
    )

  def test_collapses_to_x0_at_zero(self):
    si = self._canonical(1.0)
    x0 = jax.random.normal(jax.random.PRNGKey(0), (8, 4), dtype=jnp.float64)
    xt = jax.random.normal(jax.random.PRNGKey(1), (8, 4), dtype=jnp.float64)
    t = jnp.full((8,), 0.6, dtype=jnp.float64)
    xs = si.bridge_step(
        x0=x0, xt=xt, t=t, s=jnp.zeros((8,), dtype=jnp.float64),
        key=jax.random.PRNGKey(2),
    )
    np.testing.assert_allclose(xs, x0, atol=1e-10)


class GeodesicBridgeStepTest(absltest.TestCase):

  def _sphere_points(self, key, shape):
    x = jax.random.normal(key, shape, dtype=jnp.float64)
    return x / jnp.linalg.norm(x, axis=-1, keepdims=True)

  def test_recovers_path_endpoint_on_sphere(self):
    interp = interpolants.GeodesicInterpolant(
        manifold=manifolds.Sphere(),
        schedule=schedules.LinearRiemannianSchedule(),
    )
    x0 = self._sphere_points(jax.random.PRNGKey(0), (8, 3))
    x1 = self._sphere_points(jax.random.PRNGKey(1), (8, 3))
    t = jnp.full((8,), 0.7, dtype=jnp.float64)
    s = jnp.full((8,), 0.25, dtype=jnp.float64)

    xt, _ = interp.eval(x0, x1, t)
    xs_true, _ = interp.eval(x0, x1, s)
    xs = interp.bridge_step(x0=x0, xt=xt, t=t, s=s)

    np.testing.assert_allclose(xs, xs_true, atol=1e-9)

  def test_collapses_to_x0_at_zero(self):
    interp = interpolants.GeodesicInterpolant(
        manifold=manifolds.Sphere(),
        schedule=schedules.LinearRiemannianSchedule(),
    )
    x0 = self._sphere_points(jax.random.PRNGKey(0), (8, 3))
    x1 = self._sphere_points(jax.random.PRNGKey(1), (8, 3))
    t = jnp.full((8,), 0.6, dtype=jnp.float64)
    xt, _ = interp.eval(x0, x1, t)
    xs = interp.bridge_step(
        x0=x0, xt=xt, t=t, s=jnp.zeros((8,), dtype=jnp.float64)
    )
    np.testing.assert_allclose(xs, x0, atol=1e-9)


if __name__ == "__main__":
  absltest.main()
