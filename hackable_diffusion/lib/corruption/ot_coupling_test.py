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

"""Integration tests for :class:`MiniBatchOTCoupling` as a re-pairer.

Covers:

- ``__call__`` permutes ``x_0`` so paired indices come from the input
  pool, with the right shape; **``x_1`` order is preserved** so any
  conditioning aligned with ``x_1`` stays index-aligned.
- ``jit`` compiles without per-step recompiles.
- Gradients don't flow through the coupling output
  (``jax.lax.stop_gradient`` semantics).
- ``assert_vmappable`` raises on the OT coupling.
- OT-CFM on a 2D toy (half-moons -> circles): loss decreases over 200
  training steps; ``x_1`` is supplied as a paired-data input to
  ``corrupt``, no prior involved.
"""

from __future__ import annotations

import unittest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax

from hackable_diffusion.lib.corruption import base
from hackable_diffusion.lib.corruption import couplings
from hackable_diffusion.lib.corruption import interpolants
from hackable_diffusion.lib.corruption import priors
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.corruption import targets


def _sample_moons(key: jax.Array, batch: int) -> jax.Array:
  """Two-half-moons 2D distribution."""
  k1, k2 = jax.random.split(key)
  theta = jax.random.uniform(k1, (batch,), minval=0.0, maxval=jnp.pi)
  labels = jax.random.bernoulli(k2, 0.5, (batch,))
  x = jnp.where(labels, jnp.cos(theta), 1 - jnp.cos(theta))
  y = jnp.where(labels, jnp.sin(theta), 0.5 - jnp.sin(theta))
  return jnp.stack([x, y], axis=-1).astype(jnp.float64)


def _sample_circle(key: jax.Array, batch: int) -> jax.Array:
  theta = jax.random.uniform(
      key, (batch,), minval=0.0, maxval=2 * jnp.pi, dtype=jnp.float64,
  )
  return jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)


class MiniBatchOTCouplingTest(unittest.TestCase):

  def test_permutes_x0_preserves_x1(self):
    # x_1 is the canonical pool (preserved); x_0 is the moons batch
    # (permuted by OT so each matches the corresponding x_1).
    key = jax.random.PRNGKey(0)
    x0_pool = _sample_moons(key, batch=16)
    theta = jnp.linspace(0.0, 2 * jnp.pi, 16, dtype=jnp.float64)[:, None]
    x1 = jnp.concatenate([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    coupling = couplings.MiniBatchOTCoupling(epsilon=0.05)
    x0_out, x1_out = coupling(key, x0_pool, x1)
    self.assertEqual(x0_out.shape, x0_pool.shape)
    self.assertEqual(x1_out.shape, x1.shape)
    # x_1 is passed through unchanged -- key invariant for downstream
    # conditioning aligned with x_1.
    np.testing.assert_array_equal(np.asarray(x1_out), np.asarray(x1))
    # Every matched x_0 must be one of the input-pool rows.
    for i in range(x0_out.shape[0]):
      diffs = jnp.linalg.norm(x0_pool - x0_out[i:i + 1], axis=-1)
      self.assertLess(
          float(jnp.min(diffs)), 1e-8,
          f"Matched x_0[{i}] is not in the input pool",
      )

  def test_stop_gradient_around_plan(self):
    # Gradients w.r.t. x_1 through the (permuted) x_0 output should be zero
    # -- the plan is a non-differentiable matching signal.
    key = jax.random.PRNGKey(0)
    x0_pool = jax.random.normal(key, (8, 2), dtype=jnp.float64)
    coupling = couplings.MiniBatchOTCoupling(epsilon=0.05)

    def loss(x1):
      x0_out, _ = coupling(key, x0_pool, x1)
      return jnp.sum(x0_out ** 2)

    x1 = jax.random.normal(jax.random.PRNGKey(1), (8, 2), dtype=jnp.float64)
    grad = jax.grad(loss)(x1)
    self.assertTrue(bool(jnp.all(grad == 0.0)))

  def test_jit_compiles_and_reuses_trace(self):
    key = jax.random.PRNGKey(0)
    x1_pool = jax.random.normal(key, (16, 2), dtype=jnp.float64)
    coupling = couplings.MiniBatchOTCoupling(epsilon=0.05)
    compile_count = [0]

    @jax.jit
    def _call(key, x0):
      compile_count[0] += 1  # counted on trace; inside jit this stays 1.
      return coupling(key, x0, x1_pool)

    x0_a = _sample_moons(jax.random.PRNGKey(1), batch=16)
    x0_b = _sample_moons(jax.random.PRNGKey(2), batch=16)
    _ = _call(key, x0_a)
    _ = _call(key, x0_b)  # same shape -> trace reused
    self.assertEqual(compile_count[0], 1)

  def test_assert_vmappable_rejects_ot(self):
    coupling = couplings.MiniBatchOTCoupling()
    process = base.InterpolantProcess(
        prior=priors.GaussianPrior(),
        coupling=coupling,
        interpolant=interpolants.LinearInterpolant(schedule=schedules.RFSchedule()),
        targets=targets.VelocityOnlyTargets(),
    )
    with self.assertRaises(ValueError):
      couplings.assert_vmappable(process)


def _tiny_mlp_init(key, in_dim=2, hidden=64):
  k1, k2, k3 = jax.random.split(key, 3)
  return {
      "w1": jax.random.normal(k1, (in_dim + 1, hidden), dtype=jnp.float64) * 0.3,
      "w2": jax.random.normal(k2, (hidden, hidden), dtype=jnp.float64) * 0.3,
      "w3": jax.random.normal(k3, (hidden, in_dim), dtype=jnp.float64) * 0.3,
  }


def _tiny_mlp_apply(params, xt, time):
  t = jnp.broadcast_to(time.reshape(-1, 1), (xt.shape[0], 1))
  h = jnp.concatenate([xt, t], axis=-1)
  h = jax.nn.gelu(h @ params["w1"])
  h = jax.nn.gelu(h @ params["w2"])
  return h @ params["w3"]


class OtFlowMatchingTest(unittest.TestCase):

  def test_moons_to_circles_loss_decreases(self):
    # x_0 ~ moons.  x_1 ~ unit circle (paired-data input to corrupt).
    # OT permutes within the batch so (x_0[i], x_1[i]) minimises cost.
    process = base.InterpolantProcess(
        coupling=couplings.MiniBatchOTCoupling(epsilon=0.05),
        interpolant=interpolants.LinearInterpolant(
            schedule=schedules.RFSchedule(),
        ),
        targets=targets.VelocityOnlyTargets(),
    )

    rng = jax.random.PRNGKey(0)
    init_rng, rng = jax.random.split(rng)
    params = _tiny_mlp_init(init_rng)
    optimizer = optax.adam(learning_rate=3e-3)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, rng):
      k_x0, k_x1, k_corrupt, k_t = jax.random.split(rng, 4)
      x0 = _sample_moons(k_x0, batch=64)
      x1 = _sample_circle(k_x1, batch=64)
      t = jax.random.uniform(k_t, (64,), dtype=jnp.float64)
      xt, target_info = process.corrupt(k_corrupt, x0, t, x1=x1)
      def loss_fn(p):
        pred = _tiny_mlp_apply(p, xt, t)
        return jnp.mean((pred - target_info["velocity"]) ** 2)
      loss, grads = jax.value_and_grad(loss_fn)(params)
      updates, opt_state = optimizer.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      return params, opt_state, loss

    losses = []
    for _ in range(200):
      rng, step_rng = jax.random.split(rng)
      params, opt_state, loss = step(params, opt_state, step_rng)
      losses.append(float(loss))

    loss_first_10 = np.mean(losses[:10])
    loss_last_10 = np.mean(losses[-10:])
    self.assertLess(
        loss_last_10, loss_first_10 / 3.0,
        msg=(
            f"OT-CFM did not converge: init-avg={loss_first_10:.4f}, "
            f"final-avg={loss_last_10:.4f}"
        ),
    )


if __name__ == "__main__":
  unittest.main()
