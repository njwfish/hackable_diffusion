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

"""M4 integration tests for :class:`MiniBatchOTCoupling`.

Covers:

- ``sample`` produces a batch of matched ``x_1`` with the right shape.
- ``jit`` compiles without per-step recompiles.
- Gradients don't flow through the coupling output
  (``jax.lax.stop_gradient`` semantics).
- ``assert_vmappable`` raises on the OT coupling.
- OT-CFM on a 2D toy (half-moons -> circles): loss decreases over 200
  training steps, matching the plan's acceptance criterion.
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


class _FixedBatchSource:
  """Coupling fixture that returns a pre-computed batch."""

  is_batch_level = False

  def __init__(self, fixed_batch: jax.Array):
    self._batch = fixed_batch

  @property
  def marginal(self):
    return self

  def sample(self, key, x0):
    del key, x0
    return self._batch


class MiniBatchOTCouplingTest(unittest.TestCase):

  def test_sample_shape_and_indices_valid(self):
    key = jax.random.PRNGKey(0)
    x0 = _sample_moons(key, batch=16)
    # Target: unit circle points.
    theta = jnp.linspace(0.0, 2 * jnp.pi, 16, dtype=jnp.float64)[:, None]
    x1_pool = jnp.concatenate([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    source = _FixedBatchSource(x1_pool)
    coupling = couplings.MiniBatchOTCoupling(source=source, epsilon=0.05)
    x1 = coupling.sample(key, x0)
    self.assertEqual(x1.shape, x0.shape)
    # Every matched x_1 must be one of the source pool rows.
    for i in range(x1.shape[0]):
      diffs = jnp.linalg.norm(x1_pool - x1[i:i + 1], axis=-1)
      self.assertLess(
          float(jnp.min(diffs)), 1e-8,
          f"Matched x_1[{i}] is not in the source pool",
      )

  def test_stop_gradient_around_plan(self):
    # Gradients w.r.t. x_0 through the coupling output should be zero.
    key = jax.random.PRNGKey(0)
    x1_pool = jax.random.normal(key, (8, 2), dtype=jnp.float64)
    coupling = couplings.MiniBatchOTCoupling(
        source=_FixedBatchSource(x1_pool), epsilon=0.05,
    )

    def loss(x0):
      x1 = coupling.sample(key, x0)
      return jnp.sum(x1 ** 2)

    x0 = jax.random.normal(jax.random.PRNGKey(1), (8, 2), dtype=jnp.float64)
    grad = jax.grad(loss)(x0)
    self.assertTrue(bool(jnp.all(grad == 0.0)))

  def test_jit_compiles_and_reuses_trace(self):
    key = jax.random.PRNGKey(0)
    x1_pool = jax.random.normal(key, (16, 2), dtype=jnp.float64)
    coupling = couplings.MiniBatchOTCoupling(
        source=_FixedBatchSource(x1_pool), epsilon=0.05,
    )
    compile_count = [0]

    @jax.jit
    def _call(key, x0):
      compile_count[0] += 1  # counted on trace; inside jit this stays 1.
      return coupling.sample(key, x0)

    x0_a = _sample_moons(jax.random.PRNGKey(1), batch=16)
    x0_b = _sample_moons(jax.random.PRNGKey(2), batch=16)
    _ = _call(key, x0_a)
    _ = _call(key, x0_b)  # same shape -> trace reused
    self.assertEqual(compile_count[0], 1)

  def test_assert_vmappable_rejects_ot(self):
    coupling = couplings.MiniBatchOTCoupling(
        source=couplings.StandardNormalSource(),
    )
    process = base.InterpolantProcess(
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
    # x_0 ~ moons.  x_1 ~ unit circle (uniform).  OT-CFM training.
    class _CircleSource:
      is_batch_level = False

      @property
      def marginal(self):
        return self

      def sample(self, key, x0):
        theta = jax.random.uniform(
            key, (x0.shape[0],),
            minval=0.0, maxval=2 * jnp.pi, dtype=jnp.float64,
        )
        return jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)

    process = base.InterpolantProcess(
        coupling=couplings.MiniBatchOTCoupling(
            source=_CircleSource(), epsilon=0.05,
        ),
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
      k_x0, k_corrupt, k_t = jax.random.split(rng, 3)
      x0 = _sample_moons(k_x0, batch=64)
      t = jax.random.uniform(k_t, (64,), dtype=jnp.float64)
      xt, target_info = process.corrupt(k_corrupt, x0, t)
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
