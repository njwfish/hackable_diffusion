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

"""M3 integration test: data-to-data flow matching with ``DeterministicCoupling``.

Trains a tiny MLP velocity predictor on a 2D ``x_0 -> x_1`` task where
``x_1 = rotate(x_0, pi/4)``.  The training loop uses ``VelocityOnlyTargets``
-- no Gaussian-source identities -- exercising the generic
``InterpolantProcess(DeterministicCoupling, LinearInterpolant,
VelocityOnlyTargets)`` composition end-to-end.

Acceptance: loss decreases from its random-init value by a factor of 10+
over 200 training steps.
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


def _rotate_2d(x: jax.Array, angle: float) -> jax.Array:
  c, s = jnp.cos(angle), jnp.sin(angle)
  rot = jnp.asarray([[c, -s], [s, c]], dtype=x.dtype)
  return x @ rot.T


def _tiny_mlp_init(key, in_dim: int, hidden: int = 64):
  k1, k2, k3 = jax.random.split(key, 3)
  return {
      "w1": jax.random.normal(k1, (in_dim + 1, hidden), dtype=jnp.float64) * 0.3,
      "w2": jax.random.normal(k2, (hidden, hidden), dtype=jnp.float64) * 0.3,
      "w3": jax.random.normal(k3, (hidden, in_dim), dtype=jnp.float64) * 0.3,
  }


def _tiny_mlp_apply(params, xt, time):
  # Concatenate t as an extra feature.
  t = jnp.broadcast_to(time.reshape(-1, 1), (xt.shape[0], 1))
  h = jnp.concatenate([xt, t], axis=-1)
  h = jax.nn.gelu(h @ params["w1"])
  h = jax.nn.gelu(h @ params["w2"])
  return h @ params["w3"]  # velocity prediction


class DeterministicCouplingFlowMatchingTest(unittest.TestCase):

  def test_rotation_flow_matching_loss_decreases(self):
    # Data: x_0 from a 2D Gaussian mixture; x_1 = rotate(x_0, pi/4).
    angle = float(jnp.pi / 4)

    def sample_x0(key, batch):
      return jax.random.normal(key, (batch, 2), dtype=jnp.float64)

    # Build the data-to-data corruption process:
    # deterministic coupling + linear interpolant + velocity-only targets.
    process = base.InterpolantProcess(
        coupling=couplings.DeterministicCoupling(
            map_fn=lambda x0: _rotate_2d(x0, angle),
        ),
        interpolant=interpolants.LinearInterpolant(
            schedule=schedules.RFSchedule(),
        ),
        targets=targets.VelocityOnlyTargets(),
    )

    rng = jax.random.PRNGKey(0)
    init_rng, rng = jax.random.split(rng)
    params = _tiny_mlp_init(init_rng, in_dim=2, hidden=32)
    optimizer = optax.adam(learning_rate=3e-3)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, rng):
      k_data, k_corrupt, k_time = jax.random.split(rng, 3)
      x0 = sample_x0(k_data, batch=64)
      time = jax.random.uniform(k_time, (64,), dtype=jnp.float64)
      xt, target_info = process.corrupt(k_corrupt, x0, time)
      def loss_fn(p):
        pred_v = _tiny_mlp_apply(p, xt, time)
        return jnp.mean((pred_v - target_info["velocity"]) ** 2)
      loss, grads = jax.value_and_grad(loss_fn)(params)
      updates, opt_state = optimizer.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      return params, opt_state, loss

    # Warm-up loss.
    rng, step_rng = jax.random.split(rng)
    _, _, loss_init = step(params, opt_state, step_rng)
    losses = []
    for i in range(200):
      rng, step_rng = jax.random.split(rng)
      params, opt_state, loss = step(params, opt_state, step_rng)
      losses.append(float(loss))

    loss_first_10 = np.mean(losses[:10])
    loss_last_10 = np.mean(losses[-10:])
    self.assertLess(
        loss_last_10, loss_first_10 / 10.0,
        msg=(
            f"DeterministicCoupling flow-matching did not converge: "
            f"init-avg={loss_first_10:.4f}, final-avg={loss_last_10:.4f}"
        ),
    )

  def test_marginal_is_none_for_deterministic_coupling(self):
    # DeterministicCoupling has no well-defined x_1 marginal.
    coupling = couplings.DeterministicCoupling(map_fn=lambda x: x + 1)
    self.assertIsNone(coupling.marginal)

    process = base.InterpolantProcess(
        coupling=coupling,
        interpolant=interpolants.LinearInterpolant(schedule=schedules.RFSchedule()),
        targets=targets.VelocityOnlyTargets(),
    )
    with self.assertRaises(ValueError):
      process.sample_from_invariant(
          jax.random.PRNGKey(0), jnp.zeros((4, 2), dtype=jnp.float64),
      )


if __name__ == "__main__":
  unittest.main()
