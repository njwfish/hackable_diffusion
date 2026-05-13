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

"""Bitwise parity between the shim ``GaussianProcess`` / ``RiemannianProcess``
and the pinned legacy implementations.

Covers (post Prior/Coupling split):

- ``convert_predictions`` across all 5 ``x0 / epsilon / score / velocity / v``
  source types -- pure math, byte-parity holds.
- ``sample_from_invariant`` for both Gaussian and Riemannian -- the
  legacy ``coupling.marginal.sample(key, ...)`` and the new
  ``prior.sample(key, ...)`` consume the same key, so byte-parity holds.
- ``get_schedule_info`` for both.

``corrupt`` byte-parity is *not* tested: the Prior/Coupling refactor
re-splits the per-step key into three slots (prior, coupling, z) for
uniformity, so the sampled ``x_1`` differs from the legacy unsplit
flow even when the math is unchanged.  Correctness is covered by
``gaussian_test.py`` / ``riemannian_test.py`` / the per-shim end-to-end
flow-matching tests.
"""

from __future__ import annotations

import itertools
import unittest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from hackable_diffusion.lib import manifolds
from hackable_diffusion.lib.corruption import _gaussian_legacy
from hackable_diffusion.lib.corruption import _riemannian_legacy
from hackable_diffusion.lib.corruption import schedules as _schedules
from hackable_diffusion.lib.corruption.gaussian import GaussianProcess
from hackable_diffusion.lib.corruption.riemannian import RiemannianProcess


_GAUSSIAN_SCHEDULES = [
    ("RFSchedule", lambda: _schedules.RFSchedule()),
    ("CosineSchedule", lambda: _schedules.CosineSchedule()),
    ("InverseCosineSchedule", lambda: _schedules.InverseCosineSchedule()),
    ("LinearDiffusionSchedule", lambda: _schedules.LinearDiffusionSchedule()),
    ("GeometricSchedule",
     lambda: _schedules.GeometricSchedule(sigma_min=1e-3, sigma_max=10.0)),
]
_GAUSSIAN_TIMES = [0.01, 0.25, 0.5, 0.75, 0.99]


def _assert_tree_equal(a, b, msg=""):
  """Bitwise-identical tree equality check."""
  flat_a, tree_a = jax.tree.flatten(a)
  flat_b, tree_b = jax.tree.flatten(b)
  assert tree_a == tree_b, f"tree structure differs: {tree_a} vs {tree_b}. {msg}"
  for i, (xa, xb) in enumerate(zip(flat_a, flat_b)):
    if isinstance(xa, jax.Array):
      if not bool(jnp.array_equal(xa, xb)):
        raise AssertionError(f"leaf {i} differs. {msg}")
    else:
      assert xa == xb, f"leaf {i} differs: {xa} vs {xb}. {msg}"


class GaussianParityTest(unittest.TestCase):
  """Shim ``GaussianProcess`` matches the legacy class on everything
  that doesn't depend on the per-step key split (i.e. everything except
  ``corrupt``)."""

  def test_convert_predictions(self):
    for (name, make_sched), t in itertools.product(
        _GAUSSIAN_SCHEDULES, _GAUSSIAN_TIMES,
    ):
      schedule = make_sched()
      old = _gaussian_legacy._LegacyGaussianProcess(schedule=schedule)
      new = GaussianProcess(schedule=schedule)
      rng = jax.random.PRNGKey(123)
      xt = jax.random.normal(rng, (4, 8), dtype=jnp.float64)
      t_arr = jnp.full((4,), t, dtype=jnp.float64)
      # Sample a fake prediction for each parameterisation.
      value = jax.random.normal(
          jax.random.fold_in(rng, 7), (4, 8), dtype=jnp.float64,
      )
      # Legacy used ``epsilon`` as both a source and target parameterisation
      # name; the rename moves them all to ``x1``.  Iterate the new names
      # for the new process, and map legacy-side comparisons by the rename.
      _old_to_new_name = {"epsilon": "x1"}
      _new_to_old_name = {"x1": "epsilon"}
      for new_source in ("x0", "x1", "score", "velocity", "v"):
        old_source = _new_to_old_name.get(new_source, new_source)
        pred_old = {old_source: value}
        pred_new = {new_source: value}
        old_out = old.convert_predictions(pred_old, xt, t_arr)
        new_out = new.convert_predictions(pred_new, xt, t_arr)
        # Normalise keyset: rename ``epsilon`` -> ``x1`` on the old side.
        old_out_renamed = {
            _old_to_new_name.get(k, k): v for k, v in old_out.items()
        }
        self.assertEqual(
            set(old_out_renamed), set(new_out),
            f"[{name}, from={new_source}] conversion keyset",
        )
        for k in old_out_renamed:
          _assert_tree_equal(
              old_out_renamed[k], new_out[k],
              f"[{name}, from={new_source}, to={k}, t={t}] conversion",
          )

  def test_sample_from_invariant(self):
    for name, make_sched in _GAUSSIAN_SCHEDULES:
      schedule = make_sched()
      old = _gaussian_legacy._LegacyGaussianProcess(schedule=schedule)
      new = GaussianProcess(schedule=schedule)
      key = jax.random.PRNGKey(42)
      data_spec = jnp.zeros((8, 16), dtype=jnp.float64)
      _assert_tree_equal(
          old.sample_from_invariant(key, data_spec),
          new.sample_from_invariant(key, data_spec),
          f"[{name}] sample_from_invariant",
      )

  def test_get_schedule_info(self):
    for (name, make_sched), t in itertools.product(
        _GAUSSIAN_SCHEDULES, _GAUSSIAN_TIMES,
    ):
      schedule = make_sched()
      old = _gaussian_legacy._LegacyGaussianProcess(schedule=schedule)
      new = GaussianProcess(schedule=schedule)
      t_arr = jnp.full((4,), t, dtype=jnp.float64)
      _assert_tree_equal(
          old.get_schedule_info(t_arr),
          new.get_schedule_info(t_arr),
          f"[{name}, t={t}] get_schedule_info",
      )


class RiemannianParityTest(unittest.TestCase):
  """Shim ``RiemannianProcess`` is byte-identical to the legacy class."""

  def _make_pieces(self, schedule_name):
    schedule = getattr(_schedules, schedule_name)()
    manifold = manifolds.Sphere()
    old = _riemannian_legacy._LegacyRiemannianProcess(
        manifold=manifold, schedule=schedule,
    )
    new = RiemannianProcess(manifold=manifold, schedule=schedule)
    return old, new

  def _sphere_points(self, key, batch):
    manifold = manifolds.Sphere()
    return manifold.random_uniform(key, shape=(batch, 3))

  def test_sample_from_invariant(self):
    for schedule_name in ["RiemannianCosineSchedule", "LinearRiemannianSchedule"]:
      try:
        old, new = self._make_pieces(schedule_name)
      except AttributeError:
        continue
      key = jax.random.PRNGKey(0)
      data_spec = jnp.zeros((8, 3), dtype=jnp.float64)
      _assert_tree_equal(
          old.sample_from_invariant(key, data_spec),
          new.sample_from_invariant(key, data_spec),
          f"[{schedule_name}] sample_from_invariant",
      )


if __name__ == "__main__":
  unittest.main()
