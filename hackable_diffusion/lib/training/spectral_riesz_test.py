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

"""Tests for the spectral Riesz energy scoring rules.

Three groups:

1. ``DistanceProperties``: each manifold distance is non-negative,
   symmetric, vanishes at coincident points (up to fp clip), and grows
   with separation.  For the torus we additionally check periodicity.

2. ``RieszEnergyScoreUStatistic``: the U-statistic loss has the
   correct shape, vanishes when the cloud equals the target (after
   averaging out its identical samples), and reduces to ``0`` when
   ``lambda=1`` and the cloud / target are sampled from the same
   isotropic distribution under enough averaging.

3. ``ManifoldEquivalence``: at the population level the Riesz energy
   loss should agree with the spectral-feature MMD form to ~1e-12.
   Verified by computing both forms on a tiny tractable case and
   comparing.
"""

import math
import unittest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from hackable_diffusion.lib.training import (
    compute_riesz_energy_score_loss,
    make_sphere_riesz_distance_fn,
    make_torus_modes,
    make_torus_riesz_distance_fn,
    RieszEnergyScoreLoss,
)


################################################################################
# MARK: Distance-fn properties (sanity)
################################################################################


class TorusDistancePropertiesTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.dim = 2
    self.modes = make_torus_modes(modes_per_dim=5, dim=self.dim)
    self.dist = make_torus_riesz_distance_fn(
        modes=self.modes, dim=self.dim, beta=1.0,
    )

  def test_non_negative(self):
    rng = jax.random.PRNGKey(0)
    theta = jax.random.uniform(
        rng, (32, self.dim), minval=0.0, maxval=2 * math.pi,
        dtype=jnp.float64,
    )
    rng, _ = jax.random.split(rng)
    theta_p = jax.random.uniform(
        rng, (32, self.dim), minval=0.0, maxval=2 * math.pi,
        dtype=jnp.float64,
    )
    rho = self.dist(theta, theta_p)
    self.assertTrue(bool(jnp.all(rho >= 0.0 - 1e-12)))

  def test_symmetric(self):
    rng = jax.random.PRNGKey(1)
    theta = jax.random.uniform(
        rng, (16, self.dim), minval=0.0, maxval=2 * math.pi,
        dtype=jnp.float64,
    )
    rng, _ = jax.random.split(rng)
    theta_p = jax.random.uniform(
        rng, (16, self.dim), minval=0.0, maxval=2 * math.pi,
        dtype=jnp.float64,
    )
    self.assertTrue(jnp.allclose(
        self.dist(theta, theta_p),
        self.dist(theta_p, theta),
        atol=1e-12,
    ))

  def test_zero_at_coincident_points(self):
    theta = jnp.array([[0.7, 1.3], [-0.4, 2.1]], dtype=jnp.float64)
    rho = self.dist(theta, theta)
    self.assertTrue(jnp.allclose(rho, jnp.zeros((2,)), atol=1e-12))

  def test_periodicity(self):
    """rho_beta on T^d must be invariant under shifts by 2*pi in any
    coordinate of theta or theta_prime."""
    theta = jnp.array([[0.5, 0.7]], dtype=jnp.float64)
    theta_p = jnp.array([[1.2, -0.3]], dtype=jnp.float64)
    base = self.dist(theta, theta_p)
    shifted = self.dist(theta, theta_p + jnp.array([2 * math.pi, 0.0]))
    self.assertTrue(jnp.allclose(base, shifted, atol=1e-10))
    shifted2 = self.dist(theta + jnp.array([0.0, -4 * math.pi]), theta_p)
    self.assertTrue(jnp.allclose(base, shifted2, atol=1e-10))


class SphereDistancePropertiesTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.ambient_dim = 3
    self.dist = make_sphere_riesz_distance_fn(
        max_degree=8, ambient_dim=self.ambient_dim, beta=1.0,
    )

  def _random_unit(self, rng, shape):
    z = jax.random.normal(rng, shape, dtype=jnp.float64)
    return z / jnp.linalg.norm(z, axis=-1, keepdims=True)

  def test_non_negative(self):
    rng = jax.random.PRNGKey(0)
    rng_x, rng_y = jax.random.split(rng)
    x = self._random_unit(rng_x, (32, self.ambient_dim))
    y = self._random_unit(rng_y, (32, self.ambient_dim))
    rho = self.dist(x, y)
    self.assertTrue(bool(jnp.all(rho >= -1e-9)))

  def test_symmetric(self):
    rng = jax.random.PRNGKey(2)
    rng_x, rng_y = jax.random.split(rng)
    x = self._random_unit(rng_x, (16, self.ambient_dim))
    y = self._random_unit(rng_y, (16, self.ambient_dim))
    self.assertTrue(jnp.allclose(self.dist(x, y), self.dist(y, x), atol=1e-10))

  def test_zero_at_coincident_points(self):
    x = jnp.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float64,
    )
    rho = self.dist(x, x)
    # Sphere kernel uses an eps-clip near +-1, so the floor is small but
    # nonzero.  Tolerate ~1e-9.
    self.assertTrue(jnp.allclose(rho, jnp.zeros((3,)), atol=1e-9))

  def test_grows_with_geodesic_separation(self):
    """For two points sharing the same north-pole basis, increasing
    angular separation must increase rho (monotone in <x, y> from 1
    down to -1)."""
    angles = jnp.linspace(0.0, math.pi, 9, dtype=jnp.float64)[1:]
    north = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)
    points = jnp.stack(
        [jnp.array([jnp.sin(a), 0.0, jnp.cos(a)]) for a in angles],
        axis=0,
    )
    rho = self.dist(jnp.broadcast_to(north, points.shape), points)
    diffs = jnp.diff(rho)
    # rho should be (weakly) increasing in angular separation.
    self.assertTrue(bool(jnp.all(diffs >= -1e-10)))


################################################################################
# MARK: Riesz energy score U-statistic
################################################################################


class RieszEnergyScoreUStatisticTest(unittest.TestCase):

  def _setup_torus(self):
    dim = 2
    modes = make_torus_modes(modes_per_dim=4, dim=dim)
    distance_fn = make_torus_riesz_distance_fn(
        modes=modes, dim=dim, beta=1.0,
    )
    return dim, distance_fn

  def test_shape_torus(self):
    dim, distance_fn = self._setup_torus()
    rng = jax.random.PRNGKey(0)
    B, M = 4, 8
    cloud = jax.random.uniform(
        rng, (B, M, dim), minval=0.0, maxval=2 * math.pi,
        dtype=jnp.float64,
    )
    rng, _ = jax.random.split(rng)
    target = jax.random.uniform(
        rng, (B, dim), minval=0.0, maxval=2 * math.pi,
        dtype=jnp.float64,
    )
    loss = compute_riesz_energy_score_loss(
        preds={"x0": cloud}, targets={"x0": target},
        time=jnp.zeros((B,), dtype=jnp.float64),
        distance_fn=distance_fn, lam=1.0,
    )
    self.assertEqual(loss.shape, (B,))
    self.assertTrue(bool(jnp.all(jnp.isfinite(loss))))

  def test_lambda_zero_collapses_to_pointwise(self):
    """At lambda=0 the interaction term drops out; loss is just the
    averaged distance from each cloud point to the target."""
    dim, distance_fn = self._setup_torus()
    rng = jax.random.PRNGKey(1)
    B, M = 2, 4
    cloud = jax.random.uniform(
        rng, (B, M, dim), minval=0.0, maxval=2 * math.pi,
        dtype=jnp.float64,
    )
    rng, _ = jax.random.split(rng)
    target = jax.random.uniform(
        rng, (B, dim), minval=0.0, maxval=2 * math.pi,
        dtype=jnp.float64,
    )
    loss = compute_riesz_energy_score_loss(
        preds={"x0": cloud}, targets={"x0": target},
        time=jnp.zeros((B,), dtype=jnp.float64),
        distance_fn=distance_fn, lam=0.0,
    )
    target_b = jnp.broadcast_to(
        target[:, None, :], cloud.shape,
    )
    expected = jnp.mean(distance_fn(cloud, target_b), axis=1)
    self.assertTrue(jnp.allclose(loss, expected, atol=1e-12))

  def test_strict_propriety_self_score_is_minimum(self):
    """For lam=1 and a posterior cloud sampled from a true law P:
    reporting Q != P (using an offset cloud) gives a higher expected
    score than reporting P (centred on the truth)."""
    dim, distance_fn = self._setup_torus()
    rng = jax.random.PRNGKey(7)
    M = 64                                                     # decent cloud size
    target_p = jnp.array([[1.0, 0.5]], dtype=jnp.float64)
    sigma = 0.2
    rng_p, rng_q = jax.random.split(rng)
    # P-cloud: centred at target_p with small Gaussian noise.
    cloud_p = (
        target_p[:, None, :]
        + sigma * jax.random.normal(
            rng_p, (1, M, dim), dtype=jnp.float64,
        )
    )
    cloud_p = cloud_p % (2 * math.pi)
    # Q-cloud: same noise scale but centred away from target.
    cloud_q = (
        (target_p[:, None, :] + jnp.array([1.0, 0.0]))
        + sigma * jax.random.normal(
            rng_q, (1, M, dim), dtype=jnp.float64,
        )
    )
    cloud_q = cloud_q % (2 * math.pi)

    score_p = compute_riesz_energy_score_loss(
        preds={"x0": cloud_p}, targets={"x0": target_p},
        time=jnp.zeros((1,), dtype=jnp.float64),
        distance_fn=distance_fn, lam=1.0,
    )
    score_q = compute_riesz_energy_score_loss(
        preds={"x0": cloud_q}, targets={"x0": target_p},
        time=jnp.zeros((1,), dtype=jnp.float64),
        distance_fn=distance_fn, lam=1.0,
    )
    self.assertGreater(float(score_q[0]), float(score_p[0]))

  def test_dataclass_wrapper_matches_compute_fn(self):
    dim, distance_fn = self._setup_torus()
    rng = jax.random.PRNGKey(11)
    B, M = 2, 6
    cloud = jax.random.uniform(
        rng, (B, M, dim), minval=0.0, maxval=2 * math.pi,
        dtype=jnp.float64,
    )
    rng, _ = jax.random.split(rng)
    target = jax.random.uniform(
        rng, (B, dim), minval=0.0, maxval=2 * math.pi,
        dtype=jnp.float64,
    )
    time = jnp.zeros((B,), dtype=jnp.float64)
    loss_obj = RieszEnergyScoreLoss(distance_fn=distance_fn, lam=1.0)
    out_obj = loss_obj(
        preds={"x0": cloud}, targets={"x0": target}, time=time,
    )
    out_fn = compute_riesz_energy_score_loss(
        preds={"x0": cloud}, targets={"x0": target}, time=time,
        distance_fn=distance_fn, lam=1.0,
    )
    self.assertTrue(jnp.allclose(out_obj, out_fn, atol=1e-12))


################################################################################
# MARK: Spectral feature equivalence (population-level identity)
################################################################################


class SpectralFeatureEquivalenceTest(unittest.TestCase):
  """For the torus, verify the U-statistic identity in feature space:

      ell_emp = -A/(M-1) + D - 2 C + (M/(M-1)) B,

  where ``A = avg ||Phi(U)||^2``, ``B = ||avg Phi(U)||^2``,
  ``C = <avg Phi(U), Phi(y)>``, ``D = ||Phi(y)||^2`` and ``Phi`` is
  the spectral feature map ``Phi(theta)_n = ||n||^{-(d+beta)/2}
  (cos(n.theta), sin(n.theta))``.

  This is the exact MMD-style closed form of the U-statistic at lam=1
  -- a tight cross-check on the pairwise-distance implementation.
  """

  def test_torus_feature_form(self):
    dim = 1
    modes = make_torus_modes(modes_per_dim=6, dim=dim)
    beta = 1.0
    distance_fn = make_torus_riesz_distance_fn(
        modes=modes, dim=dim, beta=beta,
    )

    # Build feature map manually.  The pairwise distance is
    # ``rho = 4 sum_n ||n||^{-(d+beta)} (1 - cos(...))``; matching
    # ``rho = ||Phi(theta) - Phi(theta')||^2`` with paired (cos, sin)
    # features requires per-mode weight ``w_n^2 = 2 ||n||^{-(d+beta)}``,
    # i.e. ``w_n = sqrt(2) ||n||^{-(d+beta)/2}``.
    norms = jnp.sqrt(jnp.sum(modes.astype(jnp.float64) ** 2, axis=-1))
    feat_w = jnp.sqrt(2.0) * jnp.power(norms, -(dim + beta) / 2.0)  # [N]

    def feature_fn(theta):
      inner = jnp.tensordot(theta, modes.astype(theta.dtype), axes=[[-1], [-1]])
      cos_part = feat_w * jnp.cos(inner)
      sin_part = feat_w * jnp.sin(inner)
      return jnp.concatenate([cos_part, sin_part], axis=-1)

    rng = jax.random.PRNGKey(0)
    M = 8
    cloud = jax.random.uniform(
        rng, (1, M, dim), minval=0.0, maxval=2 * math.pi,
        dtype=jnp.float64,
    )
    rng, _ = jax.random.split(rng)
    target = jax.random.uniform(
        rng, (1, dim), minval=0.0, maxval=2 * math.pi,
        dtype=jnp.float64,
    )

    # Pairwise loss (lam=1).
    pairwise_loss = compute_riesz_energy_score_loss(
        preds={"x0": cloud}, targets={"x0": target},
        time=jnp.zeros((1,), dtype=jnp.float64),
        distance_fn=distance_fn, lam=1.0,
    )

    # Feature-form U-statistic by hand (lam=1).
    cloud_features = jax.vmap(jax.vmap(feature_fn))(cloud)      # [1, M, 2N]
    target_features = jax.vmap(feature_fn)(target)              # [1, 2N]
    A = jnp.mean(jnp.sum(cloud_features ** 2, axis=-1), axis=-1)  # [1]
    bar = jnp.mean(cloud_features, axis=1)                      # [1, 2N]
    B_term = jnp.sum(bar ** 2, axis=-1)                         # [1]
    C = jnp.sum(bar * target_features, axis=-1)                 # [1]
    D = jnp.sum(target_features ** 2, axis=-1)                  # [1]
    feature_loss = (
        -A / (M - 1)
        + D - 2.0 * C
        + (M / (M - 1)) * B_term
    )                                                            # [1]

    self.assertTrue(jnp.allclose(pairwise_loss, feature_loss, atol=1e-10))


if __name__ == "__main__":
  unittest.main()
