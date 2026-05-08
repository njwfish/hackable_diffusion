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

"""Tests for refinement-theory diagnostics."""

from __future__ import annotations

import math
import unittest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from hackable_diffusion.lib import posterior_bridges_diagnostics as diag
from hackable_diffusion.lib.corruption import schedules


################################################################################
# MARK: Bridge endpoint Lipschitz constants
################################################################################


class GaussianEndpointLipschitzTest(unittest.TestCase):

  def test_zero_at_coincident_times(self):
    schedule = schedules.CosineSchedule()
    t = jnp.asarray(0.5, dtype=jnp.float64)
    self.assertTrue(jnp.allclose(
        diag.vp_gaussian_endpoint_lipschitz(schedule, t, t),
        0.0,
        atol=1e-12,
    ))

  def test_lies_in_unit_interval(self):
    schedule = schedules.CosineSchedule()
    grid = jnp.linspace(0.05, 0.95, 19, dtype=jnp.float64)
    for s in grid:
      for t in grid:
        if float(s) >= float(t):
          continue
        beta = diag.vp_gaussian_endpoint_lipschitz(schedule, s, t)
        self.assertGreaterEqual(float(beta), 0.0)
        self.assertLessEqual(float(beta), 1.0 + 1e-9)

  def test_matches_closed_form(self):
    schedule = schedules.CosineSchedule()
    s, t = 0.3, 0.6
    alpha_s = float(diag._alpha_at(schedule, s))
    alpha_t = float(diag._alpha_at(schedule, t))
    expected = (alpha_s ** 2 - alpha_t ** 2) / (
        alpha_s * (1.0 - alpha_t ** 2)
    )
    out = float(diag.vp_gaussian_endpoint_lipschitz(schedule, s, t))
    self.assertAlmostEqual(out, expected, places=10)

  def test_increases_as_s_decreases_at_fixed_t(self):
    """``s -> 0`` -> the bridge step almost reveals the endpoint, so
    beta_{s, t} -> 1.  At ``s -> t`` it -> 0."""
    schedule = schedules.CosineSchedule()
    t = 0.7
    s_close = 0.69
    s_far = 0.05
    beta_close = float(diag.vp_gaussian_endpoint_lipschitz(schedule, s_close, t))
    beta_far = float(diag.vp_gaussian_endpoint_lipschitz(schedule, s_far, t))
    self.assertLess(beta_close, beta_far)


class MaskedEndpointLipschitzTest(unittest.TestCase):

  def test_zero_at_coincident_times(self):
    schedule = schedules.CosineDiscreteSchedule()
    t = jnp.asarray(0.5, dtype=jnp.float64)
    self.assertTrue(jnp.allclose(
        diag.masked_endpoint_lipschitz(schedule, t, t), 0.0, atol=1e-12,
    ))

  def test_in_unit_interval_and_monotone(self):
    schedule = schedules.CosineDiscreteSchedule()
    t = 0.7
    s_close, s_far = 0.69, 0.05
    self.assertLess(
        float(diag.masked_endpoint_lipschitz(schedule, s_close, t)),
        float(diag.masked_endpoint_lipschitz(schedule, s_far, t)),
    )

  def test_matches_closed_form(self):
    schedule = schedules.CosineDiscreteSchedule()
    s, t = 0.2, 0.8
    alpha_s = float(diag._alpha_at(schedule, s))
    alpha_t = float(diag._alpha_at(schedule, t))
    expected = (alpha_s - alpha_t) / (1.0 - alpha_t)
    out = float(diag.masked_endpoint_lipschitz(schedule, s, t))
    self.assertAlmostEqual(out, expected, places=10)


################################################################################
# MARK: Bayes-risk bounds
################################################################################


class BayesRiskBoundsTest(unittest.TestCase):

  def test_gaussian_mse_bound_matches_sigma_over_alpha(self):
    schedule = schedules.CosineSchedule()
    dim = 4
    grid = jnp.linspace(0.1, 0.9, 9, dtype=jnp.float64)
    for t in grid:
      bound = float(
          diag.gaussian_endpoint_mse_bayes_risk_bound(schedule, t, dim)
      )
      alpha_t = float(diag._alpha_at(schedule, t))
      sigma_sq = max(1.0 - alpha_t ** 2, 0.0)
      expected = (sigma_sq / alpha_t ** 2) * dim
      self.assertAlmostEqual(bound, expected, places=10)

  def test_gaussian_mse_bound_increases_toward_noise(self):
    schedule = schedules.CosineSchedule()
    early = float(
        diag.gaussian_endpoint_mse_bayes_risk_bound(schedule, 0.1, 1)
    )
    late = float(
        diag.gaussian_endpoint_mse_bayes_risk_bound(schedule, 0.9, 1)
    )
    self.assertLess(early, late)

  def test_masked_entropy_bound_matches_one_minus_alpha_log_V(self):
    schedule = schedules.CosineDiscreteSchedule()
    V = 32
    grid = jnp.linspace(0.0, 1.0, 11, dtype=jnp.float64)
    for t in grid:
      bound = float(
          diag.masked_endpoint_entropy_bayes_risk_bound(schedule, t, V)
      )
      alpha_t = float(diag._alpha_at(schedule, t))
      expected = (1.0 - alpha_t) * math.log(V)
      self.assertAlmostEqual(bound, expected, places=10)

  def test_masked_entropy_bound_endpoints(self):
    """Bound is 0 at clean time, ``log V`` at fully-masked time."""
    schedule = schedules.CosineDiscreteSchedule()
    V = 8
    bound_zero = float(
        diag.masked_endpoint_entropy_bayes_risk_bound(schedule, 0.0, V)
    )
    bound_one = float(
        diag.masked_endpoint_entropy_bayes_risk_bound(schedule, 1.0, V)
    )
    self.assertAlmostEqual(bound_zero, 0.0, places=8)
    self.assertAlmostEqual(bound_one, math.log(V), places=8)


################################################################################
# MARK: Iterative-refinement recursion
################################################################################


class RefinementRecursionTest(unittest.TestCase):

  def test_zero_local_error_propagates_only_through_kappa(self):
    """If ``delta_k = 0`` at every step, the recursion is purely the
    bridge-propagation contraction ``D_{k-1} = kappa_k D_k``.  Verify
    on a constant ``kappa = 0.5`` over 4 steps starting from
    ``D_init = 1``."""
    deltas = jnp.zeros(4, dtype=jnp.float64)
    betas = jnp.zeros(4, dtype=jnp.float64)
    kappas = jnp.full(4, 0.5, dtype=jnp.float64)
    Ls = jnp.ones(4, dtype=jnp.float64)
    D_seq, E_seq = diag.simulate_refinement_recursion(
        deltas=deltas, betas=betas, kappas=kappas, Ls=Ls, D_init=1.0,
    )
    expected_D = jnp.asarray([1.0, 0.5, 0.25, 0.125, 0.0625])
    self.assertTrue(jnp.allclose(D_seq, expected_D, atol=1e-12))
    # E_k = delta_k + L_k D_k = D_k since delta=0, L=1.
    self.assertTrue(jnp.allclose(E_seq, expected_D[:-1], atol=1e-12))

  def test_zero_kappa_with_constant_delta(self):
    """If ``kappa = 0`` (perfect bridge contraction at the noisy end),
    the next ``D_{k-1} = beta_k delta_k`` -- only the local error
    survives."""
    deltas = jnp.full(3, 0.1, dtype=jnp.float64)
    betas = jnp.full(3, 0.4, dtype=jnp.float64)
    kappas = jnp.zeros(3, dtype=jnp.float64)
    Ls = jnp.full(3, 2.0, dtype=jnp.float64)
    D_seq, E_seq = diag.simulate_refinement_recursion(
        deltas=deltas, betas=betas, kappas=kappas, Ls=Ls, D_init=1.0,
    )
    # E_0 = delta_0 + L_0 D_init = 0.1 + 2.0 * 1.0 = 2.1.
    # Then D_after_step0 = beta_0 * delta_0 + 0 * D_init = 0.04.
    self.assertAlmostEqual(float(E_seq[0]), 2.1, places=10)
    self.assertAlmostEqual(float(D_seq[1]), 0.04, places=10)
    # Subsequent E_k = 0.1 + 2 * D_k.
    self.assertAlmostEqual(float(E_seq[1]), 0.1 + 2.0 * 0.04, places=10)

  def test_recursion_matches_hand_unrolled_loop(self):
    rng = np.random.default_rng(0)
    N = 5
    deltas = jnp.asarray(rng.uniform(0.0, 0.2, N), dtype=jnp.float64)
    betas = jnp.asarray(rng.uniform(0.1, 0.5, N), dtype=jnp.float64)
    kappas = jnp.asarray(rng.uniform(0.3, 0.7, N), dtype=jnp.float64)
    Ls = jnp.asarray(rng.uniform(0.5, 1.5, N), dtype=jnp.float64)
    D_init = 0.7

    D_seq, E_seq = diag.simulate_refinement_recursion(
        deltas=deltas, betas=betas, kappas=kappas, Ls=Ls, D_init=D_init,
    )

    # Hand-unroll.
    expected_D = [D_init]
    expected_E = []
    D_k = D_init
    for k in range(N):
      E_k = float(deltas[k] + Ls[k] * D_k)
      D_k = float(betas[k] * deltas[k] + kappas[k] * D_k)
      expected_E.append(E_k)
      expected_D.append(D_k)

    self.assertTrue(jnp.allclose(
        D_seq, jnp.asarray(expected_D), atol=1e-12,
    ))
    self.assertTrue(jnp.allclose(
        E_seq, jnp.asarray(expected_E), atol=1e-12,
    ))


if __name__ == "__main__":
  unittest.main()
