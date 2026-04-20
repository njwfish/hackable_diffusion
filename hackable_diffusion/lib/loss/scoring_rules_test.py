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

"""Tests for the energy-score / scoring-rule loss."""

from hackable_diffusion.lib.loss import scoring_rules
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized


class EnergyScoreLossTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.bsz = 4
    self.pop = 5
    self.data_shape = (8, 8, 3)
    self.key = jax.random.PRNGKey(0)
    key_p, key_t = jax.random.split(self.key)
    self.preds = {
        "x0": jax.random.normal(
            key_p, (self.bsz, self.pop, *self.data_shape)
        )
    }
    self.targets = {
        "x0": jax.random.normal(key_t, (self.bsz, *self.data_shape))
    }
    self.time = jnp.full((self.bsz,), 0.5)

  @parameterized.named_parameters(
      ("beta_1_lam_1", 1.0, 1.0),
      ("beta_2_lam_1", 2.0, 1.0),
      ("beta_half_lam_1", 0.5, 1.0),
      ("beta_1_lam_half", 1.0, 0.5),
      ("beta_1_lam_0", 1.0, 0.0),
  )
  def test_shape_and_finite(self, beta, lam):
    loss = scoring_rules.EnergyScoreLoss(beta=beta, lam=lam)
    out = loss(self.preds, self.targets, self.time)
    self.assertEqual(out.shape, (self.bsz,))
    self.assertTrue(jnp.all(jnp.isfinite(out)))

  def test_raises_on_population_1(self):
    preds = {"x0": self.preds["x0"][:, :1]}
    loss = scoring_rules.EnergyScoreLoss()
    with self.assertRaisesRegex(ValueError, "M >= 2"):
      _ = loss(preds, self.targets, self.time)

  def test_collapsed_samples_zero_interaction(self):
    # When all M predictions are identical, the interaction term is 0 and the
    # loss reduces to ||x0 - xhat||^beta exactly.
    xhat_one = jax.random.normal(self.key, (self.bsz, 1, *self.data_shape))
    preds = {"x0": jnp.broadcast_to(xhat_one, (self.bsz, self.pop, *self.data_shape))}
    for beta in (1.0, 2.0, 0.7):
      with self.subTest(beta=beta):
        loss = scoring_rules.EnergyScoreLoss(beta=beta, lam=1.0)
        out = loss(preds, self.targets, self.time)
        diff = preds["x0"][:, 0] - self.targets["x0"]
        expected = jnp.sum(diff * diff, axis=tuple(range(1, diff.ndim)))
        if beta != 2.0:
          expected = expected ** (beta / 2.0)
        # pair distances are all zero -> interaction masked out contributes 0
        # up to eps^(beta/2) floor on off-diagonals. Use a loose atol.
        self.assertTrue(jnp.allclose(out, expected, atol=1e-4, rtol=1e-4))

  def test_permutation_invariant(self):
    # Swapping two population members should not change the loss.
    loss = scoring_rules.EnergyScoreLoss(beta=1.0, lam=1.0)
    out = loss(self.preds, self.targets, self.time)
    perm = jnp.array([2, 0, 1, 4, 3])  # arbitrary permutation of [M]
    permuted = {"x0": self.preds["x0"][:, perm]}
    out_p = loss(permuted, self.targets, self.time)
    self.assertTrue(jnp.allclose(out, out_p, atol=1e-5))

  def test_gradient_finite_on_coincident_samples(self):
    # If all M predictions equal x0, pair and data both hit the eps floor.
    # Gradient must remain finite (this is the numerical-stability contract).
    x0 = self.targets["x0"]
    preds = {"x0": jnp.broadcast_to(x0[:, None], (self.bsz, self.pop, *self.data_shape))}
    def loss_fn(p):
      return scoring_rules.compute_energy_score_loss(
          {"x0": p}, self.targets, self.time, beta=1.0, lam=1.0
      ).sum()
    grad = jax.grad(loss_fn)(preds["x0"])
    self.assertTrue(jnp.all(jnp.isfinite(grad)))

  def test_beta2_lam0_matches_manual_mse(self):
    # With beta=2, lam=0, the loss is mean over M of ||x0 - xhat_j||^2 summed
    # over feature dims — purely the data term.
    loss = scoring_rules.EnergyScoreLoss(beta=2.0, lam=0.0)
    out = loss(self.preds, self.targets, self.time)
    diff = self.preds["x0"] - self.targets["x0"][:, None]
    expected = jnp.sum(
        diff * diff, axis=tuple(range(2, diff.ndim))
    ).mean(axis=1)
    self.assertTrue(jnp.allclose(out, expected, atol=1e-5, rtol=1e-5))

  def test_1d_known_pairwise(self):
    # Construct a simple 1D case with hand-computable terms.
    # B=1, M=3, d=1. predictions [1, 2, 4], target 3.
    preds = {"x0": jnp.array([[[1.0], [2.0], [4.0]]])}           # [1, 3, 1]
    targets = {"x0": jnp.array([[3.0]])}                          # [1, 1]
    time = jnp.array([0.0])
    out = scoring_rules.compute_energy_score_loss(
        preds, targets, time, beta=1.0, lam=1.0
    )
    # data term: mean(|1-3|, |2-3|, |4-3|) = mean(2, 1, 1) = 4/3
    # pairs (|1-2|, |1-4|, |2-4|) = (1, 3, 2); sum_{j!=j'} = 2*(1+3+2) = 12
    # interaction = 12 / (2*3*2) = 1.0
    # L = 4/3 - 1.0 * 1.0 = 1/3
    self.assertAlmostEqual(float(out[0]), 1.0 / 3.0, places=5)


if __name__ == "__main__":
  absltest.main()
