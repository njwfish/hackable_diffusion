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

"""End-to-end tests on tractable discrete sequence targets.

Two groups of tests verify the manuscript's residual-sequence-energy
construction (eq:sequence-energy-posterior) and the masked
pseudolikelihood loss on closed-form discrete cases:

1. ``ResidualEnergyRecoversJointTest``: with the optimal ``E_phi``
   (analytically derived from the target joint and the tokenwise
   marginals), the induced model
   ``\\hat p^{q, lambda E}(x_0) propto q(x_0) exp(-lambda E(x_0))``
   reproduces ``p_0`` exactly.  We check on
   :class:`AntiCorrelatedPair` (where the tokenwise denoiser is
   uniform) and :class:`ParityConstraint` (where the joint has an
   exact bitwise constraint).

2. ``PseudolikelihoodLossOptimumTest``: the
   :func:`compute_masked_pseudolikelihood_loss` evaluated at the
   optimal residual-energy parameterisation, integrated against
   samples from ``p_0``, equals the per-coordinate average
   pseudo-conditional entropy ``(1/n) sum_i H(X_0^i | X_0^{-i}, X_t)``.
   This is the population-minimum of the PL scoring rule (manuscript
   Proposition ``sequence-pseudolikelihood-score`` / propriety
   identity).  Note: this is NOT the joint entropy ``H(X_0) / n``;
   the per-site cross-entropies condition on *all other* coordinates,
   not just earlier ones, so the chain-rule equality fails.
"""

from __future__ import annotations

import math
import unittest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from hackable_diffusion.lib.examples import (
    AntiCorrelatedPair,
    DiscreteJointPrior,
    ParityConstraint,
)
from hackable_diffusion.lib.loss import (
    compute_masked_pseudolikelihood_loss,
)


def _residual_joint(
    prior: DiscreteJointPrior,
    energy_fn,
    lam: float,
) -> jnp.ndarray:
  """Compute ``\\hat p^{q, lambda E}(x_0)`` over the catalogue.

  Returns a probability vector aligned with ``prior.catalog``.
  """
  log_marginals = prior.marginal_logits()                      # [n, V]
  cat = prior.catalog                                          # [N, n]
  pos = jnp.arange(prior.seq_len)[None, :]
  log_q = jnp.sum(log_marginals[pos, cat], axis=-1)            # [N]
  energies = energy_fn(cat)                                    # [N]
  log_p_hat = log_q - lam * energies                           # [N]
  log_p_hat = log_p_hat - jax.nn.logsumexp(log_p_hat)
  return jnp.exp(log_p_hat)


################################################################################
# MARK: Residual-energy parameterisation recovers the target joint
################################################################################


class ResidualEnergyRecoversJointTest(unittest.TestCase):

  def test_anti_correlated_pair_at_lambda_one(self):
    prior = AntiCorrelatedPair(epsilon=0.1)
    # We need ``energy_fn`` to take any candidate sequence -- use the
    # prior's optimal_energy directly.
    def energy_fn(x0):
      return prior.optimal_energy(x0, lam=1.0)

    recovered = _residual_joint(prior, energy_fn, lam=1.0)
    self.assertTrue(jnp.allclose(
        recovered, prior.catalog_probs, atol=1e-12,
    ))

  def test_parity_constraint_at_lambda_one(self):
    prior = ParityConstraint(seq_len=4)
    def energy_fn(x0):
      return prior.optimal_energy(x0, lam=1.0)
    recovered = _residual_joint(prior, energy_fn, lam=1.0)
    self.assertTrue(jnp.allclose(
        recovered, prior.catalog_probs, atol=1e-12,
    ))

  def test_off_catalog_assigned_negligible_mass(self):
    """For sequences not in ``catalog`` (forbidden under the parity
    constraint), the residual joint puts probability essentially 0.
    """
    prior = ParityConstraint(seq_len=4)
    # Build a candidate that's NOT in the even-parity catalog.
    odd_seq = jnp.array([1, 0, 0, 0], dtype=jnp.int32)         # parity 1
    energy_off = float(prior.optimal_energy(odd_seq[None], lam=1.0)[0])
    # Odd-parity energy should be very large (we used eps=1e-30 inside
    # optimal_energy as the floor for log p_0 of off-catalog sequences).
    self.assertGreater(energy_off, 50.0)


################################################################################
# MARK: Masked pseudolikelihood at optimal residual energy
################################################################################


def _average_pseudo_conditional_entropy(prior: DiscreteJointPrior) -> float:
  """``(1/n) sum_i H(X_0^i | X_0^{-i})`` -- the population minimum of
  the per-site masked PL loss with all sites masked.

  Computes the conditional entropy at each site by enumerating over
  values of ``x_0^{-i}`` (group catalog by other-coordinate signature),
  taking the local conditional distribution over ``X_0^i``, and
  averaging the entropy over the marginal of ``X_0^{-i}``.  Closed-form
  for fully-tabulated ``prior``.
  """
  n = prior.seq_len
  V = prior.vocab_size
  total = 0.0
  cat_np = np.asarray(prior.catalog)
  probs_np = np.asarray(prior.catalog_probs)
  for i in range(n):
    # Group sequences by the (n - 1) coordinates other than i.
    other = np.delete(cat_np, i, axis=1)                       # [N, n - 1]
    # Build a dict keyed by tuple(other).
    bucket = {}
    for s_idx, key in enumerate(map(tuple, other)):
      bucket.setdefault(key, []).append(s_idx)
    for indices in bucket.values():
      block_probs = probs_np[indices]                          # [<= V]
      total_block = float(block_probs.sum())
      if total_block <= 0.0:
        continue
      cond = block_probs / total_block
      H_block = -float(np.sum(cond * np.log(np.clip(cond, 1e-30, None))))
      total += total_block * H_block
  return float(total / n)


class PseudolikelihoodLossOptimumTest(unittest.TestCase):
  """At the population minimum (optimal residual energy), the masked
  PL loss equals the average pseudo-conditional entropy ``(1/n) sum_i
  H(X_0^i | X_0^{-i})``.  Smaller setup so we can enumerate everything.
  """

  def test_anti_correlated_pl_loss_equals_conditional_entropy(self):
    prior = AntiCorrelatedPair(epsilon=0.1)
    n, V = prior.seq_len, prior.vocab_size

    # Build "all-masked" x_t (the worst case where the loss is the
    # full sequence entropy / n).  Use mask token = V (out-of-vocab,
    # the convention used by tests / production code).
    mask_token = V
    B = prior.num_sequences
    x0_batch = prior.catalog                                   # [B, n]
    xt_batch = jnp.full((B, n), mask_token, dtype=jnp.int32)
    # All sites masked.
    masked_sites = jnp.ones((B, n), dtype=jnp.bool_)
    time = jnp.zeros((B,), dtype=jnp.float64)

    # Tokenwise logits = log marginals; broadcast to [B, n, V].
    log_marg = prior.marginal_logits()                         # [n, V]
    logits = jnp.broadcast_to(
        log_marg[None, :, :], (B, n, V),
    )

    def energy_fn(x0_candidate, xt_, t_):
      del xt_, t_
      return prior.optimal_energy(x0_candidate, lam=1.0)

    per_sample_loss = compute_masked_pseudolikelihood_loss(
        logits=logits, energy_fn=energy_fn,
        x0=x0_batch, xt=xt_batch, time=time,
        masked_sites=masked_sites,
        lam=1.0, vocab_size=V,
    )                                                          # [B]
    # Average over the B sequences weighted by p_0.
    population_loss = float(jnp.sum(per_sample_loss * prior.catalog_probs))

    expected = _average_pseudo_conditional_entropy(prior)
    self.assertAlmostEqual(
        population_loss, expected, delta=1e-6,
    )

  def test_parity_constraint_pl_loss_equals_conditional_entropy(self):
    prior = ParityConstraint(seq_len=3)
    n, V = prior.seq_len, prior.vocab_size

    mask_token = V
    B = prior.num_sequences
    x0_batch = prior.catalog                                   # [B, n]
    xt_batch = jnp.full((B, n), mask_token, dtype=jnp.int32)
    masked_sites = jnp.ones((B, n), dtype=jnp.bool_)
    time = jnp.zeros((B,), dtype=jnp.float64)

    log_marg = prior.marginal_logits()
    logits = jnp.broadcast_to(log_marg[None, :, :], (B, n, V))

    def energy_fn(x0_candidate, xt_, t_):
      del xt_, t_
      return prior.optimal_energy(x0_candidate, lam=1.0)

    per_sample_loss = compute_masked_pseudolikelihood_loss(
        logits=logits, energy_fn=energy_fn,
        x0=x0_batch, xt=xt_batch, time=time,
        masked_sites=masked_sites,
        lam=1.0, vocab_size=V,
    )
    population_loss = float(jnp.sum(per_sample_loss * prior.catalog_probs))

    expected = _average_pseudo_conditional_entropy(prior)
    self.assertAlmostEqual(population_loss, expected, delta=1e-6)
    # For the parity constraint, knowing X^{-i} determines X^i exactly,
    # so each conditional entropy is 0.
    self.assertAlmostEqual(expected, 0.0, delta=1e-12)


################################################################################
# MARK: Tokenwise denoiser is insufficient -- gives wrong joint
################################################################################


class TokenwiseDenoiserIsInsufficientTest(unittest.TestCase):
  """Demonstrates the manuscript's motivation: a strict tokenwise
  denoiser (``lambda = 0``, no residual energy) cannot represent
  joint structure.  For a target with uniform tokenwise marginals but
  non-trivial joint correlations, the tokenwise model puts uniform
  mass over ``V^n`` -- not the target ``p_0``.
  """

  def test_anti_correlated_at_lambda_zero_is_uniform_joint(self):
    prior = AntiCorrelatedPair(epsilon=0.1)
    def zero_energy(x0):
      return jnp.zeros(x0.shape[:-1], dtype=jnp.float64)
    joint_at_lam0 = _residual_joint(prior, zero_energy, lam=1.0)
    # Tokenwise marginals are uniform -> log_q is identical for all
    # sequences -> joint is uniform over the 4 sequences.
    self.assertTrue(jnp.allclose(
        joint_at_lam0,
        jnp.full((4,), 0.25, dtype=jnp.float64),
        atol=1e-12,
    ))
    # And this is FAR from the target (anti-correlated).
    target = prior.catalog_probs
    discrepancy = float(jnp.sum(jnp.abs(joint_at_lam0 - target)))
    # Anti-correlated mass: target=[0.1, 0.4, 0.4, 0.1]; uniform=0.25.
    # L1 = |0.1 - 0.25| + 2*|0.4 - 0.25| + |0.1 - 0.25| = 0.6.
    self.assertAlmostEqual(discrepancy, 0.6, delta=1e-10)


if __name__ == "__main__":
  unittest.main()
