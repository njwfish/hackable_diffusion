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

"""Tests for the masked-pseudolikelihood sequence loss.

Three groups of tests:

1. ``ReductionAndShape``: shape contracts, masking semantics,
   ``vocab_size`` inference, ``lambda=0`` collapse to standard masked
   cross-entropy.

2. ``StrictPropriety``: empirical KL excess-risk identity from
   manuscript Proposition ``sequence-pseudolikelihood-score``.  We pick
   a tiny vocabulary and a fully tractable joint posterior, fit two
   reported laws (``P`` -- the truth, ``Q`` -- a perturbation), and
   verify
   ``E_P[S_PL(Q)] - E_P[S_PL(P)] == sum_i E[KL(P(x^i | x^{-i}) || Q(...))]``
   to floating-point precision.  This is the operational certificate
   that the loss is the correct scoring rule for the joint compatible-
   completion posterior.

3. ``EnergyTermContribution``: confirms that nonzero ``lambda`` and a
   nontrivial ``E_phi`` shift the loss, and that gradients flow into
   both the proposal logits and the energy network's parameters.
"""

import unittest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import optax

from hackable_diffusion.lib.training import compute_masked_pseudolikelihood_loss
from hackable_diffusion.lib.training import compute_masked_pseudolikelihood_nce_loss
from hackable_diffusion.lib.training import MaskedPseudolikelihoodLoss
from hackable_diffusion.lib.training import MaskedPseudolikelihoodNCELoss


def logsumexp_np(arr: np.ndarray, axis: int | None = None):
  if axis is None:
    m = float(np.max(arr))
    return m + float(np.log(np.sum(np.exp(arr - m))))
  m = np.max(arr, axis=axis, keepdims=True)
  out = m + np.log(np.sum(np.exp(arr - m), axis=axis, keepdims=True))
  return np.squeeze(out, axis=axis)


################################################################################
# MARK: Shape and reduction semantics
################################################################################


class ReductionAndShapeTest(unittest.TestCase):

  def _zero_energy(self, x0c, xt, time):
    del xt, time
    return jnp.zeros(x0c.shape[0], dtype=jnp.float64)

  def test_shape_is_per_batch(self):
    rng = jax.random.PRNGKey(0)
    B, n, V = 3, 5, 4
    logits = jax.random.normal(rng, (B, n, V), dtype=jnp.float64)
    x0 = jnp.array(
        [[0, 1, 2, 3, 0],
         [3, 2, 1, 0, 1],
         [1, 1, 1, 1, 1]],
        dtype=jnp.int32,
    )
    xt = jnp.full((B, n), V, dtype=jnp.int32)
    time = jnp.array([0.5, 0.5, 0.5], dtype=jnp.float64)
    masked = jnp.ones((B, n), dtype=jnp.bool_)

    loss = compute_masked_pseudolikelihood_loss(
        logits=logits, energy_fn=self._zero_energy,
        x0=x0, xt=xt, time=time, masked_sites=masked,
        lam=0.0, vocab_size=V,
    )
    self.assertEqual(loss.shape, (B,))
    self.assertTrue(bool(jnp.all(jnp.isfinite(loss))))

  def test_lambda_zero_matches_standard_masked_cross_entropy(self):
    """At lambda=0 the residual energy drops out and the loss equals
    the mean per-masked-site cross-entropy of ``log_softmax(logits)``."""
    rng = jax.random.PRNGKey(1)
    B, n, V = 2, 6, 5
    logits = jax.random.normal(rng, (B, n, V), dtype=jnp.float64)
    x0 = jnp.array(
        [[0, 1, 2, 3, 4, 0], [4, 0, 1, 2, 3, 1]], dtype=jnp.int32,
    )
    xt = jnp.full((B, n), V, dtype=jnp.int32)
    time = jnp.zeros((B,), dtype=jnp.float64)
    # Mask only some sites.
    masked = jnp.array(
        [[True, False, True, True, False, True],
         [False, True, True, False, True, True]],
    )

    loss = compute_masked_pseudolikelihood_loss(
        logits=logits, energy_fn=self._zero_energy,
        x0=x0, xt=xt, time=time, masked_sites=masked,
        lam=0.0, vocab_size=V,
    )

    # Reference: standard masked CE averaged per batch element.
    log_q = jax.nn.log_softmax(logits, axis=-1)
    per_site_neg_log_p = -jnp.take_along_axis(
        log_q, x0[..., None], axis=-1,
    ).squeeze(-1)
    mask_f = masked.astype(jnp.float64)
    expected = (per_site_neg_log_p * mask_f).sum(axis=-1) / mask_f.sum(axis=-1)
    self.assertTrue(jnp.allclose(loss, expected, atol=1e-12))

  def test_unmasked_sites_do_not_contribute(self):
    """Loss must be invariant under arbitrary changes to ``logits`` at
    unmasked positions (the cross-entropy at those sites is gated out)."""
    rng = jax.random.PRNGKey(2)
    B, n, V = 2, 5, 4
    logits = jax.random.normal(rng, (B, n, V), dtype=jnp.float64)
    x0 = jnp.array([[0, 1, 2, 3, 0], [3, 2, 1, 0, 2]], dtype=jnp.int32)
    xt = jnp.full((B, n), V, dtype=jnp.int32)
    time = jnp.zeros((B,), dtype=jnp.float64)
    # Mask only positions 0 and 2.
    masked = jnp.array(
        [[True, False, True, False, False],
         [True, False, True, False, False]],
    )

    base_loss = compute_masked_pseudolikelihood_loss(
        logits=logits, energy_fn=self._zero_energy,
        x0=x0, xt=xt, time=time, masked_sites=masked,
        lam=0.0, vocab_size=V,
    )

    # Perturb logits at unmasked positions only; loss must not change.
    perturbation = jnp.zeros_like(logits).at[:, 1, :].set(100.0)
    perturbation = perturbation.at[:, 3:, :].set(-50.0)
    self.assertEqual(perturbation.shape, logits.shape)
    perturbed_logits = logits + perturbation
    perturbed_loss = compute_masked_pseudolikelihood_loss(
        logits=perturbed_logits, energy_fn=self._zero_energy,
        x0=x0, xt=xt, time=time, masked_sites=masked,
        lam=0.0, vocab_size=V,
    )
    self.assertTrue(jnp.allclose(base_loss, perturbed_loss, atol=1e-12))

  def test_vocab_size_inferred_from_logits(self):
    rng = jax.random.PRNGKey(3)
    B, n, V = 2, 4, 7
    logits = jax.random.normal(rng, (B, n, V), dtype=jnp.float64)
    x0 = jax.random.randint(rng, (B, n), 0, V).astype(jnp.int32)
    xt = jnp.full((B, n), V, dtype=jnp.int32)
    time = jnp.zeros((B,), dtype=jnp.float64)
    masked = jnp.ones((B, n), dtype=jnp.bool_)

    loss_explicit = compute_masked_pseudolikelihood_loss(
        logits=logits, energy_fn=self._zero_energy,
        x0=x0, xt=xt, time=time, masked_sites=masked,
        lam=0.0, vocab_size=V,
    )
    loss_inferred = compute_masked_pseudolikelihood_loss(
        logits=logits, energy_fn=self._zero_energy,
        x0=x0, xt=xt, time=time, masked_sites=masked,
        lam=0.0,
    )
    self.assertTrue(jnp.allclose(loss_explicit, loss_inferred, atol=1e-12))

  def test_no_masked_sites_returns_zero_safely(self):
    """When no site is masked the per-batch denominator is clipped to
    ``eps``; the numerator is zero, so the loss is zero (not NaN)."""
    B, n, V = 2, 4, 3
    logits = jnp.zeros((B, n, V), dtype=jnp.float64)
    x0 = jnp.zeros((B, n), dtype=jnp.int32)
    xt = jnp.full((B, n), V, dtype=jnp.int32)
    time = jnp.zeros((B,), dtype=jnp.float64)
    masked = jnp.zeros((B, n), dtype=jnp.bool_)
    loss = compute_masked_pseudolikelihood_loss(
        logits=logits, energy_fn=self._zero_energy,
        x0=x0, xt=xt, time=time, masked_sites=masked,
        lam=0.0, vocab_size=V,
    )
    self.assertTrue(jnp.allclose(loss, jnp.zeros((B,)), atol=1e-12))


################################################################################
# MARK: Strict propriety -- KL excess-risk identity
################################################################################


class StrictProprietyTest(unittest.TestCase):
  """Empirical certificate of strict propriety from Proposition
  ``sequence-pseudolikelihood-score`` in the manuscript:

      E_{X_0 ~ P}[S_PL(Q, X_0)] - E_{X_0 ~ P}[S_PL(P, X_0)]
        = sum_{i in M} E_{P(X_0^{-i})} KL( P(X_0^i | X_0^{-i})
                                           || Q(X_0^i | X_0^{-i}) )

  We construct a tiny tractable case (``n=2`` masked sites, ``V=3``,
  full-support joint laws) and verify the identity to ~1e-10 relative
  precision.  Pure numpy here -- the analytic check has no need for
  jit/vmap and boolean indexing on traced arrays would not work.
  """

  def _enumerate_completions(self, n: int, V: int) -> np.ndarray:
    """All V^n length-n sequences over the vocabulary, lexicographic."""
    grids = np.meshgrid(*[np.arange(V)] * n, indexing="ij")
    return np.stack([g.reshape(-1) for g in grids], axis=-1)  # [V^n, n]

  def _score_pl(
      self,
      x0: np.ndarray,
      log_probs: np.ndarray,
      completions: np.ndarray,
      n: int,
  ) -> float:
    """``S_PL(Q, x0) = - sum_i log Q(x0^i | x0^{-i})`` for joint law ``Q``."""
    total = 0.0
    for i in range(n):
      others_match = np.all(
          np.where(
              np.arange(n)[None, :] == i,
              True,
              completions == x0[None, :],
          ),
          axis=-1,
      )  # [V^n] bool: True for each v of completions matching x0 at sites != i
      block = log_probs[others_match]                          # [V]
      cond = block - logsumexp_np(block)                       # log P(.|others)
      v_at_i = completions[others_match][:, i]                 # [V] = arange(V)
      # Pick the conditional log-prob of v == x0[i].
      idx = int(np.argmax(v_at_i == x0[i]))
      total -= float(cond[idx])
    return total

  def _expected_pl(
      self,
      p_log_probs: np.ndarray,
      q_log_probs: np.ndarray,
      n: int,
      V: int,
  ) -> float:
    """E_{X_0 ~ P} S_PL(Q, X_0) by direct enumeration."""
    completions = self._enumerate_completions(n, V)
    p = np.exp(p_log_probs)
    total = 0.0
    for k in range(completions.shape[0]):
      x0 = completions[k]
      total += float(p[k]) * self._score_pl(x0, q_log_probs, completions, n)
    return total

  def _expected_kl_sum(
      self,
      p_log_probs: np.ndarray,
      q_log_probs: np.ndarray,
      n: int,
      V: int,
  ) -> float:
    """sum_i E_{P(X_0^{-i})} KL( P(X_0^i | X_0^{-i}) || Q(X_0^i | X_0^{-i}) )."""
    completions = self._enumerate_completions(n, V)
    total = 0.0
    for i in range(n):
      others_grid = np.meshgrid(*[np.arange(V)] * (n - 1), indexing="ij")
      others_axes = np.stack([g.reshape(-1) for g in others_grid], axis=-1)
      for o in others_axes:
        mask = np.ones(completions.shape[0], dtype=np.bool_)
        col = 0
        for j in range(n):
          if j == i:
            continue
          mask &= completions[:, j] == int(o[col])
          col += 1
        log_p_block = p_log_probs[mask]
        log_q_block = q_log_probs[mask]
        log_p_cond = log_p_block - logsumexp_np(log_p_block)
        log_q_cond = log_q_block - logsumexp_np(log_q_block)
        p_others = float(np.exp(logsumexp_np(log_p_block)))
        kl = float(np.sum(np.exp(log_p_cond) * (log_p_cond - log_q_cond)))
        total += p_others * kl
    return total

  def test_excess_risk_equals_sum_of_one_site_kl(self):
    n, V = 2, 3
    rng = np.random.default_rng(0)
    p = rng.standard_normal(V**n)
    p = p - logsumexp_np(p)
    q = rng.standard_normal(V**n)
    q = q - logsumexp_np(q)

    excess = self._expected_pl(p, q, n, V) - self._expected_pl(p, p, n, V)
    kl_sum = self._expected_kl_sum(p, q, n, V)
    self.assertAlmostEqual(excess, kl_sum, places=10)

  def test_propriety_self_score_is_minimum(self):
    n, V = 2, 3
    rng = np.random.default_rng(1)
    p = rng.standard_normal(V**n)
    p = p - logsumexp_np(p)
    q = rng.standard_normal(V**n)
    q = q - logsumexp_np(q)
    self_score = self._expected_pl(p, p, n, V)
    other_score = self._expected_pl(p, q, n, V)
    self.assertGreater(other_score, self_score + 1e-6)


################################################################################
# MARK: Energy term contribution / gradient flow
################################################################################


class EnergyTermContributionTest(unittest.TestCase):

  def test_nonzero_lambda_changes_loss(self):
    rng = jax.random.PRNGKey(7)
    B, n, V = 2, 4, 3
    logits = jax.random.normal(rng, (B, n, V), dtype=jnp.float64)
    x0 = jnp.array([[0, 1, 2, 0], [2, 1, 0, 1]], dtype=jnp.int32)
    xt = jnp.full((B, n), V, dtype=jnp.int32)
    time = jnp.zeros((B,), dtype=jnp.float64)
    masked = jnp.ones((B, n), dtype=jnp.bool_)

    def nontrivial_energy(x0c, xt_, t_):
      del xt_, t_
      return 0.1 * jnp.sum(x0c.astype(jnp.float64) ** 2, axis=-1)

    loss_lam0 = compute_masked_pseudolikelihood_loss(
        logits=logits, energy_fn=nontrivial_energy,
        x0=x0, xt=xt, time=time, masked_sites=masked,
        lam=0.0, vocab_size=V,
    )
    loss_lam1 = compute_masked_pseudolikelihood_loss(
        logits=logits, energy_fn=nontrivial_energy,
        x0=x0, xt=xt, time=time, masked_sites=masked,
        lam=1.0, vocab_size=V,
    )
    self.assertFalse(jnp.allclose(loss_lam0, loss_lam1, atol=1e-6))

  def test_gradient_flows_to_logits_and_energy_params(self):
    """Both the tokenwise-denoiser logits and the energy network's
    closure-bound parameters should receive gradients."""
    rng = jax.random.PRNGKey(11)
    B, n, V = 2, 4, 3
    logits0 = jax.random.normal(rng, (B, n, V), dtype=jnp.float64)
    x0 = jnp.array([[0, 1, 2, 0], [2, 1, 0, 1]], dtype=jnp.int32)
    xt = jnp.full((B, n), V, dtype=jnp.int32)
    time = jnp.zeros((B,), dtype=jnp.float64)
    masked = jnp.ones((B, n), dtype=jnp.bool_)

    # Energy with one trainable parameter w.
    def loss_at(logits_in, w):
      def energy_fn(x0c, xt_, t_):
        return w * jnp.sum(x0c.astype(jnp.float64), axis=-1)
      per_sample = compute_masked_pseudolikelihood_loss(
          logits=logits_in, energy_fn=energy_fn,
          x0=x0, xt=xt, time=time, masked_sites=masked,
          lam=1.0, vocab_size=V,
      )
      return jnp.mean(per_sample)

    grads = jax.grad(loss_at, argnums=(0, 1))(logits0, 0.3)
    self.assertEqual(grads[0].shape, logits0.shape)
    # Logits gradient should be nonzero somewhere.
    self.assertTrue(bool(jnp.any(jnp.abs(grads[0]) > 1e-9)))
    # Energy-param gradient should be nonzero.
    self.assertGreater(float(jnp.abs(grads[1])), 1e-9)


################################################################################
# MARK: Dataclass wrapper
################################################################################


class MaskedPseudolikelihoodLossWrapperTest(unittest.TestCase):

  def test_wrapper_extracts_xt_and_is_corrupted(self):
    rng = jax.random.PRNGKey(0)
    B, n, V = 2, 4, 3
    logits = jax.random.normal(rng, (B, n, V), dtype=jnp.float64)

    def zero_energy(x0c, xt_, t_):
      del xt_, t_
      return jnp.zeros(x0c.shape[0], dtype=jnp.float64)

    loss_obj = MaskedPseudolikelihoodLoss(
        energy_fn=zero_energy, lam=0.0, vocab_size=V,
    )
    preds = {"logits": logits}
    targets = {
        "x0": jnp.array(
            [[0, 1, 2, 0], [2, 1, 0, 1]], dtype=jnp.int32,
        )[..., None],  # [B, n, 1] -- existing convention
        "is_corrupted": jnp.ones((B, n, 1), dtype=jnp.bool_),
        "xt": jnp.full((B, n), V, dtype=jnp.int32),
    }
    time = jnp.zeros((B,), dtype=jnp.float64)
    out = loss_obj(preds=preds, targets=targets, time=time)
    self.assertEqual(out.shape, (B,))


################################################################################
# MARK: NCE variant -- shape, baseline behaviours, gradient flow
################################################################################


class NCELossTest(unittest.TestCase):
  """Conditional-NCE pseudolikelihood for large vocabularies.

  Verifies the K+1-candidate construction, the binary logistic
  contract, and a hand-checked baseline: when ``lambda=0``, the energy
  is identically zero, the proposal is uniform over the vocabulary,
  and the bias is set to ``log(K/V)``, the corrected classifier logit
  collapses to ``log q_theta(v|xt) - log V``, so the per-site loss
  equals ``softplus(-r_pos) + sum_k softplus(r_neg)`` with
  ``r = log_softmax(logits) - log V``.
  """

  def _zero_energy(self, x0c, xt, time):
    del xt, time
    return jnp.zeros(x0c.shape[0], dtype=jnp.float64)

  def _bias_fixed(self, val: float):
    def b(x0, xt, time):
      del xt, time
      return jnp.full(x0.shape, val, dtype=jnp.float64)
    return b

  def test_shape_is_per_batch(self):
    rng = jax.random.PRNGKey(0)
    B, n, V, K = 2, 5, 4, 3
    logits = jax.random.normal(rng, (B, n, V), dtype=jnp.float64)
    x0 = jnp.array([[0, 1, 2, 3, 0], [3, 2, 1, 0, 1]], dtype=jnp.int32)
    xt = jnp.full((B, n), V, dtype=jnp.int32)
    time = jnp.zeros((B,), dtype=jnp.float64)
    masked = jnp.ones((B, n), dtype=jnp.bool_)
    proposal = jnp.broadcast_to(
        jnp.full((V,), -jnp.log(V), dtype=jnp.float64),
        (B, n, V),
    )  # uniform proposal
    negatives = jax.random.randint(rng, (B, n, K), 0, V).astype(jnp.int32)

    out = compute_masked_pseudolikelihood_nce_loss(
        logits=logits, energy_fn=self._zero_energy,
        bias_fn=self._bias_fixed(0.0),
        x0=x0, xt=xt, time=time,
        masked_sites=masked,
        proposal_log_probs=proposal,
        negatives=negatives,
        lam=1.0,
    )
    self.assertEqual(out.shape, (B,))
    self.assertTrue(bool(jnp.all(jnp.isfinite(out))))

  def test_uniform_proposal_zero_energy_baseline(self):
    """With lambda=0, zero energy, uniform proposal, and bias=0:
    r_lambda(v; c) = log_softmax(logits)[b, i, v] - log K - (-log V)
                   = log_softmax(logits)[b, i, v] + log V - log K.

    Verify the per-site loss matches the softplus formula by hand."""
    rng = jax.random.PRNGKey(11)
    B, n, V, K = 2, 4, 5, 2
    logits = jax.random.normal(rng, (B, n, V), dtype=jnp.float64)
    x0 = jnp.array([[0, 1, 2, 3], [4, 0, 1, 2]], dtype=jnp.int32)
    xt = jnp.full((B, n), V, dtype=jnp.int32)
    time = jnp.zeros((B,), dtype=jnp.float64)
    masked = jnp.ones((B, n), dtype=jnp.bool_)
    proposal = jnp.broadcast_to(
        jnp.full((V,), -jnp.log(V), dtype=jnp.float64), (B, n, V),
    )
    # Deterministic negatives so we can check the formula.
    negatives = jnp.array(
        [[[0, 1], [1, 2], [2, 3], [3, 4]],
         [[4, 3], [0, 4], [1, 0], [2, 1]]],
        dtype=jnp.int32,
    )

    out = compute_masked_pseudolikelihood_nce_loss(
        logits=logits, energy_fn=self._zero_energy,
        bias_fn=self._bias_fixed(0.0),
        x0=x0, xt=xt, time=time,
        masked_sites=masked,
        proposal_log_probs=proposal,
        negatives=negatives,
        lam=0.0,
    )

    # By hand: r_pos = log_softmax(logits)[b, i, x0[b, i]] + log V - log K
    log_q = jax.nn.log_softmax(logits, axis=-1)
    log_V = jnp.log(jnp.asarray(V, dtype=jnp.float64))
    log_K = jnp.log(jnp.asarray(K, dtype=jnp.float64))
    pos_idx = jnp.take_along_axis(log_q, x0[..., None], axis=-1).squeeze(-1)
    r_pos = pos_idx + log_V - log_K                              # [B, n]
    neg_idx = jnp.take_along_axis(log_q, negatives, axis=-1)     # [B, n, K]
    r_neg = neg_idx + log_V - log_K                              # [B, n, K]
    expected_per_site = (
        jax.nn.softplus(-r_pos)
        + jnp.sum(jax.nn.softplus(r_neg), axis=-1)
    )
    expected_per_sample = jnp.mean(expected_per_site, axis=-1)
    self.assertTrue(jnp.allclose(out, expected_per_sample, atol=1e-12))

  def test_unmasked_sites_do_not_contribute(self):
    rng = jax.random.PRNGKey(22)
    B, n, V, K = 2, 5, 4, 3
    logits = jax.random.normal(rng, (B, n, V), dtype=jnp.float64)
    x0 = jnp.array([[0, 1, 2, 3, 0], [3, 2, 1, 0, 2]], dtype=jnp.int32)
    xt = jnp.full((B, n), V, dtype=jnp.int32)
    time = jnp.zeros((B,), dtype=jnp.float64)
    masked = jnp.array(
        [[True, False, True, False, False],
         [True, False, True, False, False]],
    )
    proposal = jnp.broadcast_to(
        jnp.full((V,), -jnp.log(V), dtype=jnp.float64), (B, n, V),
    )
    negatives = jnp.array(
        [[[0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1], [0, 1, 2]],
         [[3, 2, 1], [0, 3, 2], [1, 0, 3], [2, 1, 0], [0, 1, 2]]],
        dtype=jnp.int32,
    )

    base = compute_masked_pseudolikelihood_nce_loss(
        logits=logits, energy_fn=self._zero_energy,
        bias_fn=self._bias_fixed(0.0),
        x0=x0, xt=xt, time=time,
        masked_sites=masked,
        proposal_log_probs=proposal,
        negatives=negatives,
        lam=1.0,
    )
    # Perturb logits at unmasked positions.
    perturbed = logits.at[:, 1, :].add(50.0).at[:, 3:, :].add(-100.0)
    perturbed_loss = compute_masked_pseudolikelihood_nce_loss(
        logits=perturbed, energy_fn=self._zero_energy,
        bias_fn=self._bias_fixed(0.0),
        x0=x0, xt=xt, time=time,
        masked_sites=masked,
        proposal_log_probs=proposal,
        negatives=negatives,
        lam=1.0,
    )
    self.assertTrue(jnp.allclose(base, perturbed_loss, atol=1e-10))

  def test_gradient_flows_to_logits_energy_bias(self):
    rng = jax.random.PRNGKey(33)
    B, n, V, K = 2, 4, 4, 3
    logits0 = jax.random.normal(rng, (B, n, V), dtype=jnp.float64)
    x0 = jnp.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=jnp.int32)
    xt = jnp.full((B, n), V, dtype=jnp.int32)
    time = jnp.zeros((B,), dtype=jnp.float64)
    masked = jnp.ones((B, n), dtype=jnp.bool_)
    proposal = jnp.broadcast_to(
        jnp.full((V,), -jnp.log(V), dtype=jnp.float64), (B, n, V),
    )
    negatives = jnp.array(
        [[[0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1]],
         [[3, 2, 1], [0, 3, 2], [1, 0, 3], [2, 1, 0]]],
        dtype=jnp.int32,
    )

    def loss_at(logits_in, w_e, w_b):
      def energy_fn(x0c, xt_, t_):
        return w_e * jnp.sum(x0c.astype(jnp.float64), axis=-1)
      def bias_fn(x0_, xt_, t_):
        return w_b * jnp.ones(x0_.shape, dtype=jnp.float64)
      per_sample = compute_masked_pseudolikelihood_nce_loss(
          logits=logits_in, energy_fn=energy_fn, bias_fn=bias_fn,
          x0=x0, xt=xt, time=time, masked_sites=masked,
          proposal_log_probs=proposal, negatives=negatives,
          lam=1.0,
      )
      return jnp.mean(per_sample)

    grads = jax.grad(loss_at, argnums=(0, 1, 2))(logits0, 0.1, 0.05)
    self.assertEqual(grads[0].shape, logits0.shape)
    self.assertTrue(bool(jnp.any(jnp.abs(grads[0]) > 1e-9)))
    self.assertGreater(float(jnp.abs(grads[1])), 1e-9)
    self.assertGreater(float(jnp.abs(grads[2])), 1e-9)


class MaskedPseudolikelihoodNCELossWrapperTest(unittest.TestCase):

  def test_wrapper_roundtrip(self):
    rng = jax.random.PRNGKey(0)
    B, n, V, K = 2, 4, 4, 2
    logits = jax.random.normal(rng, (B, n, V), dtype=jnp.float64)

    def zero_e(x0c, xt_, t_):
      return jnp.zeros(x0c.shape[0], dtype=jnp.float64)
    def zero_b(x0_, xt_, t_):
      return jnp.zeros(x0_.shape, dtype=jnp.float64)

    loss_obj = MaskedPseudolikelihoodNCELoss(
        energy_fn=zero_e, bias_fn=zero_b, lam=0.0,
    )
    proposal = jnp.broadcast_to(
        jnp.full((V,), -jnp.log(V), dtype=jnp.float64), (B, n, V),
    )
    negatives = jnp.array(
        [[[0, 1], [1, 2], [2, 3], [3, 0]],
         [[3, 2], [0, 3], [1, 0], [2, 1]]],
        dtype=jnp.int32,
    )
    preds = {
        "logits": logits,
        "proposal_log_probs": proposal,
        "negatives": negatives,
    }
    targets = {
        "x0": jnp.array(
            [[0, 1, 2, 3], [3, 2, 1, 0]], dtype=jnp.int32,
        )[..., None],
        "is_corrupted": jnp.ones((B, n, 1), dtype=jnp.bool_),
        "xt": jnp.full((B, n), V, dtype=jnp.int32),
    }
    time = jnp.zeros((B,), dtype=jnp.float64)
    out = loss_obj(preds=preds, targets=targets, time=time)
    self.assertEqual(out.shape, (B,))


if __name__ == "__main__":
  unittest.main()
