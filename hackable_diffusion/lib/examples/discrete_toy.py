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

"""Tractable discrete-sequence target for posterior-bridge tests.

Two examples that exhibit non-trivial joint structure -- the regime
where the manuscript's residual-energy parameterization
(eq:sequence-energy-posterior) is needed:

  - :class:`AntiCorrelatedPair`: ``V = {0, 1}``, ``n = 2``, with
    probability mass concentrated on ``(0, 1)`` and ``(1, 0)``.  The
    tokenwise marginals are uniform (a tokenwise denoiser cannot
    distinguish the four sequences), but the joint is highly
    anti-correlated.

  - :class:`ParityConstraint`: ``V = {0, 1}``, sequence length ``n``,
    with mass uniform over sequences satisfying a parity constraint
    ``sum_i x_i = 0 mod 2``.  Half the joint support is forbidden by
    the parity rule, but every individual coordinate is still uniform
    marginally.

For either target ``p_0``, the module exposes:

  - The full catalogue of sequences and their probabilities
    (``catalog`` and ``catalog_probs``).
  - A sampler ``sample(rng, n)``.
  - The factorised tokenwise marginals ``marginal_logits`` -- the
    population minimum of a strict tokenwise CE loss.
  - The optimal residual energy ``optimal_energy(x_0)`` such that
    ``\\hat p^{q, lambda E}(x_0) propto q(x_0) exp(-lambda E(x_0))``
    matches ``p_0`` at ``lambda = 1`` (with the tokenwise marginals
    above as ``q``).

Tests use these to certify (a) that the optimal residual-energy
parameterisation reproduces the joint, and (b) that the
:func:`compute_masked_pseudolikelihood_loss` evaluated at this
optimum equals the manuscript-predicted Bayes risk
``(1/n) H(X_0 | X_t)``.
"""

from __future__ import annotations

import dataclasses
import itertools

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(kw_only=True, frozen=True)
class DiscreteJointPrior:
  """A tabulated joint over ``V^n``."""

  catalog: jax.Array       # [num_sequences, n] integer
  catalog_probs: jax.Array # [num_sequences] in [0, 1], sums to 1
  vocab_size: int

  @property
  def seq_len(self) -> int:
    return int(self.catalog.shape[-1])

  @property
  def num_sequences(self) -> int:
    return int(self.catalog.shape[0])

  def sample(self, rng: jax.Array, n: int) -> jax.Array:
    """Draw ``n`` i.i.d. sequences from ``p_0``.  Shape ``[n, seq_len]``."""
    log_probs = jnp.log(jnp.clip(self.catalog_probs, 1e-30, None))
    idx = jax.random.categorical(rng, log_probs, shape=(n,))
    return self.catalog[idx]                                   # [n, seq_len]

  def marginal_logits(self) -> jax.Array:
    """Tokenwise marginals ``log P[X^i = v]`` -- the population minimum
    of the tokenwise cross-entropy loss.  Shape ``[n, V]``."""
    n, V = self.seq_len, self.vocab_size
    counts = jnp.zeros((n, V), dtype=jnp.float64)
    for i in range(n):
      for v in range(V):
        match = (self.catalog[:, i] == v).astype(jnp.float64)
        counts = counts.at[i, v].set(jnp.sum(match * self.catalog_probs))
    return jnp.log(jnp.clip(counts, 1e-30, None))              # [n, V]

  def optimal_energy(
      self, x0_candidates: jax.Array, lam: float = 1.0,
  ) -> jax.Array:
    """Per-candidate energy that pairs with the tokenwise marginals to
    reproduce ``p_0``.

    Construction (manuscript eq:sequence-energy-posterior):

        \\hat p^{q, lambda E}(x_0) propto q(x_0) exp(-lambda E(x_0))
        with q(x_0) = prod_i marginal[i, x_0^i].

    Solving for ``E(x_0)``:

        lambda E(x_0) = log q(x_0) - log p_0(x_0) + const,

    where the constant absorbs the joint normaliser (it cancels in the
    one-site conditional and in any softmax over the vocabulary).
    Returns ``E`` of shape ``[..., ]`` -- one scalar per leading-axis
    candidate.

    For sequences off ``catalog`` (probability zero under ``p_0``), we
    return a large positive energy ``log(1/eps)`` so the residual model
    assigns near-zero mass.
    """
    eps = 1e-30
    # log q(x_0) = sum_i log marginal[i, x_0^i]
    log_marginals = self.marginal_logits()                     # [n, V]
    seq_len = self.seq_len
    # Index into log_marginals at each (i, x_0^{i})
    flat_x0 = x0_candidates.reshape(-1, seq_len)               # [N, n]
    pos = jnp.arange(seq_len)[None, :]                         # [1, n]
    log_q = jnp.sum(
        log_marginals[pos, flat_x0], axis=-1,
    )                                                           # [N]

    # log p_0(x_0): table lookup against catalog.  Sequences not in
    # catalog get probability eps.
    def _table_lookup(x0):
      matches = jnp.all(self.catalog == x0[None, :], axis=-1)  # [num_sequences]
      p = jnp.sum(matches.astype(jnp.float64) * self.catalog_probs)
      return jnp.log(jnp.clip(p, eps, None))

    log_p = jax.vmap(_table_lookup)(flat_x0)                   # [N]
    energies = (log_q - log_p) / float(lam)                    # [N]
    return energies.reshape(x0_candidates.shape[:-1])


def _enumerate_binary_sequences(n: int) -> jax.Array:
  """All ``2^n`` binary sequences of length ``n``, lexicographic."""
  seqs = list(itertools.product([0, 1], repeat=n))
  return jnp.asarray(np.asarray(seqs), dtype=jnp.int32)


def AntiCorrelatedPair(epsilon: float = 0.1) -> DiscreteJointPrior:
  """Two-token, V=2 anti-correlated joint.

  ``P((0, 0)) = epsilon``,
  ``P((0, 1)) = 1/2 - epsilon``,
  ``P((1, 0)) = 1/2 - epsilon``,
  ``P((1, 1)) = epsilon``.

  Tokenwise marginals are uniform regardless of ``epsilon``: a strict
  tokenwise denoiser sees no signal.  At ``epsilon = 0`` the joint is
  exactly the anti-diagonal; at ``epsilon = 0.25`` it becomes the
  product of uniform marginals.
  """
  if not 0.0 < float(epsilon) < 0.5:
    raise ValueError(
        f"epsilon must be in (0, 0.5); got {epsilon}."
    )
  catalog = _enumerate_binary_sequences(2)                     # [4, 2]
  probs = jnp.asarray(
      [epsilon, 0.5 - epsilon, 0.5 - epsilon, epsilon],
      dtype=jnp.float64,
  )
  return DiscreteJointPrior(
      catalog=catalog, catalog_probs=probs, vocab_size=2,
  )


def ParityConstraint(seq_len: int) -> DiscreteJointPrior:
  """``V = {0, 1}``, mass uniform over sequences with even parity.

  Catalog is the ``2^{n-1}`` even-parity sequences (those whose digit
  sum is even); each has probability ``1 / 2^{n-1}``.  The tokenwise
  marginals are uniform (every coordinate is 0 or 1 with prob 1/2),
  so a tokenwise denoiser is uninformative even though the joint
  carries an exact ``n - 1``-bit constraint.  Demonstrates the
  manuscript's claim that the residual energy is necessary even when
  every individual coordinate looks uniform.
  """
  if seq_len < 2:
    raise ValueError(f"seq_len must be >= 2; got {seq_len}.")
  all_seqs = _enumerate_binary_sequences(seq_len)              # [2^n, n]
  parities = jnp.sum(all_seqs, axis=-1) % 2                    # [2^n]
  even = parities == 0
  catalog = all_seqs[even]
  num = catalog.shape[0]
  probs = jnp.full((num,), 1.0 / num, dtype=jnp.float64)
  return DiscreteJointPrior(
      catalog=catalog, catalog_probs=probs, vocab_size=2,
  )
