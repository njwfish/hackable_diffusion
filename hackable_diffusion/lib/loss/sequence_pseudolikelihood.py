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

"""Masked pseudolikelihood loss for sequence-level posterior learning.

Implements the strictly proper conditional pseudolikelihood scoring rule
from the posterior-bridges manuscript, Section "Sequence-Level Scoring
Rules".  The reported posterior on compatible clean completions is

    \\hat p_{0|t}^{theta, phi, lambda}(x_0 | x_t)
        propto q_theta(x_0 | x_t) * exp{-lambda * E_phi(x_0, x_t, t)},

where ``q_theta`` is a positive *factorized* tokenwise denoiser (the
existing tokenwise discrete-diffusion model) and ``E_phi`` is a *residual
sequence energy* that adds joint structure on top of the factorized
proposal.  The induced one-site conditional only normalizes over the
vocabulary at the selected site:

    \\hat p_{0|t}^{...}(x^i = v | x^{-i}, x_t)
        = softmax over v in V of
              [ log q_theta^i(v | x_t)
                - lambda * E_phi(x^{i <- v}, x_t, t) ].

Crucially, no sequence-level normalizer over Omega(x_t) is estimated:
each training term is an ordinary cross-entropy against the
energy-adjusted vocabulary logits at one masked site.  The exact extra
cost over tokenwise discrete diffusion is ``sum_b |I_b| * |V|`` scalar
energy evaluations per minibatch -- tractable for small/medium
vocabularies (DNA: 4, proteins: ~20, character-level: ~64-128).

Strict propriety (manuscript Proposition sequence-pseudolikelihood-score):
on laws with full support over Omega(x_t), the score is strictly proper
for the joint compatible-completion posterior, with excess risk equal to
the sum of expected one-site KL divergences.  At lambda=0 the loss
reduces to the standard tokenwise masked cross-entropy
(``compute_discrete_diffusion_loss`` in ``lib/loss/discrete.py``); for
lambda>0 a sufficiently rich ``E_phi`` can represent any positive joint
posterior on Omega(x_t).

Cost structure.  For batch size ``B``, sequence length ``n``, vocabulary
size ``V``, the loss materializes a candidate tensor of shape
``[B, n, V, n]`` (one-site replacements x_0^{b, i <- v}) and calls the
energy network on its flattened first three axes -- ``B * n * V``
sequence evaluations of length ``n``.  Energies at non-masked sites are
discarded by the loss mask, so the exact wasted compute is
``(n - n_masked) * V * B`` evaluations; for masked discrete diffusion
``n_masked`` grows with ``t``, so the wasted fraction is small at high
``t`` (where the loss matters most) and larger at low ``t`` (small loss
contribution anyway).

This is the *exact* small-vocabulary path.  Conditional NCE / sampled
softmax variants for large vocabularies are TODO -- they have the same
adjusted-logit formula at the positive site but estimate the local
log-normalizer with negative samples.
"""

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp
import optax
import kauldron.ktyping as kt

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.loss import base


################################################################################
# MARK: Type Aliases
################################################################################

DataArray = hd_typing.DataArray
LossOutput = hd_typing.LossOutput
TargetInfo = hd_typing.TargetInfo
TimeArray = hd_typing.TimeArray

DiscreteSchedule = schedules.DiscreteSchedule


# Signature of the residual sequence energy.  ``x0_candidate`` is an
# integer-valued candidate sequence ``[..., n]``; ``xt`` is the noisy
# state ``[..., n]``; ``time`` is a time array ``[..., ...]`` whose
# leading axes match the leading axes of the candidate.  Returns a
# scalar per leading-axis element, shape ``[...]``.
EnergySequenceFn = Callable[[DataArray, DataArray, TimeArray], jax.Array]


# Signature of the per-site bias term ``b_psi(c)`` for the conditional-
# NCE loss.  Takes the observed clean sequence and the noisy state and
# emits one bias per site, shape ``[B, n]``.  In the manuscript the
# context ``c = (i, x_0^{-i}, x_t, t)`` includes the rest of the clean
# sequence, but the practical implementation feeds in the full ``x_0``
# and lets the bias network attend over it.
BiasSiteFn = Callable[[DataArray, DataArray, TimeArray], jax.Array]


################################################################################
# MARK: Candidate-tensor construction
################################################################################


def _build_one_site_candidates(
    x0: jax.Array,         # [B, n], integer
    vocab_size: int,
) -> jax.Array:
  """Build all one-site replacements of ``x0`` over the vocabulary.

  Returns a tensor of shape ``[B, n, V, n]`` whose ``[b, i, v, :]`` slice
  is ``x0[b, :]`` with position ``i`` replaced by ``v``.  Materializing
  this tensor inside ``jit`` is the dominant memory cost of the loss --
  callers that need to scale beyond ``B * n * V * n``-element memory
  should chunk along ``B`` or along the masked-site axis.
  """
  B, n = x0.shape
  V = vocab_size
  # position_mask[i, j] = (i == j) of shape [n, n].
  position_mask = jnp.eye(n, dtype=jnp.bool_)               # [n, n]
  # Replace value[v] = v of shape [V].
  replace_value = jnp.arange(V, dtype=x0.dtype)             # [V]
  # Broadcasting:
  #   x0[:, None, None, :]                           [B, 1, 1, n]
  #   position_mask[None, :, None, :]                [1, n, 1, n]
  #   replace_value[None, None, :, None]             [1, 1, V, 1]
  return jnp.where(
      position_mask[None, :, None, :],
      replace_value[None, None, :, None],
      x0[:, None, None, :],
  )                                                          # [B, n, V, n]


################################################################################
# MARK: General loss function
################################################################################


@kt.typechecked
def compute_masked_pseudolikelihood_loss(
    *,
    logits: jax.Array,
    energy_fn: EnergySequenceFn,
    x0: jax.Array,
    xt: jax.Array,
    time: TimeArray,
    masked_sites: jax.Array,
    lam: float = 1.0,
    vocab_size: int | None = None,
    eps: float = 1e-12,
    schedule: DiscreteSchedule | None = None,
    weight_fn: base.WeightFn | None = None,
) -> jax.Array:
  """Conditional pseudolikelihood loss with a residual sequence energy.

  Implements equations (eq:sequence-energy-one-site-conditional) and
  (eq:empirical-exact-sequence-pl-loss) from the posterior-bridges
  manuscript: at every selected masked site ``i``, evaluate the
  energy-adjusted vocabulary logits

      ``ell_{b, i}(v) = log q_theta^i(v | x_t^b)
                         - lambda * E_phi(x_0^{b, i <- v}, x_t^b, t_b)``

  and apply ordinary softmax cross-entropy with target token
  ``x_0^{b, i}``.  Non-masked sites are excluded from the average.

  Args:
    logits: ``[B, n, V]`` raw logits from the tokenwise denoiser
      ``q_theta``.  ``log_softmax`` is applied internally to obtain the
      proposal log-probabilities ``log q_theta^i(v | x_t)``.
    energy_fn: Callable ``E_phi(x0_candidate, xt, time) -> [...]``
      returning a scalar per leading-axis element.  We call it on a
      flattened ``[B * n * V, n]`` batch of candidate sequences (broadcast
      ``xt`` and ``time`` accordingly), so the function must support
      arbitrary leading-axis sizes -- typically a pure Flax / JAX network
      bound via closure of its params.
    x0: ``[B, n]`` integer-valued ground-truth clean sequence.
    xt: ``[B, n]`` integer-valued noisy state (mask token at masked
      positions for absorbing-mask discrete diffusion).
    time: ``[B, ...]`` time array, broadcast across the candidate axes.
    masked_sites: ``[B, n]`` boolean mask, ``True`` where the site is
      masked in ``xt`` and contributes to the loss.  Use the corruption
      process's ``targets['is_corrupted']`` (squeezed of any trailing
      length-1 axis) for the canonical convention.
    lam: ``lambda`` interaction strength.  At ``lam=0`` the energy term
      drops out and the loss reduces to standard tokenwise masked
      cross-entropy.
    vocab_size: ``V``.  Inferred from ``logits.shape[-1]`` if ``None``.
    eps: small positive constant added to the per-batch denominator to
      avoid division by zero when no site is masked in a batch element.
    schedule: optional schedule forwarded to ``weight_fn``.
    weight_fn: optional per-time loss reweighting; must produce shape
      ``[B, ...]`` reduce-able to ``[B,]``.  Multiplies the per-sample
      loss after pseudolikelihood averaging.

  Returns:
    Per-sample loss of shape ``[B,]``.
  """
  if logits.ndim != 3:
    raise ValueError(
        f"logits must have shape [B, n, V]; got {logits.shape}."
    )
  if x0.ndim != 2 or xt.ndim != 2 or masked_sites.ndim != 2:
    raise ValueError(
        "x0, xt, masked_sites must all have shape [B, n]; got "
        f"{x0.shape=}, {xt.shape=}, {masked_sites.shape=}."
    )
  bsz, n, V_logits = logits.shape
  if x0.shape != (bsz, n) or xt.shape != (bsz, n):
    raise ValueError(
        "logits / x0 / xt batch / sequence dims must match: "
        f"{logits.shape=}, {x0.shape=}, {xt.shape=}."
    )
  V = int(vocab_size) if vocab_size is not None else V_logits
  if V != V_logits:
    raise ValueError(
        f"vocab_size={V} disagrees with logits last axis {V_logits}."
    )

  # Build [B, n, V, n] candidate tensor: x0_candidates[b, i, v, :] has
  # x0[b, :] with position i replaced by v.
  x0_candidates = _build_one_site_candidates(x0, V)            # [B, n, V, n]

  # Replicate xt and time across (i, v) before flattening so each
  # candidate's energy sees the same xt/time as its source batch element.
  xt_rep = jnp.broadcast_to(
      xt[:, None, None, :], (bsz, n, V, n),
  )                                                            # [B, n, V, n]
  # Time may carry trailing axes (e.g. broadcast-1 dims). Match them.
  time_trailing = time.shape[1:]
  time_rep = jnp.broadcast_to(
      time.reshape((bsz,) + (1, 1) + time_trailing),
      (bsz, n, V) + time_trailing,
  )                                                            # [B, n, V, ...]

  flat_size = bsz * n * V
  candidates_flat = x0_candidates.reshape(flat_size, n)
  xt_flat = xt_rep.reshape(flat_size, n)
  time_flat = time_rep.reshape((flat_size,) + time_trailing)

  energies_flat = energy_fn(candidates_flat, xt_flat, time_flat)
  if energies_flat.shape != (flat_size,):
    raise ValueError(
        "energy_fn must emit shape [B*n*V,]; got "
        f"{energies_flat.shape}, expected {(flat_size,)}."
    )
  energies = energies_flat.reshape(bsz, n, V)                  # [B, n, V]

  # Adjusted vocabulary logits per (b, i):
  #   ell_{b, i}(v) = log_softmax(logits)[b, i, v] - lam * E_phi[b, i, v]
  # Note: applying log_softmax before subtracting lam*E preserves
  # propriety; the softmax over v afterwards renormalizes correctly.
  log_q = jax.nn.log_softmax(logits, axis=-1)                  # [B, n, V]
  adj_logits = log_q - float(lam) * energies                   # [B, n, V]

  # Per-site cross-entropy with target x0[b, i].
  per_site_neg_log_p = optax.softmax_cross_entropy_with_integer_labels(
      logits=adj_logits, labels=x0,
  )                                                            # [B, n]

  # Mask + average over masked sites per batch element.
  mask = masked_sites.astype(per_site_neg_log_p.dtype)         # [B, n]
  num_masked = jnp.sum(mask, axis=-1)                          # [B]
  numer = jnp.sum(per_site_neg_log_p * mask, axis=-1)          # [B]
  per_sample_loss = numer / jnp.clip(num_masked, min=eps)      # [B]

  if weight_fn is not None:
    weight = weight_fn(
        schedule=schedule,
        preds={"logits": logits},
        targets={
            "x0": x0[..., None],
            "is_corrupted": masked_sites[..., None],
        },
        time=time,
    )
    weight = utils.flatten_non_batch_dims(weight).reshape(bsz, -1).mean(axis=-1)
    per_sample_loss = per_sample_loss * weight

  return per_sample_loss


################################################################################
# MARK: Specific loss class
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class MaskedPseudolikelihoodLoss(base.DiffusionLoss):
  """Conditional pseudolikelihood loss with a residual sequence energy.

  Wrapper around :func:`compute_masked_pseudolikelihood_loss` that
  conforms to the :class:`DiffusionLoss` Protocol.  The training step
  must thread ``xt`` and the energy callable through; see the
  ``compute_*`` docstring for the contract.

  Because the standard ``DiffusionLoss`` signature does not include
  ``xt`` (it is hidden inside the corruption process during training),
  callers extend the ``targets`` dict with two extra keys:

  - ``targets['xt']``: ``[B, n]`` integer noisy state.
  - ``targets['is_corrupted']``: ``[B, n, 1]`` boolean mask (matches the
    convention used by :func:`compute_discrete_diffusion_loss`).  We
    squeeze the trailing length-1 axis internally.

  ``preds['logits']`` is the tokenwise denoiser ``q_theta``'s output as
  usual.

  Attributes:
    energy_fn: Residual sequence energy, see
      :data:`EnergySequenceFn` for the signature.
    lam: Interaction strength ``lambda``.  ``0.0`` recovers ordinary
      tokenwise masked cross-entropy (modulo the divisor convention).
    vocab_size: ``V``.  Inferred from logits if ``None``.
    schedule: Optional, forwarded to ``weight_fn``.
    weight_fn: Optional time weighting; multiplies the per-sample loss.
  """

  energy_fn: EnergySequenceFn
  lam: float = 1.0
  vocab_size: int | None = None
  eps: float = 1e-12
  schedule: DiscreteSchedule | None = None
  weight_fn: base.WeightFn | None = None

  @kt.typechecked
  def __call__(
      self,
      preds: TargetInfo,
      targets: TargetInfo,
      time: TimeArray,
  ) -> LossOutput:
    if "logits" not in preds:
      raise ValueError("preds must contain 'logits' from the tokenwise denoiser.")
    if "xt" not in targets:
      raise ValueError(
          "targets must contain 'xt' for the residual energy network. "
          "Add it to the corruption process's target_info or thread it "
          "through your training step."
      )
    if "x0" not in targets:
      raise ValueError("targets must contain 'x0'.")
    if "is_corrupted" not in targets:
      raise ValueError(
          "targets must contain 'is_corrupted' (boolean mask of masked "
          "positions in xt)."
      )

    # Squeeze the trailing length-1 axis used by the existing discrete
    # corruption process for x0 / is_corrupted.
    x0 = jnp.squeeze(targets["x0"], axis=-1)                    # [B, n]
    masked_sites = jnp.squeeze(targets["is_corrupted"], axis=-1).astype(
        jnp.bool_
    )                                                           # [B, n]
    xt = targets["xt"]
    if xt.ndim == x0.ndim + 1 and xt.shape[-1] == 1:
      xt = jnp.squeeze(xt, axis=-1)                             # [B, n]

    return compute_masked_pseudolikelihood_loss(
        logits=preds["logits"],
        energy_fn=self.energy_fn,
        x0=x0,
        xt=xt,
        time=time,
        masked_sites=masked_sites,
        lam=self.lam,
        vocab_size=self.vocab_size,
        eps=self.eps,
        schedule=self.schedule,
        weight_fn=self.weight_fn,
    )


################################################################################
# MARK: Conditional NCE variant for large vocabularies
################################################################################


@kt.typechecked
def compute_masked_pseudolikelihood_nce_loss(
    *,
    logits: jax.Array,
    energy_fn: EnergySequenceFn,
    bias_fn: BiasSiteFn,
    x0: jax.Array,
    xt: jax.Array,
    time: TimeArray,
    masked_sites: jax.Array,
    proposal_log_probs: jax.Array,
    negatives: jax.Array,
    lam: float = 1.0,
    eps: float = 1e-12,
    schedule: DiscreteSchedule | None = None,
    weight_fn: base.WeightFn | None = None,
) -> jax.Array:
  """Conditional-NCE pseudolikelihood for large vocabularies.

  Implements equations (eq:conditional-nce-sequence-loss) from manuscript
  section "Sequence-Level Scoring Rules / Large-vocabulary
  approximations".  Replaces the full conditional softmax over ``V``
  vocabulary symbols at each masked site with a binary classifier
  separating one positive replacement (the observed token) from ``K``
  proposal negatives.  Per masked site the energy network is evaluated
  on ``K + 1`` candidate sequences, not ``V`` -- the right complexity
  shape when ``V`` is large (text-scale) but ``K`` can stay modest.

  For each (b, i):
      a_lambda(v; c) = log q_theta^i(v | x_t)
                       - lambda * E_phi(x_0^{i <- v}, x_t, t)
      r_lambda(v; c) = a_lambda(v; c) + b_psi(c)
                       - log K - log pi_i(v | c)
  Binary logistic loss with the positive at the observed token
  ``y = x_0[b, i]`` and ``K`` negatives ``v_1, ..., v_K`` sampled from
  ``pi_i(. | c)``:
      ell_NCE = - log sigmoid(r_lambda(y; c))
                - sum_k log sigmoid(-r_lambda(v_k; c)).

  Under standard conditional-NCE assumptions (proposal support, logit
  identifiability, sufficient capacity in ``E_phi`` and ``b_psi``), the
  large-K, large-sample limit recovers the same one-site conditionals
  the exact pseudolikelihood (above) trains.

  The candidate at the positive index is ``x_0`` itself (replacing site
  ``i`` with its own value is a no-op), so we compute the positive
  energy once per batch element and broadcast across sites.  Negative
  candidates are ``x_0`` with site ``i`` replaced by ``negatives[b, i, k]``.

  Args:
    logits: ``[B, n, V]`` raw tokenwise denoiser logits.
    energy_fn: Residual sequence energy ``E_phi(x0_candidate, xt, time)
      -> [batch]``.  We call it on a flattened ``[B * n * (K + 1), n]``
      candidate batch.
    bias_fn: Per-site bias ``b_psi(x0, xt, time) -> [B, n]`` estimating
      the negative local log normalizer at each site.  In the
      manuscript's notation ``b_psi`` depends on ``c = (i, x_0^{-i},
      x_t, t)``; we feed it the observed clean sequence ``x_0`` plus
      ``xt`` and ``time``, and let it produce one bias per site.
    x0: ``[B, n]`` integer ground truth.
    xt: ``[B, n]`` integer noisy state.
    time: ``[B, ...]``.
    masked_sites: ``[B, n]`` boolean.  Only masked sites contribute.
    proposal_log_probs: ``[B, n, V]`` log-probs of the negative-sampling
      proposal ``pi_i(. | c)`` evaluated at every vocabulary symbol.
      Caller provides; canonical choices are the same ``q_theta`` (then
      the ratio ``a_lambda - log pi`` collapses to ``-lambda * E_phi
      - log K``) or a uniform-over-vocabulary baseline.
    negatives: ``[B, n, K]`` integer-valued negatives per site, drawn
      independently from ``proposal_log_probs[b, i, :]``.  Caller draws
      these (the loss must stay deterministic given the rng-free
      candidates so jit / grad behave predictably).
    lam: Interaction strength.
    eps: Per-batch denominator floor.
    schedule, weight_fn: Optional time weighting; same convention as
      :func:`compute_masked_pseudolikelihood_loss`.

  Returns:
    Per-sample loss of shape ``[B,]``.
  """
  if logits.ndim != 3:
    raise ValueError(f"logits must be [B, n, V]; got {logits.shape}.")
  bsz, n, V = logits.shape
  if x0.shape != (bsz, n) or xt.shape != (bsz, n):
    raise ValueError(
        "logits / x0 / xt batch / sequence dims must match: "
        f"{logits.shape=}, {x0.shape=}, {xt.shape=}."
    )
  if masked_sites.shape != (bsz, n):
    raise ValueError(f"masked_sites must be [B, n]; got {masked_sites.shape}.")
  if proposal_log_probs.shape != (bsz, n, V):
    raise ValueError(
        "proposal_log_probs must be [B, n, V]; got "
        f"{proposal_log_probs.shape}."
    )
  if negatives.ndim != 3 or negatives.shape[:2] != (bsz, n):
    raise ValueError(
        f"negatives must be [B, n, K]; got {negatives.shape}."
    )
  K = negatives.shape[-1]
  if K < 1:
    raise ValueError(f"K must be >= 1; got K={K}.")

  # all_tokens[b, i, 0] = x0[b, i] (positive); rest are negatives.
  positive_token = x0[..., None]                              # [B, n, 1]
  all_tokens = jnp.concatenate([positive_token, negatives], axis=-1)  # [B, n, K+1]

  # Build [B, n, K+1, n] candidate tensor: x0 with site i replaced by
  # all_tokens[b, i, k].
  position_mask = jnp.eye(n, dtype=jnp.bool_)                 # [n, n]
  x0_b = x0[:, None, None, :]                                 # [B, 1, 1, n]
  candidates = jnp.where(
      position_mask[None, :, None, :],                        # [1, n, 1, n]
      all_tokens[..., None],                                  # [B, n, K+1, 1]
      x0_b,
  )                                                            # [B, n, K+1, n]

  xt_rep = jnp.broadcast_to(
      xt[:, None, None, :], (bsz, n, K + 1, n),
  )                                                            # [B, n, K+1, n]
  time_trailing = time.shape[1:]
  time_rep = jnp.broadcast_to(
      time.reshape((bsz,) + (1, 1) + time_trailing),
      (bsz, n, K + 1) + time_trailing,
  )

  flat_size = bsz * n * (K + 1)
  candidates_flat = candidates.reshape(flat_size, n)
  xt_flat = xt_rep.reshape(flat_size, n)
  time_flat = time_rep.reshape((flat_size,) + time_trailing)

  energies_flat = energy_fn(candidates_flat, xt_flat, time_flat)
  if energies_flat.shape != (flat_size,):
    raise ValueError(
        f"energy_fn must emit shape [B*n*(K+1),]; got {energies_flat.shape},"
        f" expected {(flat_size,)}."
    )
  energies = energies_flat.reshape(bsz, n, K + 1)              # [B, n, K+1]

  log_q = jax.nn.log_softmax(logits, axis=-1)                  # [B, n, V]
  log_q_at_tokens = jnp.take_along_axis(log_q, all_tokens, axis=-1)
  # [B, n, K+1]

  # a_lambda(v; c) = log q_theta^i(v|xt) - lambda * E_phi(...)
  a_lambda = log_q_at_tokens - float(lam) * energies          # [B, n, K+1]

  bias = bias_fn(x0, xt, time)                                # [B, n]
  if bias.shape != (bsz, n):
    raise ValueError(
        f"bias_fn must emit [B, n]; got {bias.shape}, expected {(bsz, n)}."
    )

  log_proposal = jnp.take_along_axis(
      proposal_log_probs, all_tokens, axis=-1,
  )                                                            # [B, n, K+1]

  log_K = jnp.log(jnp.asarray(K, dtype=a_lambda.dtype))

  # r_lambda(v; c) = a_lambda + b_psi(c) - log K - log pi_i(v | c)
  r = a_lambda + bias[..., None] - log_K - log_proposal       # [B, n, K+1]

  # Binary logistic loss: positive at index 0, negatives at 1..K.
  # ``-log sigmoid(x) = softplus(-x)``.
  pos_loss = jax.nn.softplus(-r[..., 0])                       # [B, n]
  neg_loss = jnp.sum(jax.nn.softplus(r[..., 1:]), axis=-1)     # [B, n]
  per_site_loss = pos_loss + neg_loss                          # [B, n]

  mask = masked_sites.astype(per_site_loss.dtype)
  num_masked = jnp.sum(mask, axis=-1)
  numer = jnp.sum(per_site_loss * mask, axis=-1)
  per_sample_loss = numer / jnp.clip(num_masked, min=eps)

  if weight_fn is not None:
    weight = weight_fn(
        schedule=schedule,
        preds={"logits": logits},
        targets={
            "x0": x0[..., None],
            "is_corrupted": masked_sites[..., None],
        },
        time=time,
    )
    weight = utils.flatten_non_batch_dims(weight).reshape(bsz, -1).mean(axis=-1)
    per_sample_loss = per_sample_loss * weight

  return per_sample_loss


@dataclasses.dataclass(frozen=True, kw_only=True)
class MaskedPseudolikelihoodNCELoss(base.DiffusionLoss):
  """Conditional-NCE pseudolikelihood loss with a residual sequence energy.

  Wrapper around :func:`compute_masked_pseudolikelihood_nce_loss`.  The
  training step must thread three extra fields into ``targets`` /
  ``preds``:

  - ``targets['xt']``: ``[B, n]`` integer noisy state.
  - ``targets['is_corrupted']``: ``[B, n, 1]`` boolean mask, same
    convention as :func:`compute_discrete_diffusion_loss`.
  - ``preds['proposal_log_probs']``: ``[B, n, V]`` log-probs of the
    negative-sampling proposal at every site/vocabulary entry.
  - ``preds['negatives']``: ``[B, n, K]`` integer negatives drawn from
    ``proposal_log_probs``.

  Drawing the negatives must happen outside the loss (the loss stays
  deterministic in its inputs).  A typical setup uses the tokenwise
  ``q_theta`` itself as the proposal, so the term ``a_lambda - log pi``
  collapses to ``-lambda * E_phi``; that's a useful self-consistency
  baseline against which the bias term ``b_psi`` learns the local
  normalizer.

  Attributes:
    energy_fn: Residual sequence energy.
    bias_fn: Per-site bias ``b_psi``.
    lam: Interaction strength.
    schedule, weight_fn: Optional time weighting.
  """

  energy_fn: EnergySequenceFn
  bias_fn: BiasSiteFn
  lam: float = 1.0
  eps: float = 1e-12
  schedule: DiscreteSchedule | None = None
  weight_fn: base.WeightFn | None = None

  @kt.typechecked
  def __call__(
      self,
      preds: TargetInfo,
      targets: TargetInfo,
      time: TimeArray,
  ) -> LossOutput:
    if "logits" not in preds:
      raise ValueError("preds must contain 'logits'.")
    if "proposal_log_probs" not in preds:
      raise ValueError(
          "preds must contain 'proposal_log_probs' [B, n, V] for the "
          "negative-sampling proposal."
      )
    if "negatives" not in preds:
      raise ValueError(
          "preds must contain 'negatives' [B, n, K] integer negative "
          "samples drawn from proposal_log_probs."
      )
    for k in ("x0", "is_corrupted", "xt"):
      if k not in targets:
        raise ValueError(f"targets must contain '{k}'.")

    x0 = jnp.squeeze(targets["x0"], axis=-1)
    masked_sites = jnp.squeeze(targets["is_corrupted"], axis=-1).astype(
        jnp.bool_
    )
    xt = targets["xt"]
    if xt.ndim == x0.ndim + 1 and xt.shape[-1] == 1:
      xt = jnp.squeeze(xt, axis=-1)

    return compute_masked_pseudolikelihood_nce_loss(
        logits=preds["logits"],
        energy_fn=self.energy_fn,
        bias_fn=self.bias_fn,
        x0=x0,
        xt=xt,
        time=time,
        masked_sites=masked_sites,
        proposal_log_probs=preds["proposal_log_probs"],
        negatives=preds["negatives"],
        lam=self.lam,
        eps=self.eps,
        schedule=self.schedule,
        weight_fn=self.weight_fn,
    )
