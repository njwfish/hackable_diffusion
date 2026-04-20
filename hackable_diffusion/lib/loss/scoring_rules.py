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

"""Scoring-rule losses for distributional diffusion.

Implements the (generalized) energy-score loss from De Bortoli et al.,
"Distributional Diffusion Models with Scoring Rules" (arXiv:2502.02483),
Eq. (15-16). The network is expected to output ``m`` samples per input via an
extra noise input ``xi``; the loss consumes those samples through an extra
"population" axis in ``preds``.
"""

import dataclasses

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.loss import base
import jax.numpy as jnp

################################################################################
# MARK: Type Aliases
################################################################################

LossOutput = hd_typing.LossOutput
TargetInfo = hd_typing.TargetInfo
TimeArray = hd_typing.TimeArray

GaussianSchedule = schedules.GaussianSchedule

################################################################################
# MARK: General Loss function
################################################################################


def compute_energy_score_loss(
    preds: TargetInfo,
    targets: TargetInfo,
    time: TimeArray,
    *,
    beta: float = 1.0,
    lam: float = 1.0,
    eps: float = 1e-12,
    prediction_type: str = "x0",
    schedule: GaussianSchedule | None = None,
    weight_fn: base.WeightFn | None = None,
) -> LossOutput:
  """Compute the (generalized) conditional energy-score diffusion loss.

  Implements the U-statistic form of Eq. (15-16) in arXiv:2502.02483:

      L_i = mean_j ||x0_i - xhat_ij||^beta
            - lam / (2*M*(M-1)) * sum_{j != j'} ||xhat_ij - xhat_ij'||^beta

  Expected shapes:
    preds[prediction_type]:    [B, M, *data]  (M = population size, with M >= 2)
    targets[prediction_type]:  [B, *data]

  Strict properness of the conditional energy score requires lam=1 and
  beta in (0, 2). beta=2 is included for completeness but is not strictly proper.

  Efficiency / numerics:
    * Pairwise L2s use direct ``sum((a-b)**2)`` in flattened feature space, not
      a Gram expansion. XLA fuses the elementwise subtract, square, and
      reduction, so peak memory is ``[B, M, M]`` — same as the Gram route — but
      accumulation stays in the full input dtype (matmul on modern accelerators
      silently downcasts to TF32 / bf16 and leaks ~1e-4 relative noise into
      ``|u-v|^2``, which is catastrophic near coincident samples).
    * beta=2 and beta=1 are specialised to skip ``pow`` / ``sqrt`` dispatch.
    * The diagonal of the pairwise tensor is masked out (not skipped in python)
      so XLA folds the mask into the reduction.
    * An eps clip before any fractional root keeps gradients finite when two
      samples land on top of each other (which happens early in training).

  Args:
    preds: Dict of predictions; ``preds[prediction_type]`` has shape
      ``[B, M, *data]``. Produced by m forward passes of the distributional
      network with different ``xi`` draws, stacked along axis 1.
    targets: Dict of targets; ``targets[prediction_type]`` has shape
      ``[B, *data]``. Compared against each of the M prediction samples.
    time: Time array ``[B, ...]`` (only used by an optional ``weight_fn``).
    beta: The energy-score exponent. ``beta in (0, 2]``; default 1.0 (classical
      Gneiting-Raftery energy score, strictly proper with lam=1).
    lam: Interaction-term weight. ``lam in [0, 1]``; default 1.0 (strictly
      proper).
    eps: Small positive constant added inside roots for stability.
    prediction_type: Key to read from ``preds`` and ``targets``. Default ``x0``.
    schedule: Optional schedule forwarded to ``weight_fn``.
    weight_fn: Optional additional per-time weighting; multiplies the per-
      sample loss.

  Returns:
    Per-sample loss of shape ``[B,]``.
  """
  xhat = preds[prediction_type]
  x0 = targets[prediction_type]

  if xhat.ndim < 2:
    raise ValueError(
        f"preds[{prediction_type!r}] must have a leading [batch, population] "
        f"pair, got shape {xhat.shape}."
    )
  bsz, pop = xhat.shape[0], xhat.shape[1]
  if pop < 2:
    raise ValueError(
        f"Energy-score loss requires population size M >= 2, got M={pop}."
    )
  if x0.shape[0] != bsz:
    raise ValueError(
        f"Batch dim mismatch: preds has {bsz}, targets has {x0.shape[0]}."
    )

  # Flatten feature dims for the Gram/norm math.
  xhat_flat = xhat.reshape(bsz, pop, -1)                          # [B, M, D]
  x0_flat = x0.reshape(bsz, -1)                                   # [B, D]

  # Data term: squared distances between x0 and each of the M predictions.
  # Direct subtract then square-sum is one pass and clearer than the expansion.
  diff = xhat_flat - x0_flat[:, None, :]
  data_sq = jnp.sum(diff * diff, axis=-1)                         # [B, M]

  # Interaction term: pairwise squared distances among the M predictions.
  # Direct subtraction + square-reduce (fused by XLA to [B, M, M] peak memory)
  # keeps full-precision accumulation. The Gram expansion ||u||^2 + ||v||^2 -
  # 2<u,v> is faster per FLOP but the matmul silently uses TF32/bf16 on modern
  # accelerators and leaks noise that breaks gradients near coincident samples.
  pair_diff = xhat_flat[:, :, None, :] - xhat_flat[:, None, :, :]  # [B, M, M, D]
  pair_sq = jnp.sum(pair_diff * pair_diff, axis=-1)                 # [B, M, M]

  # Clip before any fractional root. Off-diagonal values can be slightly
  # negative from fp cancellation; on-diagonal is exactly zero and gets masked.
  data_sq = jnp.maximum(data_sq, eps)
  pair_sq = jnp.maximum(pair_sq, eps)

  # beta-specific dispatch at Python level — compile-time branch, no cost.
  if beta == 2.0:
    data_term = data_sq.mean(axis=1)                              # [B]
    pair = pair_sq                                                # [B, M, M]
  elif beta == 1.0:
    data_term = jnp.sqrt(data_sq).mean(axis=1)
    pair = jnp.sqrt(pair_sq)
  else:
    half_beta = beta / 2.0
    data_term = jnp.power(data_sq, half_beta).mean(axis=1)
    pair = jnp.power(pair_sq, half_beta)

  # Zero the diagonal then normalise by the U-statistic denominator 2*M*(M-1).
  # (The double sum over j != j' has M*(M-1) terms; the factor of 2 comes
  # from the paper's per-(i,j) formulation summed over j.)
  mask = 1.0 - jnp.eye(pop, dtype=pair.dtype)
  interaction = jnp.sum(pair * mask, axis=(1, 2)) / (
      2.0 * pop * (pop - 1)
  )                                                               # [B]

  per_sample = data_term - lam * interaction                      # [B]

  if weight_fn is not None:
    weight = weight_fn(
        schedule=schedule, preds=preds, targets=targets, time=time
    )
    # weight may broadcast over trailing dims; reduce to [B] to match per_sample.
    weight = jnp.asarray(weight).reshape(bsz, -1).mean(axis=-1)
    per_sample = per_sample * weight

  return per_sample


################################################################################
# MARK: Specific Loss functions
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class EnergyScoreLoss(base.DiffusionLoss):
  """Conditional generalized energy-score loss.

  See ``compute_energy_score_loss`` for details and shape expectations. Default
  hyperparameters (``lam=1.0, beta=1.0``) give the classical, strictly proper
  energy score.
  """

  beta: float = 1.0
  lam: float = 1.0
  eps: float = 1e-12
  prediction_type: str = "x0"
  schedule: GaussianSchedule | None = None
  weight_fn: base.WeightFn | None = None

  def __call__(
      self,
      preds: TargetInfo,
      targets: TargetInfo,
      time: TimeArray,
  ) -> LossOutput:
    return compute_energy_score_loss(
        preds=preds,
        targets=targets,
        time=time,
        beta=self.beta,
        lam=self.lam,
        eps=self.eps,
        prediction_type=self.prediction_type,
        schedule=self.schedule,
        weight_fn=self.weight_fn,
    )
