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

"""The posterior-bridge sampler step (Posterior Bridges, Algorithm 2).

For a distributional / posterior model the finite-step update is *not* a
score / velocity reverse-time integrator.  It is two operations, identical
across every modality:

  1. ``x_0 = process.sample_endpoint(prediction)`` -- draw one clean
     endpoint from the model's posterior report (the identity for a model
     that already emits an ``x_0`` sample, a categorical draw from logits
     for discrete, a simplex point for the Dirichlet model);
  2. ``x_s = process.bridge_step(x_0, x_t, t -> s)`` -- apply the
     corruption's own two-endpoint bridge ``K_{s|0,t}``.

A single :class:`PosteriorBridgeStep` runs this for *all* modalities,
because ``sample_endpoint`` and ``bridge_step`` are the reverse face of
the same :class:`hackable_diffusion.lib.corruption.base.CorruptionProcess`
that defined the corruption at training (``corrupt``).  Consistency with
training is therefore structural -- there is no second copy of the
geometry to keep in sync -- and multimodal falls out for free, since
``NestedProcess`` maps both methods over its sub-processes.

This is the continuous/discrete/simplex-agnostic realisation of the
manuscript's modularity claim ``(geometry = process) x (posterior =
inference_fn)``.  The geometry-specific math lives on each process
(``InterpolantProcess`` delegates to its interpolant's ``bridge_step``;
``CategoricalProcess`` reveals; ``SimplicialProcess`` mixes Beta/Dirichlet);
the learned object lives entirely in the model's report of ``x_0``.

Relationship to the existing steppers (so there is no ambiguity):

  * For a plain Gaussian ``LinearInterpolant``, the bridge step is the
    deterministic affine map -- *identical* to ``DDIMStep(stoch_coeff=0)``.
    Either can sample a Gaussian distributional model; ``DDIMStep`` is the
    conventional choice.
  * The feature-rich discrete / simplicial steppers (``UnMaskingStep``,
    ``DiscreteDDIMStep``, ``DiscreteFlowMatchingStep``, ``SimplicialDDIMStep``)
    are enhancements of the same bridge with extra knobs (remasking,
    planning, churn).  ``PosteriorBridgeStep`` is the bare, uniform path.
"""

import dataclasses

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.corruption import base as corruption_base
from hackable_diffusion.lib.sampling import base
import jax
import kauldron.ktyping as kt

################################################################################
# MARK: Type Aliases
################################################################################

DataArray = hd_typing.DataArray
TargetInfo = hd_typing.TargetInfo

DiffusionStep = base.DiffusionStep
StepInfo = base.StepInfo
SamplerStep = base.SamplerStep

CorruptionProcess = corruption_base.CorruptionProcess

# Salts that derive the endpoint-draw and bridge-noise keys from the step
# rng as independent streams (and independent of any salt the inference fn
# folds in for its own noise).  Hex constants for traceability in logs.
_ENDPOINT_RNG_SALT = 0xB1D6_0E0
_BRIDGE_RNG_SALT = 0xB1D6_0E1


################################################################################
# MARK: Posterior Bridge Step
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class PosteriorBridgeStep(SamplerStep):
  """Draw ``x_0`` from the posterior, then apply the process's bridge ``K_{s|0,t}``.

  Modality-agnostic: it delegates both the endpoint draw and the bridge to
  ``corruption_process``, so the *same* stepper drives Gaussian, stochastic
  interpolant, geodesic, categorical, Dirichlet, and multimodal
  (``NestedProcess``) data.  Pair it with an inference fn that reports the
  clean-endpoint posterior in the modality's parameterisation -- an ``x0``
  sample for continuous models (e.g.
  :class:`hackable_diffusion.lib.inference.PosteriorSamplerInferenceFn`),
  logits for discrete / simplicial ones.

  Attributes:
    corruption_process: the process whose ``sample_endpoint`` /
      ``bridge_step`` define the reverse step (see
      :class:`hackable_diffusion.lib.corruption.base.CorruptionProcess`).
  """

  corruption_process: CorruptionProcess

  @kt.typechecked
  def initialize(
      self,
      initial_noise: DataArray,
      initial_step_info: StepInfo,
  ) -> DiffusionStep:
    return DiffusionStep(
        xt=initial_noise,
        step_info=initial_step_info,
        aux=dict(),
    )

  def _step(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
  ) -> DiffusionStep:
    xt = current_step.xt
    t = current_step.step_info.time
    s = next_step_info.time

    endpoint_key = jax.random.fold_in(next_step_info.rng, _ENDPOINT_RNG_SALT)
    bridge_key = jax.random.fold_in(next_step_info.rng, _BRIDGE_RNG_SALT)

    x0 = self.corruption_process.sample_endpoint(endpoint_key, prediction, xt, t)
    x_s = self.corruption_process.bridge_step(bridge_key, x0, xt, t, s)
    return DiffusionStep(
        xt=x_s,
        step_info=next_step_info,
        aux=dict(),
    )

  @kt.typechecked
  def update(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
  ) -> DiffusionStep:
    return self._step(prediction, current_step, next_step_info)

  @kt.typechecked
  def finalize(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      last_step_info: StepInfo,
  ) -> DiffusionStep:
    # At s = 0 the bridge collapses to the clean endpoint
    # (K_{0|0,t} = delta_{x_0}); _step handles that limit directly.
    return self._step(prediction, current_step, last_step_info)
