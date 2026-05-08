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

"""Inference class for diffusion models."""

import dataclasses
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.inference import base
from hackable_diffusion.lib.inference import guidance
from hackable_diffusion.lib.inference import projection
import kauldron.ktyping as kt

################################################################################
# MARK: Type Aliases
################################################################################

Conditioning = hd_typing.Conditioning
DataTree = hd_typing.DataTree
PRNGKey = hd_typing.PRNGKey
TimeTree = hd_typing.TimeTree
TargetInfoTree = hd_typing.TargetInfoTree

InferenceFn = base.InferenceFn

################################################################################
# MARK: GuidedDiffusionInferenceFn
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class GuidedDiffusionInferenceFn(InferenceFn):
  """Inference function protocol."""

  base_inference_fn: InferenceFn
  guidance_fn: guidance.GuidanceFn = guidance.ScalarGuidanceFn()
  projection_fn: projection.ProjectionFn = projection.IdentityProjectionFn()

  @kt.typechecked
  def __call__(
      self,
      time: TimeTree,
      xt: DataTree,
      conditioning: Conditioning | None,
      rng: PRNGKey | None = None,
  ) -> TargetInfoTree:
    """Returns the model outputs.

    The optional ``rng`` is forwarded to both cond and uncond calls on
    the base inference fn, so stochastic base fns (e.g. posterior-
    sampler) share the same noise between the cond/uncond predictions
    -- necessary for a well-defined classifier-free guidance direction.
    """

    cond_outputs = self.base_inference_fn(
        time=time,
        xt=xt,
        conditioning=conditioning,
        rng=rng,
    )
    uncond_outputs = self.base_inference_fn(
        time=time,
        xt=xt,
        conditioning=None,
        rng=rng,
    )

    guided_outputs = self.guidance_fn(
        xt=xt,
        conditioning=conditioning,
        time=time,
        cond_outputs=cond_outputs,
        uncond_outputs=uncond_outputs,
    )

    projected_outputs = self.projection_fn(
        xt=xt,
        conditioning=conditioning,
        time=time,
        outputs=guided_outputs,
    )
    return projected_outputs
