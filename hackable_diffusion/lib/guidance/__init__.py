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

"""Composable conditional-sampling framework.

Independent Protocol axes -- :class:`CorrectionFn`, :class:`TwistFn`,
:class:`ResamplerFn`, :class:`ForwardFn`, :class:`PosteriorCovarianceFn`
-- compose inside :class:`ConditionalDiffusionSampler` to express
Pi-GDM, cov-aware, DPS, TDS, MCGDiff, and other published guidance
methods as configurations over a common diffusion sampler.
"""

from hackable_diffusion.lib.guidance.adapters import (
    BoundAggregateGuidanceFn,
    CFGCorrectionFn,
)
from hackable_diffusion.lib.guidance.corrections import (
    GradientCorrectionFn,
    IteratedCorrectionFn,
    PiGDMCorrectionFn,
    PrefactorFn,
    dps_prefactor,
    miyasawa_prefactor,
)
from hackable_diffusion.lib.guidance.forward_ops import (
    ComposeForwardFn,
    ConvForwardFn,
    InpaintingForwardFn,
    LinearForwardFn,
    SubsampleForwardFn,
)
from hackable_diffusion.lib.guidance.linalg import (
    batch_inner,
    batched_cg,
    linear_adjoint,
)
from hackable_diffusion.lib.guidance.posterior_covariance import (
    FixedPriorPosteriorCovarianceFn,
    IsotropicPosteriorCovarianceFn,
    ScaleFn,
    TweediePosteriorCovarianceFn,
    miyasawa_scale,
)
from hackable_diffusion.lib.guidance.protocols import (
    CorrectionFn,
    DenoiserFn,
    ForwardFn,
    PosteriorCovarianceFn,
    ResamplerFn,
    TwistFn,
)
from hackable_diffusion.lib.guidance.proposal_ratio import proposal_log_ratio
from hackable_diffusion.lib.sampling.base import StepKernel
from hackable_diffusion.lib.sampling.gaussian_step_sampler import (
    GaussianStepKernel,
)
from hackable_diffusion.lib.sampling.simplicial_step_sampler import (
    SimplicialStepKernel,
)
from hackable_diffusion.lib.guidance.resamplers import (
    ESSThresholdedResamplerFn,
    MultinomialResamplerFn,
    NoResamplerFn,
    SystematicResamplerFn,
    normalised_weights,
)
from hackable_diffusion.lib.guidance.sampler import ConditionalDiffusionSampler
from hackable_diffusion.lib.guidance.twists import (
    ClassifierTwistFn,
    DiscreteCompositionTwistFn,
    DiscreteMultiHeadCompositionTwistFn,
    EnergyFn,
    EnergyTwistFn,
    GaussianLikelihoodTwistFn,
    LogProbFn,
)
from hackable_diffusion.lib.guidance.utils import (
    accepts_rng_kwarg,
    call_inference_fn,
    make_denoiser_fn,
    replace_x0,
    scalar_alpha,
    scalar_alpha_sigma,
)

__all__ = [
    "BoundAggregateGuidanceFn",
    "CFGCorrectionFn",
    "ClassifierTwistFn",
    "ComposeForwardFn",
    "ConditionalDiffusionSampler",
    "ConvForwardFn",
    "CorrectionFn",
    "DenoiserFn",
    "DiscreteCompositionTwistFn",
    "DiscreteMultiHeadCompositionTwistFn",
    "ESSThresholdedResamplerFn",
    "EnergyFn",
    "EnergyTwistFn",
    "FixedPriorPosteriorCovarianceFn",
    "ForwardFn",
    "GaussianLikelihoodTwistFn",
    "GaussianStepKernel",
    "GradientCorrectionFn",
    "InpaintingForwardFn",
    "IsotropicPosteriorCovarianceFn",
    "IteratedCorrectionFn",
    "LinearForwardFn",
    "LogProbFn",
    "MultinomialResamplerFn",
    "NoResamplerFn",
    "PiGDMCorrectionFn",
    "PosteriorCovarianceFn",
    "PrefactorFn",
    "ResamplerFn",
    "ScaleFn",
    "SimplicialStepKernel",
    "StepKernel",
    "SubsampleForwardFn",
    "SystematicResamplerFn",
    "TweediePosteriorCovarianceFn",
    "TwistFn",
    "accepts_rng_kwarg",
    "batch_inner",
    "batched_cg",
    "call_inference_fn",
    "dps_prefactor",
    "linear_adjoint",
    "make_denoiser_fn",
    "miyasawa_prefactor",
    "miyasawa_scale",
    "normalised_weights",
    "proposal_log_ratio",
    "replace_x0",
    "scalar_alpha",
    "scalar_alpha_sigma",
]
