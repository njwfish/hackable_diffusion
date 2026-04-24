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

Six callable Protocols (``DenoiserFn``, ``ForwardFn``,
``PosteriorCovarianceFn``, ``CorrectionFn``, ``TwistFn``,
``ResamplerFn``) plus the step-level ``StepKernel`` compose inside
:class:`ConditionalDiffusionSampler` to express Pi-GDM (Kalman), DPS,
TDS, MCGDiff, CFG, classifier guidance, iterated Pi-GDM, and other
posterior-sampling methods as configurations.
"""

from hackable_diffusion.lib.guidance.adapters import BoundAggregateGuidanceFn
from hackable_diffusion.lib.guidance.corrections import (
    GradientCorrectionFn,
    IteratedCorrectionFn,
    KalmanCorrectionFn,
    PrefactorFn,
    dps_prefactor,
    miyasawa_prefactor,
)
from hackable_diffusion.lib.guidance.denoisers import (
    LinearBlendDenoiserFn,
    cfg_denoiser_fn,
    make_cfg_inference_fn,
    make_denoiser_fn,
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
    scalar_alpha,
    scalar_alpha_sigma,
)
from hackable_diffusion.lib.sampling.base import StepKernel
from hackable_diffusion.lib.sampling.gaussian_step_sampler import (
    GaussianStepKernel,
)
from hackable_diffusion.lib.sampling.simplicial_step_sampler import (
    SimplicialStepKernel,
)

__all__ = [
    "BoundAggregateGuidanceFn",
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
    "KalmanCorrectionFn",
    "LinearBlendDenoiserFn",
    "LinearForwardFn",
    "LogProbFn",
    "MultinomialResamplerFn",
    "NoResamplerFn",
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
    "cfg_denoiser_fn",
    "dps_prefactor",
    "linear_adjoint",
    "make_cfg_inference_fn",
    "make_denoiser_fn",
    "miyasawa_prefactor",
    "miyasawa_scale",
    "normalised_weights",
    "proposal_log_ratio",
    "scalar_alpha",
    "scalar_alpha_sigma",
]
