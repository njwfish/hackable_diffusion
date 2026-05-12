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

Seven callable Protocols (``DenoiserFn``, ``PosteriorCloudFn``,
``ForwardFn``, ``PosteriorCovarianceFn``, ``CorrectionFn``,
``TwistFn``, ``ResamplerFn``) plus the step-level ``StepKernel``
compose inside :class:`ConditionalDiffusionSampler` to express Pi-GDM,
DPS, TDS, MCGDiff, CFG, classifier guidance, iterated Pi-GDM,
posterior-sample SMC, projection guidance, and other posterior-
sampling methods as configurations.

``DenoiserFn`` and ``PosteriorCloudFn`` are the two interfaces a
twist or correction can read from: ``DenoiserFn`` is one prediction
``xt -> xhat_0(xt)`` (R=1); ``PosteriorCloudFn`` is the cloud-valued
analogue ``xt -> [B, R, *x0_shape]`` built by
:func:`make_posterior_cloud_fn`.  Cloud-aware twists (e.g. the SMC
potential estimator ``\\hat h_k^R = (1/R) \\sum L_y(x_0^r)``) consume
the cloud; the existing single-sample twists consume the denoiser.

Hard linear observations / clean-endpoint conditioning (inpainting,
exact-projection super-resolution, any ``A x_0 = y`` constraint) are
handled by :class:`KalmanCorrectionFn` with ``observation_noise = 0``
and ``solver="pinv"``.  Setting the posterior covariance to
``Cov = I`` -- via ``IsotropicPosteriorCovarianceFn(scale_fn=unit_scale)``
-- collapses the update to the pure affine projection
``x0 + A^T (A A^T)^+ (y - A x0)``, the elegant clean-endpoint form.
See :mod:`docs.composable_guidance` for a worked inpainting recipe.
"""

from hackable_diffusion.lib.guidance.corrections import (
    CategoricalProjectionCorrectionFn,
    GradientCorrectionFn,
    IteratedCorrectionFn,
    KalmanCorrectionFn,
    LogImportanceFn,
    ProjectionCloudCorrectionFn,
    ProjectionFn,
)
from hackable_diffusion.lib.guidance.denoisers import (
    LinearBlendDenoiserFn,
    cfg_denoiser_fn,
    make_cfg_inference_fn,
    make_denoiser_fn,
    make_posterior_cloud_fn,
)
from hackable_diffusion.lib.guidance.forward_ops import (
    ComposeForwardFn,
    ConvForwardFn,
    InpaintingForwardFn,
    LinearForwardFn,
    SubsampleForwardFn,
)
from hackable_diffusion.lib.guidance.gaussian_conditioning import (
    psd_pinv_solve,
    singular_gaussian_logpdf,
)
from hackable_diffusion.lib.guidance.linalg import (
    batch_inner,
    batched_cg,
    batched_minres,
    linear_adjoint,
)
from hackable_diffusion.lib.guidance.posterior_covariance import (
    FixedPriorPosteriorCovarianceFn,
    IsotropicPosteriorCovarianceFn,
    LowRankTweediePosteriorCovarianceFn,
    PCAPosteriorCovarianceFn,
    ScaleFn,
    TweediePosteriorCovarianceFn,
    miyasawa_scale,
    unit_scale,
)
from hackable_diffusion.lib.guidance.protocols import (
    CorrectionFn,
    DenoiserFn,
    ForwardFn,
    PosteriorCloudFn,
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
    EndpointTiltCloudTwistFn,
    EnergyFn,
    EnergyTwistFn,
    GaussianLikelihoodTwistFn,
    LogProbFn,
    NormResidualTwistFn,
    PosteriorPredictiveGaussianTwistFn,
    self_normalized_posterior_expectation,
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
    "CategoricalProjectionCorrectionFn",
    "ClassifierTwistFn",
    "ComposeForwardFn",
    "ConditionalDiffusionSampler",
    "ConvForwardFn",
    "CorrectionFn",
    "DenoiserFn",
    "DiscreteCompositionTwistFn",
    "DiscreteMultiHeadCompositionTwistFn",
    "ESSThresholdedResamplerFn",
    "EndpointTiltCloudTwistFn",
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
    "LogImportanceFn",
    "LogProbFn",
    "LowRankTweediePosteriorCovarianceFn",
    "MultinomialResamplerFn",
    "NoResamplerFn",
    "NormResidualTwistFn",
    "PCAPosteriorCovarianceFn",
    "PosteriorCloudFn",
    "PosteriorCovarianceFn",
    "PosteriorPredictiveGaussianTwistFn",
    "ProjectionCloudCorrectionFn",
    "ProjectionFn",
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
    "batched_minres",
    "call_inference_fn",
    "cfg_denoiser_fn",
    "linear_adjoint",
    "make_cfg_inference_fn",
    "make_denoiser_fn",
    "make_posterior_cloud_fn",
    "miyasawa_scale",
    "normalised_weights",
    "proposal_log_ratio",
    "psd_pinv_solve",
    "scalar_alpha",
    "scalar_alpha_sigma",
    "self_normalized_posterior_expectation",
    "singular_gaussian_logpdf",
    "unit_scale",
]
