# Composable Guidance Framework

The `lib/guidance/` subpackage exposes a small set of Protocol-based
primitives that compose to express every posterior-sampling / guidance
method in the published diffusion literature.  The framework is strictly
additive -- it complements `lib/inference/guidance.py` (classifier-free
guidance combinators) rather than replacing it.

[TOC]

## Design

Six Protocols cover every axis of variation:

| Protocol | Role | Example implementations |
| --- | --- | --- |
| `DenoiserFn` | Pure `xt -> xhat_0(xt)` closure at fixed `(time, cond, rng)` | built from `make_denoiser_fn` |
| `ForwardFn` | Linear / non-linear measurement `y = A(x_0)`; adjoint via VJP | `LinearForwardFn`, `InpaintingForwardFn`, `ConvForwardFn`, `SubsampleForwardFn`, `ComposeForwardFn` |
| `PosteriorCovarianceFn` | Linear operator `v -> Cov(x_0\|x_t) v` | `IsotropicPosteriorCovarianceFn`, `FixedPriorPosteriorCovarianceFn`, `TweediePosteriorCovarianceFn` |
| `CorrectionFn` | `outputs -> outputs` denoiser-side correction | `PiGDMCorrectionFn`, `GradientCorrectionFn`, `IteratedCorrectionFn`, `CFGCorrectionFn`, `BoundAggregateGuidanceFn` |
| `TwistFn` | `(xt, t) -> log psi(y\|xt)` SMC log-potential | `GaussianLikelihoodTwistFn`, `DiscreteCompositionTwistFn`, `ClassifierTwistFn`, `EnergyTwistFn` |
| `ResamplerFn` | `(particles, log_w) -> (particles, log_w)` | `NoResamplerFn`, `SystematicResamplerFn`, `MultinomialResamplerFn`, `ESSThresholdedResamplerFn` |

`ConditionalDiffusionSampler` wraps a `DiffusionSampler` with any
combination of `(correction_fn, twist_fn, resampler_fn)`.  The `K=1`
case with no correction / twist / resampler delegates to the base
sampler bit-for-bit, so adding the wrapper to an existing pipeline is
zero-cost when unused.

## Taxonomy

Every posterior-sampling method reduces to a one-line expression:

```python
# --- DPS (Chung et al. 2023) ------------------------------------------
GradientCorrectionFn(
    twist=GaussianLikelihoodTwistFn(observation=y, forward_fn=A),
    prefactor_fn=dps_prefactor,
)

# --- Pi-GDM (Song et al. 2023), cov-aware variant --------------------
PiGDMCorrectionFn(
    observation=y, forward_fn=A,
    posterior_covariance_fn=FixedPriorPosteriorCovarianceFn(prior_covariance=C),
)

# --- Pi-GDM via Miyasawa/Tweedie -- exact under any prior ------------
PiGDMCorrectionFn(
    observation=y, forward_fn=A,
    posterior_covariance_fn=TweediePosteriorCovarianceFn(),
)

# --- Iterated Pi-GDM -- closes the non-Gaussian intermediate-H bump --
IteratedCorrectionFn(
    base=PiGDMCorrectionFn(
        observation=y, forward_fn=A,
        posterior_covariance_fn=TweediePosteriorCovarianceFn(),
    ),
    num_iters=3,
)

# --- TDS (Wu et al. 2023) -- any correction + twist + SMC ------------
ConditionalDiffusionSampler(
    base_sampler=...,
    correction_fn=any_correction_above,
    twist_fn=GaussianLikelihoodTwistFn(observation=y, forward_fn=A),
    resampler_fn=SystematicResamplerFn(),
    num_particles=K,
)

# --- Classifier guidance (Dhariwal & Nichol 2021) --------------------
GradientCorrectionFn(
    twist=ClassifierTwistFn(log_prob_fn=lambda x0: classifier(x0, y_class)),
)

# --- Classifier-free guidance (Ho & Salimans 2022) -------------------
CFGCorrectionFn(
    unconditional_inference_fn=uncond_model,
    guidance_fn=ScalarGuidanceFn(guidance=w),  # lib.inference.guidance
)

# --- Inverse-problem benchmarks --------------------------------------
# Image inpainting:
PiGDMCorrectionFn(forward_fn=InpaintingForwardFn(mask=m), ...)
# Gaussian-blur deblurring:
PiGDMCorrectionFn(forward_fn=ConvForwardFn(kernel=k), ...)
# Super-resolution (blur + downsample):
PiGDMCorrectionFn(
    forward_fn=ComposeForwardFn(
        first=ConvForwardFn(kernel=blur),
        second=SubsampleForwardFn(indices=stride_idx),
    ),
    ...,
)

# --- Energy / constraint guidance ------------------------------------
GradientCorrectionFn(
    twist=EnergyTwistFn(energy_fn=lambda x0: penalty(x0), temperature=1.0),
)
```

## SMC weight accounting

When a `CorrectionFn` shifts the denoiser output, the proposal density
is no longer the base sampler's transition; to stay unbiased under SMC,
we accumulate a per-step importance-weight increment

```
delta log w(t -> s) = log p_theta(x_s | x_t) - log q(x_s | x_t)
                    + log psi(y | x_s) - log psi(y | x_t)
```

The proposal log-ratio `log p_theta - log q` is computed in closed form
by `proposal_log_ratio`, which dispatches by `isinstance` against a
registry of stepper types.  Pre-registered:

| Stepper | Formula | Notes |
| --- | --- | --- |
| `DDIMStep` | Linear-mean-shift Gaussian | Returns 0 at `stoch_coeff=0` (deterministic) |
| `SdeStep` | Score-form Euler-Maruyama; B = 0.5 g² (1+churn²) dt α/σ² | Returns 0 at `churn=0` |
| `VelocityStep` | Velocity-form Euler-Maruyama | Returns 0 at `epsilon=0` |
| `AdjustedDDIMStep` | Deterministic (ratio = 0) | Heuristic: uniform-weight SMC |
| `HeunStep` | Deterministic predictor-corrector (ratio = 0) | Same |
| `SimplicialDDIMStep` | Categorical log-prob ratio on sampled token | Requires `churn=0` |

Register a new stepper type with

```python
from hackable_diffusion.lib.guidance import register_proposal_ratio
register_proposal_ratio(MyStepType, my_ratio_fn)
```

where `my_ratio_fn` matches the uniform signature
`(stepper, corruption_process, outputs_uncorrected, outputs_corrected,
xt_prev, xt_next, time_prev, time_next) -> (K,)`.

## Example: end-to-end TDS-Pi-GDM on an image inpainting task

```python
from hackable_diffusion.lib.guidance import (
    ConditionalDiffusionSampler,
    InpaintingForwardFn,
    PiGDMCorrectionFn,
    TweediePosteriorCovarianceFn,
    GaussianLikelihoodTwistFn,
    SystematicResamplerFn,
)

mask = ...   # (H, W) of 0/1
y    = mask * x_observed  # partial observation
fwd  = InpaintingForwardFn(mask=mask)

correction = PiGDMCorrectionFn(
    observation=y, forward_fn=fwd,
    posterior_covariance_fn=TweediePosteriorCovarianceFn(),
    observation_noise=0.05,
)
twist = GaussianLikelihoodTwistFn(
    observation=y, forward_fn=fwd, observation_noise=0.05,
)
sampler = ConditionalDiffusionSampler(
    base_sampler=ddim_sampler,
    corruption_process=corruption,
    correction_fn=correction,
    twist_fn=twist,
    resampler_fn=SystematicResamplerFn(),
    num_particles=32,
)
final_step, _, log_weights = sampler(
    inference_fn=model, rng=rng, initial_noise=init,
)
```

## Files

| File | Contents |
| --- | --- |
| `protocols.py` | `DenoiserFn`, `ForwardFn`, `PosteriorCovarianceFn`, `CorrectionFn`, `TwistFn`, `ResamplerFn` |
| `utils.py` | `make_denoiser_fn`, `replace_x0`, `call_inference_fn`, schedule helpers |
| `linalg.py` | `batched_cg`, `batch_inner`, `linear_adjoint` |
| `resamplers.py` | The four resampler implementations |
| `corrections.py` | `PiGDMCorrectionFn`, `IteratedCorrectionFn`, `GradientCorrectionFn`, prefactors |
| `posterior_covariance.py` | `Isotropic` / `FixedPrior` / `Tweedie` covariance operators |
| `twists.py` | `GaussianLikelihood` / `DiscreteComposition` / `Classifier` / `Energy` twists |
| `forward_ops.py` | `Linear` / `Subsample` / `Inpainting` / `Conv` / `Compose` forward operators |
| `proposal_ratio.py` | Per-stepper closed-form ratios and the dispatch registry |
| `adapters.py` | `BoundAggregateGuidanceFn`, `CFGCorrectionFn` |
| `sampler.py` | `ConditionalDiffusionSampler` (the orchestrator) |
| `guidance_test.py` | 56 pure unit tests (no external project deps) |
