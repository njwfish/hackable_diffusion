# Composable Guidance Framework

The `lib/guidance/` subpackage is a set of Protocol-based primitives for
conditional sampling.  Every published posterior-sampling method
(Pi-GDM, DPS, TDS, MCGDiff, CFG, classifier guidance, iterated Pi-GDM,
...) is a configuration over these primitives.

[TOC]

## Modality compatibility

The framework supports four diffusion modalities:

| Modality | Corruption process | ``x_0`` shape | Canonical stepper |
| --- | --- | --- | --- |
| Gaussian ODE | ``GaussianProcess`` | ``(B, *euclidean)`` | ``DDIMStep(stoch_coeff=0)``, ``VelocityStep(epsilon=0)``, ``HeunStep``, ``AdjustedDDIMStep`` |
| Gaussian SDE | ``GaussianProcess`` | ``(B, *euclidean)`` | ``DDIMStep(stoch_coeff>0)``, ``SdeStep(churn>0)``, ``VelocityStep(epsilon>0)`` |
| Simplicial | ``SimplicialProcess`` | ``(B, *sites, K)`` on probability simplex | ``SimplicialDDIMStep`` |
| Distributional | any | ``(B, M, *shape)`` ensemble | matches the base modality |

Not every primitive works in every modality.  The table below is the
authoritative compatibility map; each primitive's docstring repeats its
own entry.

| Primitive | Gaussian ODE | Gaussian SDE | Simplicial | Distributional |
| --- | :-: | :-: | :-: | :-: |
| **Orchestrator** | | | | |
| `ConditionalDiffusionSampler` | ✓ | ✓ | ✓ | ✓ |
| **Protocols** | | | | |
| `DenoiserFn`, `ForwardFn`, `ResamplerFn` | ✓ | ✓ | ✓ | ✓ |
| `CorrectionFn`, `TwistFn` | ✓ | ✓ | ✓ | ✓ |
| `PosteriorCovarianceFn` | ✓ | ✓ | ✗ | ✓ |
| **Corrections** | | | | |
| `GradientCorrectionFn` | ✓ | ✓ | ~* | ✓ |
| `PiGDMCorrectionFn` | ✓ | ✓ | ✗** | ~† |
| `IteratedCorrectionFn` | (base) | (base) | (base) | (base) |
| `CFGCorrectionFn` | ✓ | ✓ | ✓ | ✓ |
| `BoundAggregateGuidanceFn` | (project-specific) | (project-specific) | (project-specific) | (project-specific) |
| **Posterior-covariance operators** | | | | |
| `IsotropicPosteriorCovarianceFn` | ✓ | ✓ | ✗ | ~ |
| `FixedPriorPosteriorCovarianceFn` | ✓ | ✓ | ✗ | ~ |
| `TweediePosteriorCovarianceFn` | ✓ | ✓ | ✓ | ✓‡ |
| **Twists** | | | | |
| `GaussianLikelihoodTwistFn` | ✓ | ✓ | ✗ | ✓ |
| `DiscreteCompositionTwistFn` | ✗ | ✗ | ✓ | ✗ |
| `DiscreteMultiHeadCompositionTwistFn` | ✗ | ✗ | ✓ | ✗ |
| `ClassifierTwistFn` | ✓ | ✓ | ✓ | ✓ |
| `EnergyTwistFn` | ✓ | ✓ | ✓ | ✓ |
| **Forward operators** | | | | |
| `LinearForwardFn`, `SubsampleForwardFn`, `InpaintingForwardFn`, `ConvForwardFn`, `ComposeForwardFn` | ✓ | ✓ | ~§ | ✓ |
| **Resamplers** | | | | |
| `NoResamplerFn`, `Systematic/Multinomial/ESSThresholded` | ✓ | ✓ | ✓ | ✓ |
| **Proposal-ratio registry** | | | | |
| `DDIMStep`, `SdeStep`, `VelocityStep`, `AdjustedDDIMStep`, `HeunStep` | ✓ | ✓ | ✗ | ✓ |
| `SimplicialDDIMStep` (churn=0) | ✗ | ✗ | ✓ | ✗ |

Footnotes:

- `*` Gradient on a simplex is a *tangent-space* object; the Euclidean
  gradient step leaves the simplex without projection.  Use a
  simplex-aware correction on simplicial ``x_0``.
- `**` The Kalman update ``x_0 + Sigma Aᵀ (...) r`` assumes a Euclidean
  inner product on ``x_0``.  For simplicial ``x_0`` use a
  logit-space or tangent-space correction.
- `†` Works per-ensemble-member; batched CG solves ``B*M`` systems if
  the distributional denoiser returns an explicit ``(B, M, n)``
  ensemble.  Make sure the ``ForwardFn`` preserves the ensemble axis.
- `‡` JVP through a distributional denoiser sees the ensemble axis
  natively.
- `§` Forward ops with numerical outputs (`LinearForwardFn`, `ConvForwardFn`)
  can operate on simplex-valued ``x_0`` *if* the map itself is defined on
  the simplex (aggregation of probabilities, for example).  The
  ``InpaintingForwardFn`` mask is always modality-agnostic.

## Protocols

Six callable Protocols are the design axes.

| Protocol | Signature | Role |
| --- | --- | --- |
| `DenoiserFn` | `(xt, time) -> xhat_0` | The learned object.  Built from a raw `inference_fn` via `make_denoiser_fn`. |
| `ForwardFn` | `.forward(x) -> y` | Linear (or non-linear) measurement map `A`.  Adjoint free via `jax.vjp`. |
| `PosteriorCovarianceFn` | `(v, *, xt, time, schedule, denoiser_fn) -> Cov v` | Linear operator for the Kalman gain.  Gaussian-modality only. |
| `CorrectionFn` | `(outputs, xt, time, *, schedule, corruption_process, ...) -> outputs` | Modifies the denoiser's outputs before the stepper advances. |
| `TwistFn` | `(xt, time, *, inference_fn, schedule, corruption_process, ...) -> log_psi` | SMC log-potential; gradient source for DPS-style corrections. |
| `ResamplerFn` | `(particles, log_weights, *, rng) -> (particles, log_weights)` | Pure resample. |

## Taxonomy

Every posterior-sampling method reduces to a one-line expression.
Modality tag in brackets.

```python
# --- DPS (Chung et al. 2023) -- Gaussian ODE / SDE, distributional ----
GradientCorrectionFn(
    twist=GaussianLikelihoodTwistFn(observation=y, forward_fn=A),
    prefactor_fn=dps_prefactor,
)

# --- Pi-GDM (Song et al. 2023), cov-aware -- Gaussian ODE / SDE --------
PiGDMCorrectionFn(
    observation=y, forward_fn=A,
    posterior_covariance_fn=FixedPriorPosteriorCovarianceFn(prior_covariance=C),
)

# --- Pi-GDM via Miyasawa/Tweedie -- universal* -------------------------
PiGDMCorrectionFn(
    observation=y, forward_fn=A,
    posterior_covariance_fn=TweediePosteriorCovarianceFn(),
)
# (*) exact for any prior the denoiser represents; still Euclidean-x_0.

# --- Iterated Pi-GDM -- modality of the base ---------------------------
IteratedCorrectionFn(
    base=PiGDMCorrectionFn(..., posterior_covariance_fn=TweediePosteriorCovarianceFn()),
    num_iters=3,
)

# --- TDS (Wu et al. 2023) -- any Gaussian ODE/SDE correction + twist + SMC --
ConditionalDiffusionSampler(
    base_sampler=...,
    correction_fn=any_correction_above,
    twist_fn=GaussianLikelihoodTwistFn(observation=y, forward_fn=A),
    resampler_fn=SystematicResamplerFn(),
    num_particles=K,
)

# --- Classifier guidance -- universal ---------------------------------
GradientCorrectionFn(
    twist=ClassifierTwistFn(log_prob_fn=lambda x0: classifier(x0, y_class)),
)

# --- CFG -- universal -------------------------------------------------
CFGCorrectionFn(
    unconditional_inference_fn=uncond_model,
    guidance_fn=ScalarGuidanceFn(guidance=w),  # lib.inference.guidance
)

# --- Inverse-problem benchmarks -- Gaussian ODE / SDE -----------------
PiGDMCorrectionFn(forward_fn=InpaintingForwardFn(mask=m), ...)
PiGDMCorrectionFn(forward_fn=ConvForwardFn(kernel=k), ...)
PiGDMCorrectionFn(
    forward_fn=ComposeForwardFn(
        first=ConvForwardFn(kernel=blur),
        second=SubsampleForwardFn(indices=stride_idx),
    ), ...,
)

# --- Energy / constraint guidance -- universal ------------------------
GradientCorrectionFn(
    twist=EnergyTwistFn(energy_fn=lambda x0: penalty(x0), temperature=1.0),
)
```

## SMC weight accounting

When a `CorrectionFn` shifts the denoiser output, SMC importance
weights require a per-step increment

```
delta log w = log p_theta(x_s | x_r) - log q(x_s | x_r)
            + log psi(y | x_s) - log psi(y | x_r)
```

where `r = prev, s = next`.  The proposal log-ratio is computed in
closed form by `proposal_log_ratio`, which dispatches by `isinstance`:

| Stepper | Formula | Deterministic limit |
| --- | --- | --- |
| `DDIMStep` | Linear-mean-shift Gaussian | `stoch_coeff = 0` → 0 |
| `SdeStep` | Score-form Euler-Maruyama | `churn = 0` → 0 |
| `VelocityStep` | Velocity-form Euler-Maruyama | `epsilon = 0` → 0 |
| `AdjustedDDIMStep` | Deterministic (ratio = 0) | always 0 |
| `HeunStep` | Deterministic predictor-corrector | always 0 |
| `SimplicialDDIMStep` | Categorical log-prob on sampled token | `churn = 0` required |

Register a new stepper type with

```python
from hackable_diffusion.lib.guidance import register_proposal_ratio
register_proposal_ratio(MyStepType, my_ratio_fn)
```

where `my_ratio_fn` has the uniform signature
`(stepper, corruption_process, outputs_uncorrected, outputs_corrected,
xt_prev, xt_next, time_prev, time_next) -> (K,)`.

## Example: TDS + Pi-GDM on image inpainting

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
y    = mask * x_observed
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
| `guidance_test.py`, `guidance_literature_test.py` | 63 pure unit + literature-validation tests |
