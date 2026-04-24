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
| `KalmanCorrectionFn` | ✓ | ✓ | ✗** | ~† |
| `IteratedCorrectionFn` | (base) | (base) | (base) | (base) |
| `LinearBlendDenoiserFn` / `make_cfg_inference_fn` | ✓ | ✓ | ✓ | ✓ |
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
| **Step kernels (via `stepper.kernel()`)** | | | | |
| `GaussianStepKernel` (DDIM, Sde, Velocity, AdjustedDDIM, Heun) | ✓ | ✓ | ✗ | ✓ |
| `SimplicialStepKernel` (`SimplicialDDIMStep`, churn=0) | ✗ | ✗ | ✓ | ✗ |

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

Seven callable Protocols are the design axes.

| Protocol | Signature | Role |
| --- | --- | --- |
| `DenoiserFn` | `(xt) -> xhat_0` | Pure soft-x0 closure at fixed (time, cond, rng).  Built via `make_denoiser_fn`. |
| `ForwardFn` | `.forward(x) -> y` | Measurement map `A`.  Adjoint free via `jax.vjp`. |
| `PosteriorCovarianceFn` | `(v, *, xt, time, schedule, denoiser_fn) -> Cov v` | Linear operator for the Kalman gain.  Gaussian-modality only. |
| `CorrectionFn` | `(x0, xt, time, *, denoiser_fn, schedule) -> x0_new` | Observation-driven x0 shift. |
| `TwistFn` | `(xt, time, *, denoiser_fn) -> log_psi` | SMC log-potential; gradient source for DPS-style corrections. |
| `ResamplerFn` | `(particles, log_weights, *, rng) -> (particles, log_weights)` | Pure resample. |
| `StepKernel` | `.log_density_ratio(xt_prev, xt_next) -> (B,)` | Transition-kernel log-density ratio under shifted xhat_0; each `SamplerStep` builds a concrete kernel via `stepper.kernel(...)`. |

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
KalmanCorrectionFn(
    observation=y, forward_fn=A,
    posterior_covariance_fn=FixedPriorPosteriorCovarianceFn(prior_covariance=C),
)

# --- Pi-GDM via Miyasawa/Tweedie -- universal* -------------------------
KalmanCorrectionFn(
    observation=y, forward_fn=A,
    posterior_covariance_fn=TweediePosteriorCovarianceFn(),
)
# (*) exact for any prior the denoiser represents; still Euclidean-x_0.

# --- Iterated Pi-GDM -- modality of the base ---------------------------
IteratedCorrectionFn(
    base=KalmanCorrectionFn(..., posterior_covariance_fn=TweediePosteriorCovarianceFn()),
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

# --- CFG -- universal -- denoiser composition, not a correction ------
# Preferred entry point: blend at the inference_fn level.
inference_fn = make_cfg_inference_fn(
    conditional_inference_fn=cond_model,
    unconditional_inference_fn=uncond_model,
    guidance_fn=ScalarGuidanceFn(guidance=w),  # lib.inference.guidance
)
# ... pass ``inference_fn`` to ConditionalDiffusionSampler.
# Lower-level DenoiserFn compositions:
#   cfg_denoiser_fn(cond_denoiser, uncond_denoiser, guidance=w)
#   LinearBlendDenoiserFn(denoisers=(d1, d2, ...), weights=(w1, w2, ...))

# --- Inverse-problem benchmarks -- Gaussian ODE / SDE -----------------
KalmanCorrectionFn(forward_fn=InpaintingForwardFn(mask=m), ...)
KalmanCorrectionFn(forward_fn=ConvForwardFn(kernel=k), ...)
KalmanCorrectionFn(
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

where `r = prev, s = next`.  The proposal log-ratio math lives on each
stepper's `kernel(...)` method, which returns a concrete `StepKernel`
(`GaussianStepKernel` or `SimplicialStepKernel`).  `proposal_log_ratio`
is a three-line polymorphic dispatcher -- no registry, no isinstance
checks.

### Universal Gaussian kernel

Every Gaussian-forward stepper (DDIM, SdeStep, VelocityStep,
AdjustedDDIMStep, HeunStep) parameterises its reverse-time transition
as

```
xt_next = coeff_x0 * xhat_0 + coeff_xt * xt + sigma_step * eps
```

with three scalar schedule-dependent coefficients
`(coeff_x0, coeff_xt, sigma_step)`.  ODE vs. SDE is the single knob
`sigma_step`:

- `sigma_step = 0`: deterministic proposal (ODE / probability flow).
  Log-ratio identically zero.
- `sigma_step > 0`: stochastic proposal (Euler-Maruyama / DDPM
  ancestral).  Log-ratio is the quadratic Gaussian form.

`GaussianStepKernel.log_density_ratio(xt_prev, xt_next)` computes the
universal formula at the kernel's stored coefficients.

| Stepper | `sigma_step` | Determined by |
| --- | --- | --- |
| `DDIMStep` | `sigma_s * eta` | `stoch_coeff = eta` |
| `SdeStep` | `sqrt(dt) * g * churn` | `churn` |
| `VelocityStep` | `sqrt(dt) * g * epsilon` | `epsilon` |
| `AdjustedDDIMStep` | `0` | (deterministic by construction) |
| `HeunStep` | `0` | (predictor-corrector, no noise term) |

### Simplicial kernel

`SimplicialDDIMStep.kernel(...)` returns a `SimplicialStepKernel` at
`churn = 0`: the transition mixture reduces to a categorical draw
against `softmax(logits)`, and the log-density ratio at a shifted
`xhat_0` collapses to the per-site categorical log-prob difference at
the sampled token (the beta-weight factor cancels).  `churn > 0`
raises -- the full Dirichlet-shrinkage kernel is not yet derived.

### Adding a new stepper

Implement `kernel` on the stepper class, returning either a built-in
`StepKernel` or your own.  Example for a hypothetical
`MyExoticStep`:

```python
@dataclasses.dataclass(frozen=True, kw_only=True)
class MyExoticStep(SamplerStep):
    ...

    def kernel(self, *, prediction_uncorrected, prediction_corrected,
               xt, time_prev, time_next) -> StepKernel:
        return GaussianStepKernel(
            coeff_x0=..., coeff_xt=..., sigma_step=...,
            x0_uncorrected=..., x0_corrected=...,
        )
```

### External steppers (not owned by you)

If you can't modify a third-party `SamplerStep`, wrap it:

```python
@dataclasses.dataclass(frozen=True)
class _KerneledWrapper:
    base: SamplerStep
    kernel_factory: Callable[..., StepKernel]

    def initialize(self, *a, **kw): return self.base.initialize(*a, **kw)
    def update(self, *a, **kw):     return self.base.update(*a, **kw)
    def finalize(self, *a, **kw):   return self.base.finalize(*a, **kw)
    def kernel(self, **kw):         return self.kernel_factory(self.base, **kw)

wrapped = _KerneledWrapper(base=third_party_stepper, kernel_factory=my_factory)
```

Hand the wrapped object to `DiffusionSampler` / `ConditionalDiffusionSampler`.
The polymorphic dispatcher only looks for a `kernel` attribute on the
stepper it receives; there's no framework-side registry.

## Example: TDS + Pi-GDM on image inpainting

```python
from hackable_diffusion.lib.guidance import (
    ConditionalDiffusionSampler,
    InpaintingForwardFn,
    KalmanCorrectionFn,
    TweediePosteriorCovarianceFn,
    GaussianLikelihoodTwistFn,
    SystematicResamplerFn,
)

mask = ...   # (H, W) of 0/1
y    = mask * x_observed
fwd  = InpaintingForwardFn(mask=mask)

correction = KalmanCorrectionFn(
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
| `denoisers.py` | `make_denoiser_fn`, `LinearBlendDenoiserFn`, `cfg_denoiser_fn`, `make_cfg_inference_fn` |
| `corrections.py` | `KalmanCorrectionFn`, `IteratedCorrectionFn`, `GradientCorrectionFn`, prefactors |
| `posterior_covariance.py` | `Isotropic` / `FixedPrior` / `Tweedie` covariance operators |
| `twists.py` | `GaussianLikelihood` / `DiscreteComposition` / `Classifier` / `Energy` twists |
| `forward_ops.py` | `Linear` / `Subsample` / `Inpainting` / `Conv` / `Compose` forward operators |
| `proposal_ratio.py` | Per-stepper closed-form ratios and the dispatch registry |
| `adapters.py` | `BoundAggregateGuidanceFn` (legacy-shape project guidance) |
| `sampler.py` | `ConditionalDiffusionSampler` (the orchestrator) |
| `guidance_test.py`, `guidance_literature_test.py` | 63 pure unit + literature-validation tests |
