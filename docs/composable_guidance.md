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
| `CategoricalProjectionCorrectionFn` | ✗ | ✗ | ✓ | ✗ |
| `IteratedCorrectionFn` | (base) | (base) | (base) | (base) |
| `LinearBlendDenoiserFn` / `make_cfg_inference_fn` | ✓ | ✓ | ✓ | ✓ |
| **Posterior-covariance operators** | | | | |
| `IsotropicPosteriorCovarianceFn` | ✓ | ✓ | ✗ | ~ |
| `FixedPriorPosteriorCovarianceFn` | ✓ | ✓ | ✗ | ~ |
| `PCAPosteriorCovarianceFn` | ✓ | ✓ | ✗ | ~ |
| `TweediePosteriorCovarianceFn` | ✓ | ✓ | ✓ | ✓‡ |
| `LowRankTweediePosteriorCovarianceFn` | ✓ | ✓ | ✓ | ✓‡ |
| **Twists** | | | | |
| `GaussianLikelihoodTwistFn` | ✓ | ✓ | ✗ | ✓ |
| `PosteriorPredictiveGaussianTwistFn` | ✓ | ✓ | ✗ | ✓ |
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
# Self-normalising via the residual-norm twist; pair with the Tweedie-
# scaled gradient correction.
GradientCorrectionFn(
    twist=NormResidualTwistFn(observation=y, forward_fn=A),
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

# --- Hard / clean-endpoint conditioning ``A x_0 = y`` (inpainting, --
# --- exact-projection super-resolution) -- Gaussian ODE / SDE --------
# Singular-Gaussian branch: ``observation_noise = 0`` plus ``solver="pinv"``
# plus ``Cov = I`` (``IsotropicPosteriorCovarianceFn(scale_fn=unit_scale)``)
# collapses the Kalman update to the exact affine projection
#     x0_new = x0 + A^T (A A^T)^+ (y - A x0).
# For an inpainting mask this is bit-for-bit ``mask * y + (1 - mask) * x0``.
KalmanCorrectionFn(
    observation=y, forward_fn=A,
    posterior_covariance_fn=IsotropicPosteriorCovarianceFn(scale_fn=unit_scale),
    observation_noise=0.0,
    solver="pinv",
)
# Companion twist for SMC weighting (singular Gaussian on affine support):
PosteriorPredictiveGaussianTwistFn(
    observation=y, forward_fn=A,
    posterior_covariance_fn=IsotropicPosteriorCovarianceFn(scale_fn=unit_scale),
    schedule=schedule, observation_noise=0.0, enforce_support=True,
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

## Example: image inpainting on a clean endpoint

Conditioning on a *clean* partial observation -- you know ``x_0`` exactly
at some pixels and want the rest filled in -- is the
``observation_noise = 0`` branch.  With ``Cov = I``
(``IsotropicPosteriorCovarianceFn(scale_fn=unit_scale)``) the Kalman
update degenerates to the exact affine projection
``x0 + A^T (A A^T)^+ (y - A x0)``, which for an inpainting mask is
``mask * y + (1 - mask) * x0``: observed pixels are clamped to ``y``,
free pixels are left to the denoiser.  The step kernel then re-noises
the clamped ``x_0`` to ``x_{t-1}`` so observed coordinates ride the
clean forward process all the way down to ``t = 0``.

```python
from hackable_diffusion.lib.guidance import (
    ConditionalDiffusionSampler,
    InpaintingForwardFn,
    IsotropicPosteriorCovarianceFn,
    KalmanCorrectionFn,
    PosteriorPredictiveGaussianTwistFn,
    SystematicResamplerFn,
    unit_scale,
)

mask = ...                      # (H, W) of 0/1; 1 = observed
y    = mask * x_clean           # clean-endpoint observation
fwd  = InpaintingForwardFn(mask=mask)
cov  = IsotropicPosteriorCovarianceFn(scale_fn=unit_scale)  # ``C = I``

correction = KalmanCorrectionFn(
    observation=y, forward_fn=fwd,
    posterior_covariance_fn=cov,
    observation_noise=0.0,           # hard constraint
    solver="pinv",                   # rank-deficient hard-projection path
)
twist = PosteriorPredictiveGaussianTwistFn(
    observation=y, forward_fn=fwd,
    posterior_covariance_fn=cov,
    schedule=corruption.schedule,
    observation_noise=0.0,
    enforce_support=True,            # delta-on-affine-support log psi
)

# Seed x_T consistent with the clean endpoint on observed coords:
#   x_T = mask * (alpha_T x_clean + sigma_T eps) + (1 - mask) * eps_free.
alpha_T = corruption.schedule.alpha(jnp.asarray([1.0]))
sigma_T = corruption.schedule.sigma(jnp.asarray([1.0]))
eps      = jax.random.normal(rng, x_clean.shape)
init     = mask * (alpha_T * x_clean + sigma_T * eps) + (1.0 - mask) * eps

sampler = ConditionalDiffusionSampler(
    base_sampler=ddim_sampler,
    corruption_process=corruption,
    correction_fn=correction,
    twist_fn=twist,                  # omit for non-SMC; use any sampler
    resampler_fn=SystematicResamplerFn(),
)
final_step, _, log_weights = sampler(
    inference_fn=model, rng=rng, initial_noise=init,
)
```

**Soft-noise variant.** For noisy partial observations switch
``KalmanCorrectionFn`` to ``solver="cg"`` (or ``"minres"`` if the
covariance can be indefinite) with ``observation_noise > 0``, and
pair with ``GaussianLikelihoodTwistFn``.  Use
``TweediePosteriorCovarianceFn()`` (or another learned-prior variant)
for ``Cov``.  The wiring is otherwise identical -- the choice of
correction / covariance / twist is independent of the rest of the
pipeline.

## Files

| File | Contents |
| --- | --- |
| `protocols.py` | `DenoiserFn`, `ForwardFn`, `PosteriorCovarianceFn`, `CorrectionFn`, `TwistFn`, `ResamplerFn` |
| `utils.py` | `make_denoiser_fn`, `call_inference_fn`, schedule helpers |
| `linalg.py` | `batched_cg`, `batched_minres`, `batch_inner`, `linear_adjoint`, `randomized_svd_jvp` |
| `resamplers.py` | `NoResamplerFn`, `Systematic` / `Multinomial` / `ESSThresholded` resamplers |
| `denoisers.py` | `make_denoiser_fn`, `LinearBlendDenoiserFn`, `cfg_denoiser_fn`, `make_cfg_inference_fn` |
| `corrections.py` | `KalmanCorrectionFn` (solver='pinv'\|'cg'\|'minres'), `GradientCorrectionFn`, `IteratedCorrectionFn`, `CategoricalProjectionCorrectionFn` |
| `posterior_covariance.py` | `Isotropic` / `FixedPrior` / `PCA` / `Tweedie` / `LowRankTweedie` covariance operators, `miyasawa_scale`, `unit_scale` |
| `gaussian_conditioning.py` | `psd_pinv_solve`, `singular_gaussian_logpdf`, `_materialize_observation_covariance` -- linalg helpers for the singular-Gaussian branch |
| `twists.py` | `GaussianLikelihood` / `PosteriorPredictiveGaussian` / `NormResidual` / `DiscreteComposition` / `Classifier` / `Energy` twists |
| `forward_ops.py` | `Linear` / `Subsample` / `Inpainting` / `Conv` / `Compose` forward operators |
| `proposal_ratio.py` | Per-stepper closed-form proposal-ratio dispatch |
| `sampler.py` | `ConditionalDiffusionSampler` (the orchestrator) |
| `guidance_test.py`, `guidance_literature_test.py` | Pure unit + literature-validation tests |
