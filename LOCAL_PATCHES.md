# Local patches on top of upstream `google/hackable_diffusion`

This fork (`njwfish/hackable_diffusion`, branch `local-mdt`) tracks upstream
`main` with a small set of additive patches needed by the
[multiscale](https://github.com/njwfish/multiscale) project.

Upstream remote: `https://github.com/google/hackable_diffusion`

## Patch inventory

### 1. `SimplicialProcess.corrupt()` — float simplex input support
**File:** `hackable_diffusion/lib/corruption/simplicial.py`

Upstream `corrupt()` assumes integer token input `(*batch, 1)` and converts
to one-hot internally.  Our patch adds a dtype check so that float simplex
vectors `(*batch, K)` are accepted and used directly as the clean direction
for Dirichlet corruption.

**Why:** Multiscale training produces coarse-level data that is a
block-average of one-hot vectors — a true distribution on the simplex, not
a single token.  This cannot be represented as an integer without losing
information (argmax discards the composition structure).

### 2. `DiffusionSampler.return_trajectory` — skip trajectory materialisation
**File:** `hackable_diffusion/lib/sampling/sampling.py`

Adds a `return_trajectory: bool = True` field.  When `False`, `scan_body`
returns `(carry, None)` and the sampler short-circuits before
`_concat_pytree`, returning `(last_step, None)`.

**Why:** 14+ mdt configs and eval scripts set `return_trajectory: false` to
avoid materialising the full step history during inference, which is the
dominant memory cost for long sampling chains.  Default `True` preserves
upstream behaviour.

### 3. `DiT.remat` — per-block gradient checkpointing
**File:** `hackable_diffusion/lib/architecture/dit.py`

Adds a `remat: bool = False` field.  When `True`, each DiT block's forward
pass is wrapped with `nn.remat` so activations are recomputed during the
backward pass instead of stored.  `is_training` is closure-captured as a
static Python bool to avoid `nn.remat` tracing it (which would break
`@kt.typechecked` on the block).

**Why:** The `chunk_dit_dna_moments` config uses a 12-block DiT that OOMs
during training without per-block remat on a single H100.  Default `False`
preserves upstream behaviour.

### 4. Upstream bug fixes (Protocol migration)
**File:** `hackable_diffusion/lib/architecture/simplicial.py`

Two fixes for issues introduced by upstream commit `fd23354` (abc → Protocol
migration):

- **`DenseEmbedder.embedding_dim`:** The field was declared in the
  `BaseLogitEmbedder` ABC but not carried over to the concrete class when
  the ABC became a Protocol.  `__call__` still references
  `self.embedding_dim`, so any config that passes `embedding_dim` crashes.
  Fix: re-declare the field on `DenseEmbedder`.

- **`ConditionalSimplicialBackbone` return annotation:** The return type
  `Float['batch *other V']` reuses the input dimension name `V`, but input
  has `V=1` (token index) while output has `V=num_categories` (logits).
  The old `@typechecked` from jaxtyping was lenient; `@kt.typechecked` from
  kauldron enforces strict dimension-name matching.  Fix: rename return dim
  to `K`.

### 5. New subpackage: `hackable_diffusion.lib.guidance`
**Directory:** `hackable_diffusion/lib/guidance/`

Additive-only subpackage providing a composable conditional-sampling
framework.  Three protocols (`CorrectionFn`, `TwistFn`, `ResamplerFn`)
compose inside `ConditionalDiffusionSampler` to express Pi-GDM, cov-aware,
DPS, TDS, MCGDiff, and other published guidance methods as configurations
over the existing `DiffusionSampler`.

**Why:** upstream `lib/inference/guidance.py` covers only classifier-free
guidance (combining conditional and unconditional outputs).  The broader
family of inverse-problem / posterior-sampling methods -- which all
require shifting the denoiser output or weighting particles by a
log-potential -- is not representable there.  This subpackage provides
the missing abstractions without touching any upstream module.

Key public objects:

- `ConditionalDiffusionSampler`: orchestrator wrapping a `DiffusionSampler`
  with optional `correction_fn`, `twist_fn`, `resampler_fn`.
- `GradientCorrectionFn`: bridges `TwistFn` -> `CorrectionFn` via
  autograd (generalises DPS / Pi-GDM-via-Miyasawa).  Prefactor schedule
  injected as a callable (`miyasawa_prefactor`, `dps_prefactor`).
- `IteratedCorrectionFn`: K inner Kalman sweeps with denoiser
  re-evaluation (closes the intermediate-H non-Gaussian bump).
- `GaussianLikelihoodTwistFn` / `DiscreteCompositionTwistFn`: canonical
  log-potentials for linear-Gaussian and multinomial observations.
- `proposal_log_ratio`: one-line polymorphic dispatcher that calls
  `stepper.kernel(...).log_density_ratio(xt_prev, xt_next)`.  No
  registry and no isinstance checks -- the math lives on each
  stepper's `kernel` method (see patch 6).

### 6. `StepKernel` protocol on `SamplerStep`
**Files:** `hackable_diffusion/lib/sampling/base.py`,
`hackable_diffusion/lib/sampling/gaussian_step_sampler.py`,
`hackable_diffusion/lib/sampling/simplicial_step_sampler.py`

Adds a `StepKernel` Protocol to `lib/sampling/base.py` and a `kernel(...)`
method to every Gaussian stepper (`DDIMStep`, `SdeStep`, `VelocityStep`,
`AdjustedDDIMStep`, `HeunStep`) and to `SimplicialDDIMStep`.  Each
method returns a concrete `StepKernel` (``GaussianStepKernel`` or
``SimplicialStepKernel``) with a single universal operation,
`log_density_ratio(xt_prev, xt_next) -> (B,)`.

**Why:** every Gaussian-forward stepper parameterises its transition as
``xt_next = coeff_x0 xhat_0 + coeff_xt xt + sigma_step eps`` -- one
formula with three scalar schedule-dependent coefficients.  ODE (DDIM
η=0, AdjustedDDIM, Heun) and SDE (SdeStep, DDIM η>0, VelocityStep ε>0)
differ only in whether ``sigma_step`` is zero or positive.  Centering
the kernel primitive lets `lib/guidance/proposal_ratio.py` collapse
from a per-stepper registry (~250 lines, five parallel formulas) to a
three-line polymorphic dispatcher.  ODE stays a noiseless limit of the
same primitive rather than a parallel code path.

**Doing this without modifying upstream.** If you can't add `kernel`
to a stepper class, wrap it -- the polymorphic dispatcher only looks
for a `kernel` attribute on the object you hand in.  For example:

```python
@dataclasses.dataclass(frozen=True)
class _KerneledWrapper:
    base: SamplerStep
    kernel_fn: Callable[..., StepKernel]

    def initialize(self, *args, **kwargs): return self.base.initialize(*args, **kwargs)
    def update(self, *args, **kwargs):     return self.base.update(*args, **kwargs)
    def finalize(self, *args, **kwargs):   return self.base.finalize(*args, **kwargs)
    def kernel(self, **kwargs):            return self.kernel_fn(self.base, **kwargs)

wrapped = _KerneledWrapper(base=third_party_stepper, kernel_fn=my_factory)
sampler = DiffusionSampler(..., stepper=wrapped, ...)
```

No framework-side registry needed; wrapping is how external steppers
opt in.  This patch is only needed because we prefer the primary path
(one method on each upstream stepper) to be the clean one.

### 7. Interpolant refactor of `lib/corruption/`
**Files:** `hackable_diffusion/lib/corruption/base.py`,
`hackable_diffusion/lib/corruption/couplings.py` (new),
`hackable_diffusion/lib/corruption/interpolants.py` (new),
`hackable_diffusion/lib/corruption/targets.py` (new),
`hackable_diffusion/lib/corruption/gaussian.py`,
`hackable_diffusion/lib/corruption/riemannian.py`,
`hackable_diffusion/lib/corruption/interpolant_parity_test.py` (new),
`hackable_diffusion/lib/loss/gaussian.py`,
`hackable_diffusion/lib/sampling/gaussian_step_sampler.py`,
`hackable_diffusion/lib/diffusion_network.py` (docstrings).

Factor the legacy ``GaussianProcess.corrupt`` into three composable
Protocol axes -- ``Coupling``, ``Interpolant``, ``TargetAdapter`` --
and a ``InterpolantProcess`` orchestrator that wires them.  Every
data-to-data generative framework becomes a triple:

    InterpolantProcess(
        coupling    = IndependentCoupling(StandardNormalSource()),   # or OT, or deterministic, or dataloader
        interpolant = LinearInterpolant(schedule=...),               # or geodesic, or stochastic
        targets     = GaussianSourceTargets(),                       # or VelocityOnlyTargets
    )

``GaussianProcess`` and ``RiemannianProcess`` remain as named shim
classes that delegate to internally-built ``InterpolantProcess``
instances; type annotations throughout downstream code keep working
unchanged.

**Why:** ``corrupt`` was fusing three distinct responsibilities --
sampling ``x_1`` given ``x_0``, interpolating ``x_t = I(t, x_0, x_1)``,
and computing the derived targets dict.  Splitting them enables:

- Arbitrary couplings (mini-batch OT, deterministic blur-deblur,
  data-to-data) without touching the interpolation or target logic.
- Arbitrary interpolation paths (linear, geodesic, stochastic with
  ``gamma(t) z`` augmentation) without touching the coupling or
  target logic.
- Target dicts that honestly advertise which parameterisations are
  valid for the chosen source (a non-Gaussian source can't emit
  ``score = -x_1/sigma``; ``VelocityOnlyTargets`` drops that key).

**Global rename: ``epsilon`` -> ``x1``.**

This refactor renames the Gaussian-source noise key from ``epsilon``
to ``x1`` everywhere -- modality-agnostic endpoint naming.  Affected
surfaces:

- ``target_info`` dict emitted by ``corrupt``: ``epsilon`` -> ``x1``.
- ``CONVERTERS`` table in ``targets.py`` (relocated from
  ``gaussian.py``): source and target parameterisation names
  ``epsilon`` -> ``x1``.  Callers of ``process.convert_predictions``
  passing ``{"epsilon": value}`` must update to ``{"x1": value}``.
- ``lib/loss/gaussian.py``: ``GaussianPredictionType`` literal, the
  scaling-function conversion table, and the internal ``_*_to_*``
  function names.
- ``lib/sampling/gaussian_step_sampler.py``: ``AdjustedDDIMStep``
  reads ``prediction_dict["x1"]`` instead of ``["epsilon"]``.
- ``lib/diffusion_network.py``: docstring parameterisation examples.

No aliases, no conditional shims -- the rename is total.  Downstream
consumers training with ``prediction_type="epsilon"`` must update to
``"x1"``.

**Doing this without modifying upstream.** Not easily -- the refactor
touches internal ``CorruptionProcess`` behaviour and the rename
propagates through loss/sampling.  If you must stay off-fork, keep
both names in ``target_info`` and in ``CONVERTERS`` under a wrapping
``CompatCorruptionProcess`` subclass; but that's the kind of
bidirectional-alias plumbing this refactor exists to kill.

## Rebasing on upstream

```bash
cd hackable_diffusion
git fetch upstream
git rebase upstream/main
# Expect minor conflicts in simplicial.py if upstream changes corrupt()
git push origin local-mdt --force-with-lease
cd ..
git add hackable_diffusion
git commit -m "Bump hackable_diffusion submodule to latest upstream"
```
