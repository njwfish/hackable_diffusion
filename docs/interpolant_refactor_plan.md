# Interpolant Refactor Plan

Separate the (source distribution, coupling, interpolation path, target emission) concerns currently fused inside `GaussianProcess.corrupt` / `RiemannianProcess.corrupt`, so that data-to-data generative frameworks become first-class without regressing any existing code.

**The three required endpoints of this program:**

1. **M1-M2** — factor the existing `GaussianProcess` / `RiemannianProcess` into composable `(Coupling, Interpolant, TargetAdapter)` triples, with byte-exact preservation of today's behavior.
2. **M4** — full mini-batch optimal-transport coupling via `ott-jax`, supporting Sinkhorn and Hungarian matching, with well-defined jit / pmap / vmap semantics. **This is a required deliverable, not optional.** OT flow matching is the dominant data-to-data training recipe in current practice (Tong et al. 2024, Pooladian et al. 2023) and we want it natively supported.
3. **M5** — full stochastic-interpolant implementation in the sense of Albergo–Boffe–Vanden-Eijnden (2023): `x_t = I(t, x_0, x_1) + γ(t) z` with an independent Brownian-like augmentation `z ~ N(0, I)`, a dual-head velocity + score-of-ρ_t target parameterization, and a new SDE integrator that uses both heads. **This is a required deliverable, not optional.**

Because M4 and M5 are required, §10 below is devoted to the design questions each one forces open. These questions should be resolved *before* the colleague starts writing M1 code, because several of them influence the M1 protocol signatures.

**Non-goal:** changing any outputs of the existing Gaussian / Riemannian code paths. Every byte of `target_info`, every sample from `sample_from_invariant`, every trained model's loss value must be bitwise identical before and after M1-M2.

**Non-goal (M1-M3 only):** touching `lib/sampling/`, `lib/guidance/`, `lib/loss/`, or `lib/architecture/`. M5 adds exactly one new stepper class in `lib/sampling/gaussian_step_sampler.py` — no other sampling-layer changes are authorized anywhere in the program.

---

## 1. Motivation

Today, `GaussianProcess.corrupt` (see `lib/corruption/gaussian.py:199-220`) fuses three distinct responsibilities:

```python
def corrupt(self, key, x0, time):
  epsilon = self.sample_from_invariant(key, data_spec=x0)   # (1) coupling: sample x_1 | x_0
  xt = alpha(t) * x0 + sigma(t) * epsilon                   # (2) interpolation: I(t, x_0, x_1)
  target_info = {                                            # (3) target emission: derived quantities
      'x0': x0, 'epsilon': epsilon,
      'score': -epsilon / sigma,
      'velocity': alpha_der * x0 + sigma_der * epsilon,
      'v': alpha * epsilon - sigma * x0,
  }
  return xt, target_info
```

The identities `score = -epsilon/sigma` and `v = alpha*epsilon - sigma*x0` are only valid because `x_1 ~ N(0, I)` — a source assumption baked implicitly into the target dict.

Factoring the three concerns apart gives us:

- Plug-in arbitrary sources on the `x_1` endpoint (images, masked data, blurred data, a second modality).
- Plug-in arbitrary couplings `π(x_0, x_1)` (independent, mini-batch OT, deterministic pair, user-supplied).
- Plug-in arbitrary interpolation paths (linear, trigonometric, geodesic, SI with noise augmentation).
- Target dicts that honestly advertise *which* parameterizations are valid for the chosen source.

---

## 2. Design: three protocols

All live in `lib/corruption/base.py` alongside the existing `CorruptionProcess` protocol.

### 2.1 `Coupling`

```python
class Coupling(Protocol):
  """Samples x_1 given a batch of x_0.

  MUST operate on whole batches: OT-type couplings are inherently set
  operations and cannot be vmapped per-sample. Implementations that
  happen to be per-sample (IndependentCoupling) simply ignore the
  cross-batch structure and remain vmap-friendly.
  """

  def sample(self, key: PRNGKey, x0: DataTree) -> DataTree: ...
```

**Concrete classes** (new file `lib/corruption/couplings.py`):

- `IndependentCoupling(source: Source)` — draws `x_1 ~ source` independently of `x_0`. Pure per-sample; trivially vmappable.
- `DeterministicCoupling(map_fn: Callable)` — `x_1 = map_fn(x_0)`. Pure per-sample; trivially vmappable.
- `MiniBatchOTCoupling(source, cost_fn, ...)` — mini-batch optimal transport pairing. Batch-level; NOT per-sample vmappable. Deferred to milestone 4.

### 2.2 `Source`

```python
class Source(Protocol):
  """A marginal distribution over the x_1 endpoint.

  Used by IndependentCoupling. For data-to-data pipelines where x_1 is
  drawn from a dataloader, the Source wraps the dataloader.
  """

  def sample(self, key: PRNGKey, data_spec: DataTree) -> DataTree: ...
  def is_standard_normal(self) -> bool: ...  # enables GaussianSourceTargets
```

**Concrete classes:**

- `StandardNormalSource()` — `jax.random.normal(key, data_spec.shape)`. Byte-equivalent to today's `sample_from_invariant` in `GaussianProcess`.
- `UniformManifoldSource(manifold)` — equivalent to today's `RiemannianProcess.sample_from_invariant`.
- `DataloaderSource(queue)` — pulls a pre-batched tensor supplied by the training loop.

### 2.3 `Interpolant`

```python
class Interpolant(Protocol):
  """Deterministic path from x_0 to x_1, with optional noise augmentation γ(t) z.

  Full stochastic-interpolant form:
      x_t = I(t, x_0, x_1) + γ(t) z,       z ~ N(0, I)

  When the concrete interpolant has no noise augmentation, `needs_noise`
  is False and `eval` is called with z=None. When `needs_noise` is True,
  the `InterpolantProcess` is responsible for drawing z and passing it in.
  """

  schedule: Schedule                                  # forwarded to samplers that peek
  needs_noise: ClassVar[bool]                          # trace-time dispatch; see §3.3

  def eval(
      self,
      x0: DataTree,
      x1: DataTree,
      t: TimeArray,
      z: DataTree | None = None,
  ) -> tuple[DataTree, DataTree]: ...
  # returns (x_t, dx_t/dt)
```

**Concrete classes** (new file `lib/corruption/interpolants.py`):

- `LinearInterpolant(schedule: GaussianSchedule)` — `I(t, x_0, x_1) = α(t) x_0 + σ(t) x_1`. `needs_noise = False`. Byte-equivalent to today's Gaussian interpolation.
- `GeodesicInterpolant(manifold, schedule: RiemannianSchedule)` — `needs_noise = False`. Byte-equivalent to today's Riemannian interpolation.
- `StochasticInterpolant(alpha, beta, gamma)` — `x_t = α(t) x_0 + β(t) x_1 + γ(t) z`. `needs_noise = True`. Enforces `γ(0) = γ(1) = 0` at construction via a symbolic or sampled check (see §10.5). Shipped in M5.
- `TrigonometricInterpolant` — optional, future work.

### 2.4 `TargetAdapter`

```python
class TargetAdapter(Protocol):
  """Emits the target_info dict from (x_0, x_1, z, x_t, t, interpolant).

  Different adapters are valid for different sources. A Gaussian-source
  adapter emits {x0, x1, epsilon=x1, score, velocity, v}; a generic
  adapter emits only {x0, x1, velocity}. The choice is made at
  construction time and typechecked at wire-up; downstream steppers
  that need keys the adapter doesn't emit fail fast.
  """

  def emit(self, *, x0, x1, z, xt, t, interpolant) -> TargetInfo: ...
  def convert(self, *, prediction, xt, t, interpolant) -> TargetInfo: ...
  # convert replaces GaussianProcess.convert_predictions
```

**Concrete classes** (new file `lib/corruption/targets.py`):

- `GaussianSourceTargets()` — emits `{x0, x1, epsilon=x1, score, velocity, v}`. Owns the `CONVERTERS` table currently hard-coded in `gaussian.py:268+`. Only valid when the source `is_standard_normal()`.
- `VelocityOnlyTargets()` — emits `{x0, x1, velocity}`. Valid for any source. Needed for data-to-data.
- `RiemannianVelocityTargets()` — emits `{x0, x1, velocity}` with manifold-aware velocity. Byte-equivalent to today's `RiemannianProcess` target dict.

### 2.5 The composed process

`lib/corruption/base.py` gains:

```python
@dataclasses.dataclass(kw_only=True, frozen=True)
class InterpolantProcess(CorruptionProcess):
  coupling:    Coupling
  interpolant: Interpolant
  targets:     TargetAdapter

  @property
  def schedule(self):
    return self.interpolant.schedule

  def corrupt(self, key, x0, time):
    # Key splitting is conditional on the interpolant needing z.  This is
    # essential for byte-exact preservation: LinearInterpolant.needs_noise
    # is False, so StandardNormalSource receives the original `key`
    # unchanged, and RNG consumption exactly matches today's
    # GaussianProcess.corrupt.  See §3.3 and §10.1.
    if self.interpolant.needs_noise:
      key_coupling, key_z = jax.random.split(key)
      z = jax.random.normal(key_z, shape=x0.shape)
    else:
      key_coupling, z = key, None
    x1 = self.coupling.sample(key_coupling, x0)
    xt, _ = self.interpolant.eval(x0, x1, time, z)
    target_info = self.targets.emit(
        x0=x0, x1=x1, z=z, xt=xt, t=time, interpolant=self.interpolant,
    )
    return xt, target_info

  def sample_from_invariant(self, key, data_spec):
    # Preserve public API. Delegates to the coupling's source when defined,
    # otherwise raises — the legacy name maps to "sample x_1 given nothing".
    return self.coupling.sample_marginal(key, data_spec)

  def convert_predictions(self, prediction, xt, time):
    return self.targets.convert(prediction=prediction, xt=xt, t=time,
                                interpolant=self.interpolant)

  def get_schedule_info(self, time):
    return self.schedule.evaluate(time)
```

`GaussianProcess` and `RiemannianProcess` are kept as **shim constructors** that build the equivalent `InterpolantProcess`:

```python
def GaussianProcess(*, schedule):                     # same public signature
  return InterpolantProcess(
      coupling    = IndependentCoupling(source=StandardNormalSource()),
      interpolant = LinearInterpolant(schedule=schedule),
      targets     = GaussianSourceTargets(),
  )
```

All existing call sites continue to work unchanged.

---

## 3. Efficiency and vmap contract (critical)

The refactor must preserve today's performance characteristics. Enforce the following as acceptance criteria:

### 3.1 No new Python-level branching in hot paths

All dispatch between concrete `Coupling` / `Interpolant` / `TargetAdapter` types happens at **trace time** (construction of the dataclass). Inside `corrupt`, `eval`, `emit`, `convert` there is no `isinstance`, no dict lookup, no Python conditional that depends on input arrays. The JAX trace must inline the concrete implementations exactly as `GaussianProcess.corrupt` does today.

### 3.2 Vmap-compatibility tiers, declared per coupling

```python
class Coupling(Protocol):
  is_batch_level: ClassVar[bool]    # False => safe to vmap over batch dim
```

- `IndependentCoupling`, `DeterministicCoupling`: `is_batch_level = False`. Each `.sample` call is equivalent to vmapping a per-sample operation. Behaves exactly like today's `sample_from_invariant(key, x0)` call inside `corrupt`.
- `MiniBatchOTCoupling`: `is_batch_level = True`. Training loops that vmap `corrupt` over the batch dimension must not use this coupling; instead, call `corrupt` once on the whole batch and let the coupling handle its own batch axis. Document this in the coupling's docstring and add an assertion helper `assert_vmappable(corruption_process)` the training loop can call.

### 3.3 RNG consumption identical for the Gaussian-source case

`StandardNormalSource.sample(key, data_spec)` is `jax.random.normal(key, shape=data_spec.shape)` — **exactly** the same call as today's `GaussianProcess.sample_from_invariant`. Same key, same shape, same output. No split, no fold_in, no intermediate wrappers that change the byte sequence.

Key splitting inside `InterpolantProcess.corrupt` only happens when `interpolant.needs_noise` is True. `LinearInterpolant` and `GeodesicInterpolant` both have `needs_noise = False`, so the non-split branch is taken for every existing call site. This is why the shim `GaussianProcess(schedule=S)` remains byte-exact despite the new `corrupt` containing a key-split branch: the branch is compile-time-determinate (the `needs_noise` field is static on the frozen dataclass) and JAX traces only the non-split arm.

### 3.4 Dataclass structure

- All protocols implemented as `@dataclasses.dataclass(frozen=True, kw_only=True)`.
- Fields that hold arrays (e.g., a learned schedule's parameters) follow the same PyTree registration convention the current schedules use. Fields that hold Python config are static under `jit`.
- `InterpolantProcess` has the same PyTree shape as `GaussianProcess` — an empty PyTree with three static fields — so `jax.jit(f, static_argnums=(...))` boundaries are unchanged.

### 3.5 Acceptance benchmark

Before milestone 1 merges, run a `jax.jit`-compiled training step with the shim `GaussianProcess` and the old `GaussianProcess` on the same inputs and RNG. Requirements:

1. Loss values bitwise identical across ≥3 schedule choices and ≥100 seeds.
2. Trace size (number of ops in the compiled HLO) differs by no more than a small constant from the old trace. Record the numbers in the PR description.
3. Wall-clock per step on a single A100 differs by <2%. Record the numbers in the PR description.

If any of these fail, the refactor is not done.

---

## 4. Byte-exact preservation strategy

### 4.1 What "preserved" means

For every existing `CorruptionProcess` instance, every method call with every input must produce bitwise-identical output before and after the refactor.

### 4.2 How to verify

Add `lib/corruption/interpolant_parity_test.py`:

```python
@pytest.mark.parametrize("schedule", [
    RFSchedule(), CosineSchedule(), InverseCosineSchedule(),
    LinearDiffusionSchedule(), GeometricSchedule(sigma_min=1e-3, sigma_max=10),
    ShiftedSchedule(original_schedule=CosineSchedule(),
                    target_resolution=64, base_resolution=32),
])
@pytest.mark.parametrize("seed", range(16))
@pytest.mark.parametrize("t", [0.01, 0.25, 0.5, 0.75, 0.99])
def test_gaussian_corrupt_bitwise_parity(schedule, seed, t):
  old = OldGaussianProcess(schedule=schedule)        # pinned copy
  new = GaussianProcess(schedule=schedule)           # new shim
  key = jax.random.PRNGKey(seed)
  x0  = jax.random.normal(jax.random.fold_in(key, 1), (8, 16))
  t_arr = jnp.full((8,), t)
  old_xt, old_ti = old.corrupt(key, x0, t_arr)
  new_xt, new_ti = new.corrupt(key, x0, t_arr)
  chex.assert_trees_all_equal(old_xt, new_xt)
  chex.assert_trees_all_equal(old_ti, new_ti)
```

Pin an "old" copy of `GaussianProcess` (e.g., `_gaussian_legacy.py`) so the test still exercises the real previous implementation after the refactor lands. Delete that file once milestone 1 is merged and the test is rewritten against git-archaeology fixtures.

Add equivalent tests for:
- `GaussianProcess.convert_predictions` across all 5 `source_type` values.
- `GaussianProcess.sample_from_invariant`.
- `RiemannianProcess.corrupt` + `convert_predictions` + `sample_from_invariant`.
- `DiffusionSampler.__call__` end-to-end (sampling golden test) on a trivial inference fn, checking the full trajectory is bitwise identical.

All existing tests in `lib/corruption/` and `lib/sampling/` must continue to pass without modification.

---

## 5. Milestones

### M1 — Protocols + Gaussian parity (target: 2-3 days)

**Scope.**
- Add `Coupling`, `Source`, `Interpolant`, `TargetAdapter` protocols to `lib/corruption/base.py`.
- Add `InterpolantProcess` concrete class to `lib/corruption/base.py`.
- Add `couplings.py` with `IndependentCoupling`, `StandardNormalSource`, `UniformManifoldSource`.
- Add `interpolants.py` with `LinearInterpolant` (wraps `GaussianSchedule`) and `GeodesicInterpolant` (wraps `RiemannianSchedule`).
- Add `targets.py` with `GaussianSourceTargets`, `RiemannianVelocityTargets`, `VelocityOnlyTargets`.
- Convert `GaussianProcess` and `RiemannianProcess` to shim constructors.

**Acceptance.**
- Parity test suite (§4.2) passes at 100%.
- Performance benchmark (§3.5) passes.
- All existing tests in `lib/corruption/`, `lib/sampling/`, `lib/guidance/`, `lib/loss/` pass with zero modifications.
- `GaussianProcess(schedule=...)` and `RiemannianProcess(manifold=..., schedule=...)` are the only new public API; everything else is internal.

### M2 — Discrete / simplicial decision

**Scope.**
- Document in `lib/corruption/base.py` whether `DiscreteProcess` and `SimplicialProcess` will adopt the (coupling, interpolant, targets) factoring.
- My recommendation: **leave them on their current abstraction**. Masking as a "coupling" is formally true but practically awkward, and there is no user-visible win. Add a docstring note rather than doing the work.

**Acceptance.**
- Decision recorded in `lib/corruption/base.py` module docstring and `LOCAL_PATCHES.md`.

### M3 — First new coupling: `DeterministicCoupling` (target: 1-2 days)

**Scope.**
- `DeterministicCoupling(map_fn: Callable[[DataTree], DataTree])` in `couplings.py`.
- Example instantiation in a test: blur-deblur flow matching on MNIST-scale data.
- Training loop runs to low loss with `VelocityStep` sampling.

**Acceptance.**
- A short integration test under `lib/corruption/` that trains a tiny MLP velocity predictor for ~200 steps and verifies loss decreases.
- Confirm no changes needed in `lib/sampling/` or `lib/guidance/` for the velocity-only path. Document any PiGDM / score-form friction encountered.

### M4 — Mini-batch OT coupling via `ott-jax` (target: 3-5 days, **required**)

**Scope.**

- Add `ott-jax` as a required dependency.
- `MiniBatchOTCoupling(source: Source)` in `couplings.py` — a thin wrapper that hands the batch to `ott.solvers.linear.sinkhorn.Sinkhorn` and samples matched `x_1` per row from the transport plan via `jax.random.categorical`. Sensible ott-jax defaults (`epsilon`, iteration count) are fine; expose only what we actually need to tune.
- `is_batch_level = True`. The only structural consequence is that training loops must not wrap `corrupt` in a per-sample `vmap`. Ship `assert_vmappable(process)` as a one-liner check and document the restriction in the coupling's docstring and `docs/corruption.md`.
- `jax.lax.stop_gradient` around the coupling output. Gradients don't flow through the OT plan. No knob for this in the first cut.
- Anything more (Hungarian, unbalanced OT, low-rank, custom cost functions, end-to-end differentiable plans) is out of scope. Users who need those can subclass `MiniBatchOTCoupling` or drop in a different ott-jax solver; the framework doesn't try to abstract over ott-jax's configuration surface.

**Acceptance.**

- Trains and samples inside a `jit`ed step function without per-step recompiles.
- Integration test: 2D `moons → circles` OT-flow-matching. Sampled distribution is closer (sliced-Wasserstein) to the target than a matched-compute `IndependentCoupling` baseline. Runnable script in `examples/`.
- Performance note in the PR description: step wall-clock with vs without OT, at our real training batch size.

### M5 — Full stochastic interpolant with `γ(t) z` augmentation (target: 1-2 weeks, **required**)

**Scope.**

- `StochasticInterpolant(alpha, beta, gamma)` in `interpolants.py`, implementing

  ```
  x_t = α(t) x_0 + β(t) x_1 + γ(t) z,     z ~ N(0, I) independent of (x_0, x_1)
  ```

  with `γ(0) = γ(1) = 0` enforced at construction time (either symbolically via known closed forms — `γ(t) = √(t(1-t))` is the canonical choice — or by explicit numerical check on a grid).
- `needs_noise = True`; key splitting in `InterpolantProcess.corrupt` routes the second key into `z`.
- `StochasticInterpolantTargets` adapter emits `{x0, x1, z, velocity, denoiser_z}` where `denoiser_z = z` so the model can be trained to predict `E[z | x_t]`. The score of ρ_t is recovered at inference via the SI identity

  ```
  ∇ log ρ_t(x) = - E[z | x_t = x] / γ(t)             for γ(t) > 0
  ```

  with the `γ(t) = 0` boundary handled by dropping the score term in the reverse SDE at the endpoints (the reverse SDE's diffusion coefficient `√((γ²)')` also vanishes there).
- Dual-head training: the `InferenceFn` protocol already supports predicting multiple keys in `TargetInfo`. The model gains a second output head for `denoiser_z`; the training loss is a weighted sum of `velocity` MSE and `denoiser_z` MSE. Add a `CombinedInterpolantLoss` in `lib/loss/` composed from the existing `compute_continuous_diffusion_loss` twice — no new loss primitive, just a named composition.
- `InterpolantSdeStep` in `lib/sampling/gaussian_step_sampler.py` — **the one sampling-layer change authorized in this program.** Implements the SI reverse SDE:

  ```
  dx = [b(t, x_t) - ½ ε(t)² s(t, x_t)] dt + ε(t) dW,
      ε(t) controlled by a new schedule protocol or a fixed function
  ```

  with `b = velocity prediction`, `s = - (denoiser_z / γ(t))`. The ODE limit (`ε(t) = 0`) reuses the existing `VelocityStep` — no code duplication.
- Provide a `StepKernel` implementation for `InterpolantSdeStep` so the composable-guidance machinery (SMC proposal ratios, PiGDM-style corrections on the score head) continues to apply. The kernel shape is two-endpoint in the mean — see §10.3.

**Acceptance.**

- Pure-unit tests of the interpolant math checked against Albergo–Boffe–Vanden-Eijnden 2023 (arXiv:2303.08797) equations 2.4, 2.13, 2.19: sampled `x_t` moments, derived velocity field, derived score field, reverse-time SDE drift and diffusion.
- Degenerate-case check: `StochasticInterpolant(α=1-t, β=t, γ=0)` is bitwise-equivalent to `LinearInterpolant(RFSchedule())` — same `x_t`, same velocity, same samples. This is the SI-subsumes-flow-matching consistency test. Required before M5 is declared done.
- Degenerate-case check: `StochasticInterpolant(α=1-t, β=0, γ=t)` with `x_1 ~ N(0, I)` via a trivial coupling reduces to the Gaussian diffusion `x_t = (1-t) x_0 + t ε` case and the emitted score matches today's `-ε/σ` identity.
- Integration test: train + sample a non-Gaussian → non-Gaussian 2D problem (two image-like toy distributions), comparing ODE sampling (`ε=0`) vs SDE sampling (`ε>0`) and verifying both recover the target distribution.
- `StepKernel`-based SMC proposal-ratio test: the `InterpolantSdeStep.kernel(...)` method returns a valid `GaussianStepKernel` (or sibling `InterpolantStepKernel` — see §10.3) whose `log_density_ratio` satisfies the same end-to-end smoke tests in `lib/guidance/guidance_literature_test.py`.
- Update `docs/composable_guidance.md`'s modality-compatibility matrix with a new row `Stochastic Interpolant (ODE / SDE)`.

**Open questions to resolve before implementation** — see §10.

---

## 6. File-by-file scope

### M1-M3 (factoring, no sampling / loss changes)

| File | Change |
|---|---|
| `lib/corruption/base.py` | Add 4 protocols + `InterpolantProcess` class (~150 lines added). |
| `lib/corruption/couplings.py` | New file (~150 lines): `IndependentCoupling`, `DeterministicCoupling`, `StandardNormalSource`, `UniformManifoldSource`, `DataloaderSource`. |
| `lib/corruption/interpolants.py` | New file (~200 lines): `LinearInterpolant`, `GeodesicInterpolant`. |
| `lib/corruption/targets.py` | New file (~200 lines). `GaussianSourceTargets`, `RiemannianVelocityTargets`, `VelocityOnlyTargets`. The `CONVERTERS` table from `gaussian.py` relocates here unchanged. |
| `lib/corruption/gaussian.py` | Collapses to ~40 lines: shim constructor `GaussianProcess` + re-exports. |
| `lib/corruption/riemannian.py` | Collapses to ~40 lines: shim constructor `RiemannianProcess` + re-exports. |
| `lib/corruption/schedules.py` | No change. |
| `lib/corruption/discrete.py`, `simplicial.py` | No change. See M2. |
| `lib/corruption/interpolant_parity_test.py` | New file (~200 lines) — the parity suite of §4.2. |
| `lib/sampling/` | **No change in M1-M3.** |
| `lib/guidance/` | **No change in M1-M3.** |
| `lib/loss/` | **No change in M1-M3.** |
| `lib/architecture/` | **No change.** |
| `LOCAL_PATCHES.md` | New entry 7 documenting the refactor. |
| `docs/corruption.md` | Updated to describe the factoring; old examples still work as-is. |

### M4 (mini-batch OT coupling)

| File | Change |
|---|---|
| `pyproject.toml` | Add `ott-jax` dependency, pin version. |
| `lib/corruption/couplings.py` | Add `MiniBatchOTCoupling` (~200 lines). |
| `lib/corruption/ot_solvers.py` | New file (~150 lines): `OTSolver` protocol, `SinkhornSolver`, `HungarianSolver`. |
| `lib/corruption/ot_coupling_test.py` | New file (~250 lines): parity, jit, pmap, gradient tests. |
| `examples/ot_flow_matching_2d.py` | New file (~120 lines): runnable OT-CFM example. |
| `docs/corruption.md` | New section on OT coupling patterns. |

### M5 (full stochastic interpolant)

| File | Change |
|---|---|
| `lib/corruption/interpolants.py` | Add `StochasticInterpolant` (~200 lines) + `γ(t) = 0` boundary check. |
| `lib/corruption/targets.py` | Add `StochasticInterpolantTargets` (~100 lines). |
| `lib/sampling/gaussian_step_sampler.py` | Add `InterpolantSdeStep` + its `StepKernel` (~150 lines). **The single authorized sampling-layer change in this program.** |
| `lib/loss/` | Add `CombinedInterpolantLoss` (~50 lines) — named composition over `compute_continuous_diffusion_loss`. |
| `lib/corruption/stochastic_interpolant_test.py` | New file (~300 lines): SI-math unit tests + degenerate-case equivalences. |
| `lib/sampling/interpolant_sde_step_test.py` | New file (~200 lines). |
| `examples/stochastic_interpolant_2d.py` | New file (~150 lines). |
| `docs/composable_guidance.md` | Update modality-compatibility matrix. |
| `docs/sampling.md` | Document `InterpolantSdeStep` and the `ε(t)` schedule protocol. |

---

## 7. Risks and mitigations

### 7.1 Hidden Gaussian-source assumptions in `lib/sampling/` or `lib/guidance/`

`SdeStep` reads `score`; `PiGDMCorrectionFn` reads `x0` and the Gaussian denoising identity. These are correctly Gaussian-source-only, and the refactor does not attempt to change that. What the refactor *does* is make the restriction explicit: a user constructing `InterpolantProcess(IndependentCoupling(DataloaderSource(...)), ..., VelocityOnlyTargets())` gets a process whose `convert_predictions` cannot emit `score`, and any downstream stepper asking for `score` fails at construction/trace time rather than silently producing garbage samples.

**Mitigation:** add a simple compatibility check at `DiffusionSampler` construction time: iterate the stepper's required target keys, check they are all emittable by the corruption process's target adapter, raise a clear `ValueError` listing the missing keys if not. ~20 lines, and it's purely additive.

### 7.2 PyTree registration drift under `jit`

The current `GaussianProcess` is a frozen dataclass holding a `schedule` frozen dataclass. `InterpolantProcess` holds three frozen dataclasses. If the JAX version treats these differently than the current single-field case, `jit` boundaries could change.

**Mitigation:** the performance benchmark in §3.5 catches this. If the benchmark passes, there is no regression.

### 7.3 The `sample_from_invariant` method is public API

Downstream code (including the multiscale repo's own sampling scripts) calls `corruption_process.sample_from_invariant(key, data_spec)` directly to build `initial_noise` for `DiffusionSampler`. The refactor must preserve this method on the `InterpolantProcess`.

**Mitigation:** `InterpolantProcess.sample_from_invariant` delegates to `self.coupling.sample_marginal(key, data_spec)`. For `IndependentCoupling(source)`, that's `source.sample(key, data_spec)`. Byte-identical for the standard-normal case. For couplings that don't have a well-defined marginal on `x_1` (e.g., `DeterministicCoupling(f)` — the marginal depends on `p_0`, which the coupling doesn't hold), raise at call time with a clear message. This is the correct behavior.

### 7.4 `LOCAL_PATCHES.md` divergence

This refactor is local. Upstream `hackable_diffusion` may change `GaussianProcess` in incompatible ways during the next rebase.

**Mitigation:** add a new entry to `LOCAL_PATCHES.md` documenting the refactor, with enough detail that a future rebase can either re-apply it on top of upstream changes or adopt a new upstream direction. Follow the format used by existing entries 1-6.

---

## 8. Out of scope

- Any change to how the training loop drives `corrupt` (e.g., changing the vmap boundary). Training code continues to call `corruption_process.corrupt(key, x0, t)` on a batch.
- Any change to the loss module. Losses read whatever keys are in `target_info` and don't care about the factoring.
- Any change to schedule classes or their `alpha` / `sigma` / `f` / `g` APIs.
- Any change to the `inference_fn` / `DenoiserFn` protocols in `lib/inference/` or `lib/guidance/`.
- Renaming `CorruptionProcess` → `Interpolant`. The file names stay. "Corruption" is a misnomer for the general case, but renaming would churn every downstream import for zero benefit.

---

## 9. Design questions to resolve before writing code

These are the load-bearing design decisions for M4 and M5. **Resolve them before starting M1, because several of them shape the M1 protocol signatures and it is much cheaper to fix the interfaces once than to migrate them later.** Each question should be answered in a short design note that becomes a follow-up section of this document or a dedicated ADR.

### 9.1 Key-splitting policy and byte-exactness

The shim `GaussianProcess` must produce bytewise-identical samples to today. `LinearInterpolant.needs_noise = False` causes `InterpolantProcess.corrupt` to take the non-split branch, so `StandardNormalSource` receives the original key unchanged. Verify at implementation time that:

- No intermediate `jax.random.fold_in` or `jax.random.split` sits between the caller's key and `jax.random.normal(key, shape=x0.shape)`.
- The same holds for `RiemannianProcess` under the shim.
- When `needs_noise = True`, the split order is fixed by the `InterpolantProcess.corrupt` implementation: `key_coupling, key_z = jax.random.split(key)` — coupling first, noise second. Lock this order in the spec; reversing it silently breaks reproducibility for any checkpoint trained on a given seed convention.

### 9.2 Coupling semantics: is the coupling a *distribution* or a *sampler*?

`Coupling.sample(key, x0)` is a sampler. For `IndependentCoupling`, the sampler is memoryless per-sample. For `DeterministicCoupling`, it's a pure function of `x_0`. For `MiniBatchOTCoupling`, it's a function of the *whole batch*.

The protocol needs to be honest about this. Two defensible designs:

**Option A (simpler, picks efficiency):** `Coupling.sample(key, x0_batch) -> x1_batch`. Every coupling accepts and returns a batch. `IndependentCoupling` trivially vmaps over the batch axis internally; `MiniBatchOTCoupling` doesn't. The caller does not vmap over `corrupt`; the training loop calls `corrupt` on the full batch at once.

**Option B (more flexible, picks ergonomics):** `Coupling.sample(key, x0) -> x1` is per-sample for per-sample couplings; batch-level couplings implement a separate `sample_batch` method and declare themselves via `is_batch_level`. Training loops dispatch.

**Recommendation:** Option A. One signature, one mental model, no dispatch. The `is_batch_level` class attribute is still declared (for `assert_vmappable` checks), but it's an advisory flag, not a protocol bifurcation. The "vmap over `corrupt`" idiom — if it exists anywhere in the codebase today — gets rewritten to "call `corrupt` on the batch directly." Verify this rewrite is localized; if it's scattered across many training loops, reconsider.

**Required**: before M1 starts, grep the codebase for calls to `corruption_process.corrupt` and `jax.vmap(... corrupt ...)` and document the current vmap boundary conventions. Attach the findings to this section.

### 9.3 `StepKernel` shape for two-endpoint interpolants

`GaussianStepKernel` (see `lib/sampling/gaussian_step_sampler.py:67-115`) parameterizes the reverse-time transition as

```
xt_next = coeff_x0 · xhat_0 + coeff_xt · xt + sigma_step · eps
```

For `StochasticInterpolant` with both `x_0` and `x_1` data-distributed, the natural reverse-SDE mean is two-endpoint:

```
xt_next = coeff_x0 · xhat_0 + coeff_x1 · xhat_1 + coeff_xt · xt + sigma_step · eps
```

Three ways to integrate this with the existing `StepKernel`:

**Option A**: Route everything through velocity. The stepper's `kernel(...)` method consumes `velocity` (which is `α̇ x_0 + β̇ x_1 + γ̇ z` internally) and collapses both endpoints into `coeff_velocity · vhat + coeff_xt · xt`. This matches the existing `VelocityStep.kernel` pattern and requires no new kernel class. The drawback: PiGDM-style corrections that want to project in `x_0`-space can't see `x_1`; they'd need a new adapter.

**Option B**: Add `InterpolantStepKernel` as a sibling of `GaussianStepKernel`, with both `coeff_x0` and `coeff_x1` fields plus both `x0_corrected` and `x1_corrected`. The `log_density_ratio` formula is the same quadratic form with a wider mean. `proposal_ratio.py`'s polymorphic dispatcher handles it automatically (no changes needed there; that's the whole point of the recent refactor).

**Option C**: Generalize `GaussianStepKernel` in place to a list of endpoints. This touches every existing stepper and breaks the "one formula, one primitive" property the last refactor established. Reject.

**Recommendation:** Option B. It respects the `StepKernel` primitive's one-liner contract, requires zero changes to `proposal_ratio.py`, and preserves all existing kernels unchanged. The implementation is a ~50-line new dataclass in `gaussian_step_sampler.py`.

Before M5 starts, confirm with the guidance subsystem that two-endpoint corrections (a Pi-GDM variant that projects onto a linear measurement of *either* endpoint) is a near-future need. If yes, Option B. If no, Option A is cheaper.

### 9.4 OT coupling: what we pass through from ott-jax

We wrap ott-jax; we don't reimplement it. `MiniBatchOTCoupling(source)` uses `ott.solvers.linear.sinkhorn.Sinkhorn` with library defaults, wraps the output in `jax.lax.stop_gradient`, and samples matched pairs via `jax.random.categorical`. The coupling doesn't try to abstract ott-jax's configuration surface (no custom solver protocol, no unbalanced/low-rank toggles surfaced, no pluggable cost function in the first cut). If we later need any of those knobs, add them one at a time, driven by an actual use case.

One implementation-time check: verify `jax.lax.stop_gradient` around the solver output plays nicely with `jax.checkpoint` / `remat` (no re-solving on the backward pass). If there's friction, pin the pattern that avoids it.

### 9.5 `γ(t)` admissibility check for `StochasticInterpolant`

The SI boundary conditions `γ(0) = γ(1) = 0` are non-negotiable — violating them corrupts the endpoints and the trained model produces biased samples. Two enforcement levels:

- **Numerical check at construction**: evaluate `γ` on a small grid near 0 and 1, assert values are below a tolerance. Cheap, catches typos.
- **Symbolic check for a small library of known-good γ choices**: `γ(t) = √(t(1-t))`, `γ(t) = sin(π t)`, `γ(t) = t(1-t)`. Ship these as named `GammaSchedule` subclasses with `γ(0) = γ(1) = 0` guaranteed by construction. User-supplied callables get the numerical check.

**Decision: both.** Named schedules skip the numerical check. Callables get the numerical check. Document that the tolerance is strict and that users with unusual schedules should supply a `tolerance=` override.

### 9.6 Dual-head training loss weighting

The M5 training loss is `w_v · L(velocity) + w_z · L(denoiser_z)`. The weights matter — setting them wrong makes one head dominate and the other underfit. The Albergo–Boffe–Vanden-Eijnden paper gives specific normalizations.

**Decision: ship three loss-weighting schemes explicitly**: equal weights, SI-paper normalizations, and a user-supplied pair `(w_v(t), w_z(t))`. Document the equal-weights choice as "will work on toy problems but probably wrong at scale"; document the SI-paper choice as the default for serious runs.

### 9.7 The `score_of_rho` identity at γ(t) = 0

At `t ∈ {0, 1}` with the canonical `γ(t) = √(t(1-t))`, the score identity `s = -E[z|x_t] / γ(t)` has a 0/0 form. The SDE's diffusion coefficient `√((γ²)')` also vanishes (the same rate, as can be checked), so the full score-multiplied drift term has a finite limit. Implementation must handle this cleanly.

**Decision: clamp `γ(t)` to `max(γ, ε_floor)` inside the score identity and multiply the reverse-SDE drift by `(γ²)'` directly rather than splitting into `s · √((γ²)')`.** This keeps the combined `-½ (γ²)' s = -½ ((γ²)' / γ) E[z|x_t]` term finite as long as the ratio is bounded (it is, for the canonical `γ`). Add a pure-unit test that evaluates the full drift at `t ∈ {0, ε, 1-ε, 1}` for `ε ∈ {1e-3, 1e-5, 1e-7}` and verifies no NaNs.

### 9.8 What stays in `target_info`, what moves out

The current `target_info` dict is five keys for Gaussian, three for Riemannian. M5 adds `z` and `denoiser_z`. The dict is growing.

**Decision: keep the flat-dict contract.** Downstream losses already dispatch on key presence; they can handle new keys. Do *not* introduce a typed schema or a wrapper dataclass for `TargetInfo` — that churn doesn't earn its complexity at this scale. The one discipline: each `TargetAdapter` subclass must declare its emitted keys as a `ClassVar[frozenset[str]]`, and `DiffusionSampler` compatibility checks (see §7.1) consult this.

---

## 10. Checklist for the executing colleague

### Before starting M1

- [ ] Resolve §9 design questions. Write a short design note for each; paste into this document as §9.x resolutions or link to an ADR.
- [ ] Read `lib/corruption/gaussian.py` end-to-end, then `lib/corruption/riemannian.py`, then `lib/corruption/base.py`. These are the three files whose behavior you must preserve exactly.
- [ ] Read `lib/sampling/gaussian_step_sampler.py:180-200` and `:430-460` to confirm which methods on `corruption_process` the steppers call. There should be exactly three: `convert_predictions`, `schedule.<attr>`, and `sample_from_invariant`. Confirm this before starting.
- [ ] Grep for `jax.vmap` and `corruption_process.corrupt` in the multiscale and mdt consumer codebases. Document the current vmap-boundary convention as part of resolving §9.2.

### During M1

- [ ] Pin a copy of `GaussianProcess` as `_gaussian_legacy.py` before touching anything.
- [ ] M1 implementation in a single PR. No stacked PRs; the parity test suite needs to exercise the full refactor at once.
- [ ] Record the §3.5 benchmark numbers in the PR description.
- [ ] Add the `LOCAL_PATCHES.md` entry in the same PR.

### Before starting M4

- [ ] Add ott-jax to `pyproject.toml` in a separate small PR; verify CI picks it up.
- [ ] Verify no training loop in mdt/multiscale vmaps over `corrupt` (per the §9.2 decision). If any does, plan the rewrite as part of M4 and record it here.

### Before starting M5

- [ ] Re-read §9.3, §9.5, §9.7 before writing any code. The boundary-condition handling and kernel shape are the two places where small mistakes produce wrong samples that look plausible.
- [ ] Write the degenerate-case tests (`γ=0` reduces to flow matching; `β=0, γ=t` reduces to diffusion) *before* writing the general `StochasticInterpolant` implementation. These tests pin down the intended behavior.

### After M1

- [ ] Delete `_gaussian_legacy.py` in a follow-up PR once M1 is merged and you've verified downstream (mdt, multiscale) training is unaffected for at least one real training run.
