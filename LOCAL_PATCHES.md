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
