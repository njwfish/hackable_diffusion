# Hackable diffusion

Hackable Diffusion is a modular toolbox written in Jax to experiment and educate
around Diffusion modeling.

## Philosophy

The core philosophy of this library is **hackability**. It is designed from the
ground up to be modular, composable, and easy to modify, enabling rapid
experimentation with new research ideas. Key principles include:

*   **Composition over Configuration**: Build models and training loops by composing small, well-defined Python objects.
*   **Clear Separation of Concerns**: The codebase is organized into logical sub-libraries for architecture, corruption, inference, loss, and sampling.
*   **Native Multimodality**: The library has first-class support for handling multimodal data (e.g., images and text) through a consistent "Nested" component pattern that applies different diffusion parameters to different parts of the data.

## Tutorials

The `notebooks/` directory contains several tutorials to get you started:

*   **`2d_training.ipynb`**: A minimal example on a 2D toy dataset.
*   **`mnist.ipynb`**: Standard image diffusion on MNIST.
*   **`mnist_discrete.ipynb`**: An example of discrete diffusion.
*   **`mnist_multimodal.ipynb`**: A showcase of the multimodal capabilities, generating images and labels jointly.

## Sampling with distributional (posterior) models

A standard diffusion sampler turns a *point* denoiser estimate into a
transition via a score/velocity SDE or ODE update (`DDIMStep`, `SdeStep`,
`VelocityStep`, ...).  A **distributional** model is different: it reports
the clean-endpoint posterior `p(x_0 | x_t)`, and the finite step is then
just two operations, *the same for every modality*:

1.  `x_0 = sample_endpoint(prediction)` -- draw one clean endpoint from the
    model's report;
2.  `x_s = bridge_step(x_0, x_t, t -> s)` -- apply the corruption's own
    two-endpoint bridge `K(x_s | x_0, x_t)`.

Both are the **reverse face of the same `CorruptionProcess`** that defined
the corruption at training (`corrupt`).  Because it is the same object
(with the same noise), the sampling step is consistent with training *by
construction* -- there is no separate stepper math to keep in sync, and no
score or velocity in between.

A single stepper, `PosteriorBridgeStep` (in `lib.sampling`), runs this for
every modality by delegating both steps to the process:

```python
from hackable_diffusion.lib.corruption.gaussian import GaussianProcess
from hackable_diffusion.lib.corruption.schedules import RFSchedule
from hackable_diffusion.lib.inference.posterior_sampler import (
    PosteriorSamplerInferenceFn,
)
from hackable_diffusion.lib.sampling.bridge_step_sampler import PosteriorBridgeStep
from hackable_diffusion.lib.sampling.sampling import DiffusionSampler
from hackable_diffusion.lib.sampling.time_scheduling import UniformTimeSchedule

process = GaussianProcess(schedule=RFSchedule())
inference_fn = PosteriorSamplerInferenceFn(               # reports the posterior
    network=net, params=params,
)

sampler = DiffusionSampler(
    time_schedule=UniformTimeSchedule(),
    stepper=PosteriorBridgeStep(corruption_process=process),
    num_steps=num_steps,
)
last_step, trajectory = sampler(
    inference_fn=inference_fn,
    rng=key,
    initial_noise=process.sample_from_invariant(key, data_spec),
    conditioning=conditioning,
)
```

The same `PosteriorBridgeStep` works across every modality, because the
modality lives entirely in the process's `sample_endpoint` / `bridge_step`:

| Modality | `sample_endpoint` | `bridge_step` |
| --- | --- | --- |
| Gaussian / flow (`LinearInterpolant`) | read the `x0` sample | deterministic affine |
| Stochastic interpolant | read the `x0` sample | Brownian bridge (its own `gamma`) |
| Riemannian (`GeodesicInterpolant`) | read the `x0` sample | geodesic |
| Categorical | categorical draw from logits | coordinate-switch reveal |
| Dirichlet (`SimplicialProcess`) | `softmax(logits)` | Beta/Dirichlet mix |
| Multimodal (`NestedProcess`) | per-modality | per-modality (mapped) |

For continuous models the endpoint draw happens inside the inference fn
(it reports an `x0` sample), so `sample_endpoint` is the identity; for
categorical/Dirichlet the inference fn reports logits and `sample_endpoint`
materializes the draw.  Multimodal falls out for free: `NestedProcess`
maps both methods over its sub-processes, so one `PosteriorBridgeStep`
drives mixed image+label data.

**Relationship to the existing steppers (so there's no ambiguity).** For a
plain Gaussian `LinearInterpolant`, the bridge step is *identical* to
`DDIMStep(stoch_coeff=0)` -- both compute
`x_s = (sigma_s/sigma_t) x_t + (alpha_s - alpha_t sigma_s/sigma_t) x_0` --
so `DDIMStep(stoch_coeff=0)` remains a fine conventional choice for that
case.  The feature-rich discrete / simplicial steppers (`UnMaskingStep`,
`DiscreteDDIMStep`, `DiscreteFlowMatchingStep`, `SimplicialDDIMStep`) are
enhancements of the same bridge with extra knobs (remasking, planning,
churn); `PosteriorBridgeStep` is the bare, uniform path. In particular
`SimplicialProcess.bridge_step` equals `SimplicialDDIMStep` at `churn=1`.

For the **categorical** and **Dirichlet (simplicial)** models the posterior
view is already native: the network predicts a *distribution* over the clean
endpoint (logits / Dirichlet parameters), and the dedicated steppers
(`UnMaskingStep`, `DiscreteDDIMStep`, `DiscreteFlowMatchingStep`,
`SimplicialDDIMStep`) sample the endpoint and apply the modality's
two-endpoint bridge in a single `update`.  Use those steppers directly --
no separate posterior-sampler inference fn is required.

## Installation

To install the necessary dependencies, you can use pip with the provided
`pyproject.toml` file:

```bash
pip install -e .
```

To install development dependencies (for running tests), use:

```bash
pip install -e .[dev]
```

This will install libraries such as JAX, Flax, and other utilities required to
run the code.

## Projects

For experimental projects, please refer to `third_party/py/hd_projects`.

## Disclaimer

Copyright 2025 Google LLC \
All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you
may not use this file except in compliance with the Apache 2.0 license. You may
obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0 All other materials are licensed
under the Creative Commons Attribution 4.0 International License (CC-BY). You
may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode Unless required by
applicable law or agreed to in writing, all software and materials distributed
here under the Apache 2.0 or CC-BY licenses are distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
licenses for the specific language governing permissions and limitations under
those licenses.
This is not an official Google product.
