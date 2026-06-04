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
`VelocityStep`, ...).  A **distributional** model is different: at a state
`x_t` it produces a *sample* from the clean-endpoint posterior
`p(x_0 | x_t)`, and the finite step is then just the fixed two-endpoint
bridge of the corruption geometry:

1.  draw `x_0 ~ p(x_0 | x_t)` from the model;
2.  apply `x_s ~ K(x_s | x_0, x_t)` -- the interpolant's bridge step.

There is no score or velocity in between.  This is implemented by
`BridgeStep` (in `lib.sampling`), which reads the endpoint sample from the
model's `x0` output and delegates the move to `Interpolant.bridge_step`.
It is a drop-in `SamplerStep`, so running a distributional model is a
one-line stepper swap:

```python
from hackable_diffusion.lib.corruption.gaussian import GaussianProcess
from hackable_diffusion.lib.corruption.schedules import RFSchedule
from hackable_diffusion.lib.inference.posterior_sampler import (
    PosteriorSamplerInferenceFn,
)
from hackable_diffusion.lib.sampling.bridge_step_sampler import BridgeStep
from hackable_diffusion.lib.sampling.sampling import DiffusionSampler
from hackable_diffusion.lib.sampling.time_scheduling import UniformTimeSchedule

process = GaussianProcess(schedule=RFSchedule())          # exposes `.interpolant`
inference_fn = PosteriorSamplerInferenceFn(               # emits {"x0": sample}
    network=net, params=params,
)

sampler = DiffusionSampler(
    time_schedule=UniformTimeSchedule(),
    stepper=BridgeStep(corruption_process=process),       # <- the only change vs DDIM/SDE
    num_steps=num_steps,
)
last_step, trajectory = sampler(
    inference_fn=inference_fn,
    rng=key,
    initial_noise=process.sample_from_invariant(key, data_spec),
    conditioning=conditioning,
)
```

`BridgeStep` works across the continuous geometries through one interface:

*   **Gaussian / flow** (`LinearInterpolant`): the deterministic affine bridge
    -- equivalently the ODE-limit DDIM update.
*   **Stochastic interpolant** (`StochasticInterpolant`): the Brownian-bridge
    step, with bridge noise drawn from the per-step rng.
*   **Riemannian** (`GeodesicInterpolant`): the geodesic bridge.

Guidance composes for free: a `CorrectionFn` shifts `x_0`, then the
*unchanged* bridge step applies, and `BridgeStep.kernel(...)` supplies the
SMC proposal ratio -- so it slots into `ConditionalDiffusionSampler` like
any other stepper.

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
