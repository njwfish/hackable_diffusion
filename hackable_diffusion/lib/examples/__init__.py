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

"""Tractable example targets for testing the posterior-bridges framework.

Each module here provides:

  - An analytic target distribution ``p_0``,
  - The analytic VP-Gaussian posterior ``p_{0|t}(. | x_t)`` (or its
    finite-state analogue),
  - A "ground-truth" posterior sampler that the framework can use as
    an ``inference_fn``,
  - Analytic ``H_t(x_t)`` for canonical tilts,

so that integration tests against the sampler can verify framework
output against closed-form references.

These are not training utilities -- they are *reference implementations*
of the manuscript's analytic test cases (single-Gaussian, two-component
mixture, etc.) that the integration tests in this directory use to
certify the cloud-aware twists / corrections produce the right
distributions.
"""

from hackable_diffusion.lib.examples.discrete_toy import (
    AntiCorrelatedPair,
    DiscreteJointPrior,
    ParityConstraint,
)
from hackable_diffusion.lib.examples.gaussian_mixture import (
    alpha_from_schedule,
    GaussianMixture,
    GaussianMixtureBridge,
    posterior_sampler_inference_fn,
)
