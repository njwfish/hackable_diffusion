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

"""Distributional inference fn x guidance composability tests.

A distributional inference fn (De Bortoli et al. 2025) is **stochastic**:
the same ``(xt, time)`` produces different ``x_0`` samples under different
``rng`` keys.  This breaks anything in the guidance stack that assumes the
denoiser is deterministic, and the existing literature tests
(``guidance_literature_test.py``) only exercise deterministic inference
fns.  The fixture below fills that gap.

:class:`StochasticInferenceFnRngTest` drives a minimal "rng required"
inference fn through :class:`ConditionalDiffusionSampler` with a twist
and a resampler.  The fixture's ``__call__`` raises if it is invoked
without ``rng`` -- so the test fails loudly if any code path in the
sampler builds a denoiser closure with ``rng=None``.  The bug we
specifically lock in: the initial twist evaluation that happens *before*
the scan loop used to pass ``rng=None`` and was invisible to every
deterministic-inference-fn test.
"""

from __future__ import annotations

import dataclasses
import unittest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.corruption.gaussian import GaussianProcess
from hackable_diffusion.lib.inference.base import InferenceFn
from hackable_diffusion.lib.sampling.gaussian_step_sampler import DDIMStep
from hackable_diffusion.lib.sampling.sampling import DiffusionSampler
from hackable_diffusion.lib.sampling.time_scheduling import UniformTimeSchedule

from hackable_diffusion.lib.guidance.forward_ops import SubsampleForwardFn
from hackable_diffusion.lib.guidance.resamplers import (
    ESSThresholdedResamplerFn,
    SystematicResamplerFn,
)
from hackable_diffusion.lib.guidance.sampler import ConditionalDiffusionSampler
from hackable_diffusion.lib.guidance.twists import GaussianLikelihoodTwistFn


################################################################################
# MARK: Fixture -- "rng required" inference fn
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class _RequireRngInferenceFn(InferenceFn):
  """Identity inference fn that raises if ``rng`` is not provided.

  Drops every internal stochastic op behind a ``rng is None`` guard so
  that any code path in the guidance stack that builds a denoiser
  closure with ``rng=None`` blows up loudly.  Pair this with a sampler
  composition that touches every closure-building site (initial twist
  eval, per-step denoiser, final twist eval) to lock in correctness.
  """

  def __call__(
      self,
      time: jax.Array,
      xt: jax.Array,
      conditioning=None,
      rng: jax.Array | None = None,
  ) -> dict[str, jax.Array]:
    if rng is None:
      raise ValueError(
          "_RequireRngInferenceFn: rng is None -- the sampler is calling "
          "us without a key, which would break a real distributional "
          "inference fn."
      )
    del time, conditioning
    return {"x0": xt}


################################################################################
# MARK: Sampler must thread rng through every denoiser-closure site
################################################################################


class StochasticInferenceFnRngTest(unittest.TestCase):
  """``ConditionalDiffusionSampler`` with a twist + resampler must pass an
  rng to every denoiser-closure-building site, including the initial
  twist evaluation that runs before the scan loop.
  """

  n = 4
  m = 2
  batch = 8
  num_steps = 6
  rng_seed = 0

  def test_full_smc_loop_with_rng_required_inference_fn(self):
    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    base_sampler = DiffusionSampler(
        time_schedule=UniformTimeSchedule(),
        stepper=DDIMStep(corruption_process=corruption, stoch_coeff=1.0),
        num_steps=self.num_steps,
        store_trajectory=False,
    )

    rng = np.random.default_rng(self.rng_seed)
    indices = np.sort(rng.choice(self.n, size=self.m, replace=False))
    y = rng.standard_normal(self.m).astype(np.float64)
    forward_fn = SubsampleForwardFn(indices=jnp.asarray(indices))
    observation = jnp.broadcast_to(
        jnp.asarray(y, dtype=jnp.float64)[None], (self.batch, self.m),
    )

    twist = GaussianLikelihoodTwistFn(
        observation=observation,
        forward_fn=forward_fn,
        observation_noise=0.5,
    )
    resampler = ESSThresholdedResamplerFn(
        base=SystematicResamplerFn(), threshold=0.5,
    )

    sampler = ConditionalDiffusionSampler(
        base_sampler=base_sampler,
        corruption_process=corruption,
        correction_fn=None,
        twist_fn=twist,
        resampler_fn=resampler,
    )

    rng_key = jax.random.PRNGKey(self.rng_seed)
    init = jax.random.normal(rng_key, (self.batch, self.n), dtype=jnp.float64)
    inference_fn = _RequireRngInferenceFn()

    final_step, _, log_w_final = sampler(
        inference_fn=inference_fn,
        rng=rng_key,
        initial_noise=init,
    )

    self.assertTrue(bool(jnp.all(jnp.isfinite(final_step.xt))))
    self.assertTrue(bool(jnp.all(jnp.isfinite(log_w_final))))
    self.assertEqual(final_step.xt.shape, (self.batch, self.n))


if __name__ == "__main__":
  unittest.main()
