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

"""Corrections built from other corrections / twists.

- :class:`IteratedCorrectionFn` wraps any ``CorrectionFn`` to run K inner
  Kalman sweeps with denoiser re-evaluation (closes the intermediate-H
  non-Gaussian bump on mixture priors).
- :class:`GradientCorrectionFn` bridges ``TwistFn`` -> ``CorrectionFn``
  by backpropagating the twist gradient through the denoiser
  (generalises DPS / Pi-GDM-via-Miyasawa).

The step-size prefactor is an injected callable (``prefactor_fn``) so
new schedules can be plugged in without subclassing; two built-ins are
exported: :func:`miyasawa_prefactor` (Tweedie-identity scaling, exact on
Gaussian priors) and :func:`dps_prefactor` (canonical DPS
``1 / ||residual||`` step).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import jax
import jax.numpy as jnp

from hackable_diffusion.lib.guidance.protocols import CorrectionFn, TwistFn
from hackable_diffusion.lib.guidance.utils import (
    call_inference_fn,
    scalar_alpha_sigma,
)


def miyasawa_prefactor(
    *,
    alpha: jax.Array,
    sigma: jax.Array,
    xt: jax.Array,
    x0: jax.Array,
) -> jax.Array:
  """Tweedie-identity prefactor ``sigma^2 / alpha``.

  Exact in the single-Gaussian-prior limit via the Miyasawa identity
  ``Cov(x0 | xt) = (sigma^2 / alpha) ∂xhat_0/∂xt``.
  """
  del xt, x0
  return (sigma ** 2) / jnp.maximum(alpha, 1e-8)


def dps_prefactor(
    *,
    alpha: jax.Array,
    sigma: jax.Array,
    xt: jax.Array,
    x0: jax.Array,
) -> jax.Array:
  """Canonical DPS step ``1 / ||residual||``."""
  del sigma
  flat = (x0 - xt / jnp.maximum(alpha, 1e-8))
  norm = jnp.linalg.norm(
      flat.reshape(flat.shape[0], -1), axis=-1, keepdims=True,
  ) + 1e-8
  return 1.0 / jnp.maximum(norm, 1e-8)


# Signature of a step-size prefactor callable used by GradientCorrectionFn.
# Kept as a typing alias rather than a Protocol because it's a free
# function, not an object with methods.
PrefactorFn = Callable[..., jax.Array]


@dataclasses.dataclass(kw_only=True, frozen=True)
class IteratedCorrectionFn(CorrectionFn):
  """Apply a base correction ``num_iters`` times with denoiser re-evaluation.

  At each inner iteration:

    1. Apply ``base`` correction to the current outputs → updated ``xhat_0``.
    2. Shift ``xt`` by ``alpha_t (xhat_0_new - xhat_0_old)`` so the noise
       realisation is preserved.
    3. Re-evaluate the denoiser at the shifted ``xt``.
    4. Repeat for ``num_iters`` total iterations; the final correction
       applies without a further xt shift (the sampler advances from the
       actual xt).

  For a Pi-GDM-style base correction on a mixture prior, each inner
  iteration lets the denoiser's implicit mixture weights respond to the
  shifted ``xt``, reducing the linearisation error that gives the
  intermediate-H bump on non-Gaussian posteriors.  K=3 is the empirical
  sweet spot; K >= 8 destabilises.
  """

  base: CorrectionFn
  num_iters: int = 3

  def __call__(
      self,
      outputs: dict[str, jax.Array],
      xt: jax.Array,
      time: jax.Array,
      *,
      schedule: Any,
      corruption_process: Any,
      conditioning: Any = None,
      rng: jax.Array | None = None,
      inference_fn: Callable | None = None,
  ) -> dict[str, jax.Array]:
    if inference_fn is None:
      raise ValueError(
          "IteratedCorrectionFn needs ``inference_fn`` threaded through so "
          "it can re-evaluate the denoiser at the shifted xt on each inner "
          "iteration.  ConditionalDiffusionSampler wires this in "
          "automatically; if invoking directly, pass inference_fn=... ."
      )
    alpha, _ = scalar_alpha_sigma(schedule, time)

    xt_curr = xt
    outputs_curr = outputs

    for _ in range(int(self.num_iters) - 1):
      x0_before = corruption_process.convert_predictions(
          outputs_curr, xt_curr, time,
      )["x0"]
      corrected = self.base(
          outputs_curr, xt_curr, time,
          schedule=schedule, corruption_process=corruption_process,
          conditioning=conditioning, rng=rng,
      )
      x0_after = corruption_process.convert_predictions(
          corrected, xt_curr, time,
      )["x0"]
      xt_curr = xt_curr + alpha * (x0_after - x0_before)
      outputs_curr = call_inference_fn(
          inference_fn, xt=xt_curr, time=time,
          conditioning=conditioning, rng=rng,
      )

    return self.base(
        outputs_curr, xt_curr, time,
        schedule=schedule, corruption_process=corruption_process,
        conditioning=conditioning, rng=rng,
    )


@dataclasses.dataclass(kw_only=True, frozen=True)
class GradientCorrectionFn(CorrectionFn):
  """Correction formed by backpropagating a twist gradient through the denoiser.

  Implements DPS-style guidance generically:

      ``xhat_0_new = xhat_0 + strength * prefactor * ∇_{xt} log psi(y | xt)``

  and hence a modified score ``s_guided = (alpha_t xhat_0_new - xt) / sigma_t^2``.

  The prefactor is an injected callable, keyword-only:

      ``prefactor_fn(alpha=, sigma=, xt=, x0=) -> scalar or (B, 1, ...)``

  Built-ins in this module: :func:`miyasawa_prefactor` (default; exact
  Bayes-linear scaling on Gaussian priors via the Miyasawa identity) and
  :func:`dps_prefactor` (canonical DPS ``1 / ||residual||``).  Plug in
  any callable matching the signature for custom schedules.
  """

  twist: TwistFn
  strength: float = 1.0
  prefactor_fn: PrefactorFn = miyasawa_prefactor

  def __call__(
      self,
      outputs: dict[str, jax.Array],
      xt: jax.Array,
      time: jax.Array,
      *,
      schedule: Any,
      corruption_process: Any,
      conditioning: Any = None,
      rng: jax.Array | None = None,
      inference_fn: Callable | None = None,
  ) -> dict[str, jax.Array]:
    if inference_fn is None:
      raise ValueError(
          "GradientCorrectionFn needs ``inference_fn`` threaded through the "
          "call.  Wire it via ConditionalDiffusionSampler or pass it "
          "explicitly when invoking the correction."
      )

    alpha, sigma = scalar_alpha_sigma(schedule, time)

    def scalar_twist(xt_inner: jax.Array) -> jax.Array:
      return jnp.sum(self.twist(
          xt=xt_inner, time=time,
          inference_fn=inference_fn, schedule=schedule,
          corruption_process=corruption_process,
          conditioning=conditioning, rng=rng,
      ))

    grad_xt = jax.grad(scalar_twist)(xt)

    x0 = corruption_process.convert_predictions(outputs, xt, time)["x0"]
    prefactor = self.prefactor_fn(
        alpha=alpha, sigma=sigma, xt=xt, x0=x0,
    )

    delta_x0 = float(self.strength) * prefactor * grad_xt
    x0_guided = x0 + delta_x0
    guided_outputs = corruption_process.convert_predictions(
        {"x0": x0_guided}, xt, time,
    )
    pred_type = next(iter(outputs.keys()))
    return {pred_type: guided_outputs[pred_type]}
