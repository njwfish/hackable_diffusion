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

"""The posterior-bridge sampler step (Posterior Bridges, Algorithm 2).

For a distributional / posterior model the finite-step update is *not* a
score / velocity reverse-time integrator.  It is two operations:

  1. draw a clean-endpoint sample ``x_0 ~ p_{0|t}(. | x_t)`` -- supplied
     by the inference fn as the ``x0`` entry of its prediction (e.g.
     :class:`hackable_diffusion.lib.inference.PosteriorSamplerInferenceFn`);
  2. apply the fixed two-endpoint bridge ``x_s ~ K_{s|0,t}(. | x_0, x_t)``
     -- supplied by the interpolant's ``bridge_step``.

That is the whole step.  There is no schedule-derived drift, no score
conversion, no churn knob: the geometry lives entirely in the
interpolant's ``bridge_step`` and the learned object lives entirely in
``x_0``.  This is the literal realisation of the manuscript's modularity
claim -- ``(geometry = interpolant) x (posterior = inference_fn)`` -- and
is strictly simpler than the Gaussian SDE/ODE steppers, which exist to
turn a *point* denoiser estimate into a transition.

Compared with :class:`hackable_diffusion.lib.sampling.gaussian_step_sampler.DDIMStep`
et al., ``BridgeStep``:

  * reads only ``x_0`` and ``x_t`` -- never a score or velocity;
  * works across every interpolant geometry (Euclidean linear/Gaussian,
    Brownian-bridge stochastic interpolant, Riemannian geodesic) through
    the single ``Interpolant.bridge_step`` interface;
  * is marginally exact whenever the supplied posterior is exact
    (Theorem: finite-step posterior-bridge identity), with no
    discretisation of a continuous-time SDE.

The discrete / finite-state analogue of this step is the coordinate-switch
reveal already implemented by the routing samplers in
:mod:`hackable_diffusion.lib.sampling.discrete_step_sampler`; this module is
the continuous-state member of the same family.
"""

import dataclasses
from typing import Protocol

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.corruption import base as corruption_base
from hackable_diffusion.lib.sampling import base
from hackable_diffusion.lib.sampling import gaussian_step_sampler
import jax
import jax.numpy as jnp
import kauldron.ktyping as kt

################################################################################
# MARK: Type Aliases
################################################################################

DataArray = hd_typing.DataArray
TargetInfo = hd_typing.TargetInfo
TimeTree = hd_typing.TimeTree

DiffusionStep = base.DiffusionStep
StepInfo = base.StepInfo
SamplerStep = base.SamplerStep

Interpolant = corruption_base.Interpolant
InterpolantProcess = corruption_base.InterpolantProcess
GaussianStepKernel = gaussian_step_sampler.GaussianStepKernel


class BridgeProcess(Protocol):
  """A corruption process that exposes a bridge geometry for :class:`BridgeStep`.

  The minimal contract the posterior-bridge sampler reads: an
  ``interpolant`` carrying the two-endpoint :meth:`Interpolant.bridge_step`
  (and, for SMC, the optional ``bridge_coefficients`` hook), plus
  ``convert_predictions`` as the fallback that maps a non-``x0``
  parameterisation to a clean-endpoint sample.  :class:`InterpolantProcess`
  and the ``GaussianProcess`` / ``RiemannianProcess`` shims all satisfy it
  structurally -- no registration needed.
  """

  interpolant: Interpolant

  def convert_predictions(
      self,
      prediction: TargetInfo,
      xt: DataArray,
      time: TimeTree,
  ) -> TargetInfo: ...


def _time_1d(time: jax.Array) -> jax.Array:
  """First entry of a (possibly batched) time array, kept shape ``(1,)``.

  Schedules are ``@kt.typechecked`` to reject a bare scalar ``()``; the
  Gaussian steppers feed shape ``(1,)`` and reshape the *result*.  We
  mirror that: the bridge coefficients are scalar across particles (all
  share the step's time), so we evaluate at one time and broadcast.
  """
  return jnp.atleast_1d(time).reshape(-1)[0:1]


################################################################################
# MARK: Bridge Step
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class BridgeStep(SamplerStep):
  """Posterior-bridge step: draw ``x_0`` from the posterior, apply ``K_{s|0,t}``.

  Pair this with a distributional inference fn whose prediction carries a
  clean-endpoint *sample* under the ``x0`` key (e.g.
  :class:`hackable_diffusion.lib.inference.PosteriorSamplerInferenceFn`).
  The step itself is geometry-agnostic: it delegates the move from ``x_t``
  to ``x_s`` to ``corruption_process.interpolant.bridge_step``.

  Attributes:
    corruption_process: A process exposing an ``interpolant`` with a
      ``bridge_step`` (any :class:`InterpolantProcess`, or the
      ``GaussianProcess`` / ``RiemannianProcess`` shims) -- see
      :class:`BridgeProcess`.
  """

  corruption_process: BridgeProcess

  def _x0(
      self, prediction: TargetInfo, xt: DataArray, time,
  ) -> DataArray:
    """Extract the clean-endpoint sample ``x_0`` from the prediction.

    A posterior sampler emits ``x_0`` directly; we use it verbatim and
    only fall back to the parameterisation-conversion table for
    predictions reported in another coordinate.
    """
    if "x0" in prediction:
      return prediction["x0"]
    return self.corruption_process.convert_predictions(prediction, xt, time)[
        "x0"
    ]

  @kt.typechecked
  def initialize(
      self,
      initial_noise: DataArray,
      initial_step_info: StepInfo,
  ) -> DiffusionStep:
    return DiffusionStep(
        xt=initial_noise,
        step_info=initial_step_info,
        aux=dict(),
    )

  @kt.typechecked
  def update(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
  ) -> DiffusionStep:
    xt = current_step.xt
    time = current_step.step_info.time
    next_time = next_step_info.time

    x0 = self._x0(prediction, xt, time)
    x_s = self.corruption_process.interpolant.bridge_step(
        x0=x0,
        xt=xt,
        t=time,
        s=next_time,
        key=next_step_info.rng,
    )
    return DiffusionStep(
        xt=x_s,
        step_info=next_step_info,
        aux=dict(),
    )

  @kt.typechecked
  def finalize(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      last_step_info: StepInfo,
  ) -> DiffusionStep:
    # At s = 0 the bridge collapses to the clean endpoint
    # (K_{0|0,t} = delta_{x_0}); update handles that limit directly.
    return self.update(
        prediction,
        current_step,
        last_step_info,
    )

  def kernel(
      self,
      *,
      prediction_uncorrected: TargetInfo,
      prediction_corrected: TargetInfo,
      xt: DataArray,
      time_prev: jax.Array,
      time_next: jax.Array,
  ) -> GaussianStepKernel:
    """Linear-Gaussian proposal kernel for the bridge step.

    The bridge transition is ``x_s = coeff_x0 x_0 + coeff_xt x_t +
    sigma_step Z`` with coefficients supplied by the interpolant's
    ``bridge_coefficients``.  Since the noise is independent of ``x_0``,
    the SMC proposal ratio between the corrected and uncorrected
    endpoints is the standard :class:`GaussianStepKernel` quadratic form
    -- and is identically zero for a deterministic bridge
    (``sigma_step = 0``: linear / flow-matching).

    Interpolants without a Gaussian bridge (e.g. the geodesic bridge)
    expose no ``bridge_coefficients``; their bridge is deterministic, so
    we return a zero-ratio Dirac kernel.
    """
    interpolant = self.corruption_process.interpolant
    coeff_fn = getattr(interpolant, "bridge_coefficients", None)
    if coeff_fn is None:
      zero = jnp.asarray(0.0, dtype=xt.dtype)
      return GaussianStepKernel(
          coeff_x0=zero, coeff_xt=zero, sigma_step=zero,
          x0_uncorrected=jnp.zeros_like(xt),
          x0_corrected=jnp.zeros_like(xt),
      )
    coeff_x0, coeff_xt, sigma_step = coeff_fn(
        _time_1d(time_prev), _time_1d(time_next),
    )
    return GaussianStepKernel(
        coeff_x0=coeff_x0.reshape(()),
        coeff_xt=coeff_xt.reshape(()),
        sigma_step=sigma_step.reshape(()),
        x0_uncorrected=self._x0(prediction_uncorrected, xt, time_prev),
        x0_corrected=self._x0(prediction_corrected, xt, time_prev),
    )
