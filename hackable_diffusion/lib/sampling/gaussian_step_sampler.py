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

"""Actual implementation of the sampling steps.

This module proposes various implementations but they all have in common
the core logic:

* An `initialize` function that takes a starting state and returns the
  first step of the diffusion process.
* An `update` function that takes the current state and returns the next step.
* A `finalize` function that takes the last state and returns the final
  state.

At every step, the update function takes the current state and returns the next
state. The update is also in charge of computing other auxiliary informations
such as volatility, drifts, etc.

The `InferenceFn is also called within the step and converted into the
relevant representation, for instance score, velocity, etc.
"""

import dataclasses

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.corruption import gaussian
from hackable_diffusion.lib.sampling import base
from hackable_diffusion.lib.sampling import time_scheduling
import jax
import jax.numpy as jnp
import kauldron.ktyping as kt

################################################################################
# MARK: Type Aliases
################################################################################

PRNGKey = hd_typing.PRNGKey

DataArray = hd_typing.DataArray
TargetInfo = hd_typing.TargetInfo

DiffusionStep = base.DiffusionStep
StepInfo = base.StepInfo
SamplerStep = base.SamplerStep

GaussianProcess = gaussian.GaussianProcess
TimeSchedule = time_scheduling.TimeSchedule
StepKernel = base.StepKernel

################################################################################
# MARK: Gaussian proposal kernel
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class GaussianStepKernel(StepKernel):
  """Linear-Gaussian transition kernel for reverse-time Gaussian steppers.

  Every Gaussian-forward stepper (DDIM, SDE, Velocity, AdjustedDDIM,
  Heun) parameterises its step as

      xt_next = coeff_x0 * xhat_0 + coeff_xt * xt + sigma_step * eps,
                eps ~ N(0, I),

  with scalar schedule-dependent coefficients
  ``(coeff_x0, coeff_xt, sigma_step)``.  The ODE and SDE limits are the
  single knob ``sigma_step``:

    - ``sigma_step = 0``: deterministic proposal (ODE / probability
      flow).  Log-ratio is identically zero.
    - ``sigma_step > 0``: stochastic proposal (Euler-Maruyama / DDPM
      ancestral).  Log-ratio is the quadratic Gaussian form.

  ``xhat_0_uncorrected`` and ``xhat_0_corrected`` are captured at
  construction time so :meth:`log_density_ratio` has a uniform
  ``(xt_prev, xt_next)`` signature across all modalities.
  """

  coeff_x0: jax.Array
  coeff_xt: jax.Array
  sigma_step: jax.Array
  x0_uncorrected: jax.Array
  x0_corrected: jax.Array

  def log_density_ratio(
      self, xt_prev: jax.Array, xt_next: jax.Array,
  ) -> jax.Array:
    mu_p = self.coeff_x0 * self.x0_uncorrected + self.coeff_xt * xt_prev
    mu_q = self.coeff_x0 * self.x0_corrected + self.coeff_xt * xt_prev
    sq_p = jnp.sum(
        (xt_next - mu_p).reshape(xt_next.shape[0], -1) ** 2, axis=-1,
    )
    sq_q = jnp.sum(
        (xt_next - mu_q).reshape(xt_next.shape[0], -1) ** 2, axis=-1,
    )
    # sigma_step == 0 is the deterministic / ODE limit: proposal is a
    # Dirac, ratio is identically zero.  Guard the division and mask via
    # jnp.where so the branch works under jit / scan (sigma_step is
    # traced when constructed from the time-dependent schedule).
    deterministic = self.sigma_step == 0.0
    denom = 2.0 * jnp.where(deterministic, 1.0, self.sigma_step ** 2)
    raw = (sq_q - sq_p) / denom
    return jnp.where(deterministic, jnp.zeros_like(raw), raw)


def _scalar(time, schedule_fn) -> jax.Array:
  t = jnp.atleast_1d(time).reshape(-1)[0:1]
  return schedule_fn(t).reshape(())


def _x0_from_prediction(
    corruption_process: GaussianProcess,
    prediction: TargetInfo,
    xt: DataArray,
    time: jax.Array,
) -> jax.Array:
  """Convert any native prediction parameterisation to ``x0``."""
  return corruption_process.convert_predictions(prediction, xt, time)["x0"]


################################################################################
# MARK: SDE Step
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class SdeStep(SamplerStep):
  """Stochastic Differential Equation (SDE) sampling.

  Attributes:
    corruption_process: The corruption process to use.
    churn: The churn parameter.
    stochastic_last_step: Whether the last step is stochastic.
  """

  corruption_process: GaussianProcess
  churn: float
  stochastic_last_step: bool = False

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
      stochastic: bool = True,
  ) -> DiffusionStep:
    current_step_info = current_step.step_info
    xt = current_step.xt

    time = current_step_info.time
    next_time = next_step_info.time
    time = utils.bcast_right(time, xt.ndim)
    next_time = utils.bcast_right(next_time, xt.ndim)

    f = self.corruption_process.schedule.f(time)
    g = self.corruption_process.schedule.g(time)

    score = self.corruption_process.convert_predictions(
        prediction=prediction,
        xt=xt,
        time=time,
    )["score"]

    dt = time - next_time
    z = jax.random.normal(
        key=next_step_info.rng,
        shape=score.shape,
    )

    delta = (
        -f * xt + 0.5 * jnp.square(g) * (1.0 + jnp.square(self.churn)) * score
    )
    mean = xt + delta * dt
    volatility = jnp.sqrt(dt) * g * self.churn

    if stochastic:
      new_xt = mean + volatility * z
    else:
      new_xt = mean

    return DiffusionStep(
        xt=new_xt,
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
    return self.update(
        prediction,
        current_step,
        last_step_info,
        stochastic=self.stochastic_last_step,
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
    """Score-form Euler-Maruyama kernel.

    ``mu = xt + dt * (-f xt + 0.5 g^2 (1 + churn^2) score)`` with
    ``score = (alpha xhat_0 - xt) / sigma^2``.  ``sigma_step = sqrt(dt) g churn``
    (= 0 in the ODE limit ``churn = 0``).
    """
    schedule = self.corruption_process.schedule
    alpha = _scalar(time_prev, schedule.alpha)
    sigma = _scalar(time_prev, schedule.sigma)
    g_t = _scalar(time_prev, schedule.g)
    f_t = _scalar(time_prev, schedule.f)
    dt = _scalar(time_prev, lambda t: t) - _scalar(time_next, lambda t: t)

    sigma2_safe = jnp.maximum(sigma ** 2, 1e-12)
    coeff_x0 = 0.5 * g_t ** 2 * (1.0 + self.churn ** 2) * dt * alpha / sigma2_safe
    coeff_xt = (
        1.0 - dt * f_t
        - 0.5 * g_t ** 2 * (1.0 + self.churn ** 2) * dt / sigma2_safe
    )
    sigma_step = jnp.sqrt(dt) * g_t * self.churn
    return GaussianStepKernel(
        coeff_x0=coeff_x0, coeff_xt=coeff_xt, sigma_step=sigma_step,
        x0_uncorrected=_x0_from_prediction(
            self.corruption_process, prediction_uncorrected, xt, time_prev,
        ),
        x0_corrected=_x0_from_prediction(
            self.corruption_process, prediction_corrected, xt, time_prev,
        ),
    )


################################################################################
# MARK: Adjusted DDIM Step
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class AdjustedDDIMStep(SamplerStep):
  """Adjusted DDIM sampler from https://arxiv.org/pdf/2403.06807.

  We refer to Algorithm 3 of the paper. The strategy used to estimate the
  covariance is the one described page 7, corresponding to 10% of the posterior
  variance if the prior was a factorized Gaussian with variance 0.5.

  Attributes:
    corruption_process: The corruption process to use.
  """

  corruption_process: GaussianProcess

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
    current_step_info = current_step.step_info
    xt = current_step.xt

    time = current_step_info.time
    next_time = next_step_info.time
    time = utils.bcast_right(time, xt.ndim)
    next_time = utils.bcast_right(next_time, xt.ndim)

    alpha = self.corruption_process.schedule.alpha(time)
    sigma = self.corruption_process.schedule.sigma(time)
    next_alpha = self.corruption_process.schedule.alpha(next_time)
    next_sigma = self.corruption_process.schedule.sigma(next_time)

    prediction_dict = self.corruption_process.convert_predictions(
        prediction=prediction,
        xt=xt,
        time=time,
    )
    x0 = prediction_dict["x0"]
    x1 = prediction_dict["x1"]

    # Estimation according to page 7 of the paper.
    xt_var = 0.1 / (2 + jnp.square(alpha / sigma))
    # Ajusted DDIM update.
    next_xt_var = jnp.square(next_alpha - alpha * next_sigma / sigma) * xt_var
    norm_x1 = jnp.mean(jnp.square(x1), keepdims=True)
    new_xt = (
        alpha * x0
        + jnp.sqrt(jnp.square(next_sigma) + next_xt_var / norm_x1)
        * x1
    )

    return DiffusionStep(
        xt=new_xt,
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
    """Deterministic (Dirac) kernel -- ``sigma_step = 0``.

    ``AdjustedDDIMStep``'s adjusted-noise formula has no driving ``z``
    term; the proposal is a deterministic function of ``(xt, xhat_0)``.
    We return ``sigma_step = 0`` and leave the mean coefficients at
    zero since the ratio is identically zero regardless of the mean.
    """
    del prediction_uncorrected, prediction_corrected, time_next
    zero = jnp.asarray(0.0, dtype=xt.dtype)
    return GaussianStepKernel(
        coeff_x0=zero, coeff_xt=zero, sigma_step=zero,
        x0_uncorrected=jnp.zeros_like(xt),
        x0_corrected=jnp.zeros_like(xt),
    )


################################################################################
# MARK: DDIM Step
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class DDIMStep(SamplerStep):
  """DDIM sampler from https://arxiv.org/abs/2010.02502.

  stoch_coeff controls the interpolation between DDIM and DDPM:
  stoch_coeff = 0.0 gives (deterministic) DDIM and stoch_coeff = 1.0 gives DDPM.

  Attributes:
    corruption_process: The corruption process to use.
    stoch_coeff: The interpolation parameter between DDIM and DDPM.
    stochastic_last_step: Whether the last step is stochastic.
  """

  corruption_process: GaussianProcess
  stoch_coeff: float
  stochastic_last_step: bool = False

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
      stochastic: bool = True,
  ) -> DiffusionStep:
    current_step_info = current_step.step_info
    xt = current_step.xt

    time = current_step_info.time
    next_time = next_step_info.time
    time = utils.bcast_right(time, xt.ndim)
    next_time = utils.bcast_right(next_time, xt.ndim)

    x0 = self.corruption_process.convert_predictions(
        prediction=prediction,
        xt=xt,
        time=time,
    )["x0"]
    z = jax.random.normal(key=next_step_info.rng, shape=x0.shape)

    alpha = self.corruption_process.schedule.alpha(time)
    sigma = self.corruption_process.schedule.sigma(time)
    next_alpha = self.corruption_process.schedule.alpha(next_time)
    next_sigma = self.corruption_process.schedule.sigma(next_time)

    # alpha sigma ratios
    r01 = next_sigma / sigma
    r11 = alpha / next_alpha * r01
    r12 = r11 * r01
    r22 = r11 * r11

    # DDIM update
    coeff_xt = self.stoch_coeff * r12 + (1.0 - self.stoch_coeff) * r01
    coeff_x0 = next_alpha * (
        1.0 - self.stoch_coeff * r22 - (1.0 - self.stoch_coeff) * r11
    )
    volatility = next_sigma * jnp.sqrt(
        1.0 - jnp.square(self.stoch_coeff * r11 + (1.0 - self.stoch_coeff))
    )
    new_mean = coeff_xt * xt + coeff_x0 * x0
    if stochastic:
      new_xt = new_mean + volatility * z
    else:
      new_xt = new_mean

    return DiffusionStep(
        xt=new_xt,
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
    return self.update(
        prediction,
        current_step,
        last_step_info,
        stochastic=self.stochastic_last_step,
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
    """Linear-mean-shift DDIM kernel.

    ``mu = coeff_x0 xhat_0 + coeff_xt xt`` with

        coeff_x0 = alpha_s - alpha_r sigma_s sqrt(1 - eta^2) / sigma_r
        coeff_xt = sigma_s sqrt(1 - eta^2) / sigma_r

    and ``sigma_step = sigma_s eta``.  ODE limit ``eta = 0``:
    ``sigma_step = 0`` and the ratio is identically zero.
    """
    schedule = self.corruption_process.schedule
    alpha_r = _scalar(time_prev, schedule.alpha)
    sigma_r = _scalar(time_prev, schedule.sigma)
    alpha_s = _scalar(time_next, schedule.alpha)
    sigma_s = _scalar(time_next, schedule.sigma)
    eta = self.stoch_coeff

    det_factor = jnp.sqrt(jnp.maximum(1.0 - eta ** 2, 0.0))
    coeff_x0 = (
        alpha_s - alpha_r * sigma_s * det_factor / jnp.maximum(sigma_r, 1e-12)
    )
    coeff_xt = sigma_s * det_factor / jnp.maximum(sigma_r, 1e-12)
    sigma_step = sigma_s * eta

    return GaussianStepKernel(
        coeff_x0=coeff_x0, coeff_xt=coeff_xt, sigma_step=sigma_step,
        x0_uncorrected=_x0_from_prediction(
            self.corruption_process, prediction_uncorrected, xt, time_prev,
        ),
        x0_corrected=_x0_from_prediction(
            self.corruption_process, prediction_corrected, xt, time_prev,
        ),
    )


################################################################################
# MARK: Velocity Step
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class VelocityStep(SamplerStep):
  """DDIM sampler from https://arxiv.org/abs/2010.02502.

  epsilon controls the interpolation between DDIM and DDPM:
  epsilon = 0.0 gives (deterministic) DDIM and epsilon = 1.0 gives DDPM.
  """

  corruption_process: GaussianProcess
  epsilon: float

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
      stochastic: bool = True,
  ) -> DiffusionStep:
    current_step_info = current_step.step_info
    xt = current_step.xt

    time = current_step_info.time
    next_time = next_step_info.time
    time = utils.bcast_right(time, xt.ndim)
    next_time = utils.bcast_right(next_time, xt.ndim)

    g = self.corruption_process.schedule.g(time)

    dt = time - next_time

    prediction_dict = self.corruption_process.convert_predictions(
        prediction=prediction,
        xt=xt,
        time=time,
    )
    velocity = prediction_dict["velocity"]
    score = prediction_dict["score"]
    z = jax.random.normal(key=next_step_info.rng, shape=xt.shape)

    delta = -velocity + 0.5 * self.epsilon**2 * g**2 * score
    new_mean = xt + delta * dt
    volatility = jnp.sqrt(dt) * g * self.epsilon

    if stochastic:
      new_xt = new_mean + volatility * z
    else:
      new_xt = new_mean

    return DiffusionStep(
        xt=new_xt,
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
    return self.update(
        prediction,
        current_step,
        last_step_info,
        stochastic=False,
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
    """Velocity-form Euler-Maruyama kernel.

    ``mu = xt + dt * (-velocity(xhat_0) + 0.5 eps^2 g^2 score(xhat_0))``
    with ``velocity = alpha_der xhat_0 + (sigma_der/sigma)(xt - alpha xhat_0)``
    and ``score = (alpha xhat_0 - xt) / sigma^2``.  Schedule derivatives
    are obtained with ``jax.grad``.

    ODE limit ``epsilon = 0``: ``sigma_step = 0``.
    """
    schedule = self.corruption_process.schedule
    alpha = _scalar(time_prev, schedule.alpha)
    sigma = _scalar(time_prev, schedule.sigma)
    g_t = _scalar(time_prev, schedule.g)
    t_scalar = jnp.atleast_1d(time_prev).reshape(-1)[0:1].reshape(())
    t_next_scalar = jnp.atleast_1d(time_next).reshape(-1)[0:1].reshape(())
    dt = t_scalar - t_next_scalar

    alpha_der = jax.grad(
        lambda t: schedule.alpha(t[None]).reshape(())
    )(t_scalar)
    sigma_der = jax.grad(
        lambda t: schedule.sigma(t[None]).reshape(())
    )(t_scalar)

    sigma2_safe = jnp.maximum(sigma ** 2, 1e-12)
    sigma_safe = jnp.maximum(sigma, 1e-12)

    v_x0 = alpha_der - alpha * sigma_der / sigma_safe
    v_xt = sigma_der / sigma_safe
    s_x0 = alpha / sigma2_safe
    s_xt = -1.0 / sigma2_safe

    coeff_x0 = dt * (-v_x0 + 0.5 * self.epsilon ** 2 * g_t ** 2 * s_x0)
    coeff_xt = 1.0 + dt * (-v_xt + 0.5 * self.epsilon ** 2 * g_t ** 2 * s_xt)
    sigma_step = jnp.sqrt(dt) * g_t * self.epsilon

    return GaussianStepKernel(
        coeff_x0=coeff_x0, coeff_xt=coeff_xt, sigma_step=sigma_step,
        x0_uncorrected=_x0_from_prediction(
            self.corruption_process, prediction_uncorrected, xt, time_prev,
        ),
        x0_corrected=_x0_from_prediction(
            self.corruption_process, prediction_corrected, xt, time_prev,
        ),
    )


################################################################################
# MARK: Heun Step
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class HeunStep(SamplerStep):
  """Heun sampler, adapted from https://arxiv.org/abs/2206.00364.

  Importantly, since the Heun sampler is a multi-step sampler and contains one
  internal step, we must ensure that the time schedule used in the sampler has a
  number of steps which is even. The state at each odd step is an **internal**
  state.

  Attributes:
    corruption_process: The corruption process to use.
    time_schedule: The time schedule used.
    num_steps: The total number of steps (including internal steps).
  """

  corruption_process: GaussianProcess
  time_schedule: TimeSchedule
  num_steps: int

  def __post_init__(self):
    if self.num_steps % self.num_internal_steps != 0:
      raise ValueError(
          "The number of steps should be divisible by"
          f" {self.num_internal_steps}, got {self.num_steps}"
      )

  @property
  def num_internal_steps(self):
    return 2

  def _get_all_step_infos(self, rng: PRNGKey, data_spec: DataArray) -> StepInfo:
    return self.time_schedule.all_step_infos(
        rng=rng, num_steps=self.num_steps, data_spec=data_spec
    )

  def _get_specific_step_info(
      self, rng: PRNGKey, data_spec: DataArray, step: int
  ):
    all_step_infos = self._get_all_step_infos(rng=rng, data_spec=data_spec)
    return jax.tree.map(lambda x: x[step], all_step_infos)

  @kt.typechecked
  def initialize(
      self,
      initial_noise: DataArray,
      initial_step_info: StepInfo,
  ) -> DiffusionStep:
    """Initializes the Heun sampler.

    Note that DiffusionStep contains an auxiliary dictionary which is used to
    keep track of the internal step. This dictionary contains the following
    keys:
      - internal_counter: An integer indicating the current internal step.
      - current_update: The current internal step zero.
      - current_velocity_step_one: The current velocity step one.

    The internal counter is updated at each step and initialized at 0. If the
    internal counter is 0, then we apply `first_step`. Otherwise, we apply the
    `second_step`.

    The `current_update` is the state corresponding to the current
    initial_counter=0. For instance at step 3, this contains the information of
    step 2. This `current_update` is updated in `second_step`.

    The `current_velocity_step_one` is the current intermediate velocity. It is
    updated in `first_step`. Combining `current_velocity_step_one` the velocity
    computed in `second_step` gives the final velocity.

    Args:
      initial_noise: The initial noise.
      initial_step_info: The initial step info.

    Returns:
      The initial step.
    """

    current_update = DiffusionStep(
        xt=initial_noise,
        step_info=initial_step_info,
        aux=dict(),
    )
    return DiffusionStep(
        xt=initial_noise,
        step_info=initial_step_info,
        aux=dict(
            internal_counter=jnp.array(0),
            current_update=current_update,
            current_velocity_step_one=initial_noise,  # placeholder
        ),
    )

  @kt.typechecked
  def first_step(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
  ) -> DiffusionStep:
    """First step of the Heun sampler.

    This is the first internal step. It should be called when the internal
    counter is 0.

    Args:
      prediction: The prediction from the corruption process.
      current_step: The current step.
      next_step_info: The next step info.

    Returns:
      The next step.
    """

    current_step_info = current_step.step_info

    xt = current_step.xt

    rng = next_step_info.rng
    # Obtain the next next step info (next external step).
    next_next_step = next_step_info.step + 1
    next_next_step_info = self._get_specific_step_info(
        rng=rng, data_spec=xt, step=next_next_step
    )

    time = current_step_info.time
    next_next_time = next_next_step_info.time
    time = utils.bcast_right(time, xt.ndim)
    next_next_time = utils.bcast_right(next_next_time, xt.ndim)

    # Perform the intermediate step.
    dt = time - next_next_time
    velocity = self.corruption_process.convert_predictions(
        prediction=prediction,
        xt=xt,
        time=time,
    )["velocity"]
    new_xt = xt - dt * velocity

    aux = current_step.aux
    # Update the internal counter and current_velocity_step_one.
    internal_counter = jnp.mod(
        aux["internal_counter"] + 1, self.num_internal_steps
    )
    current_velocity_step_one = velocity
    aux.update(
        dict(
            internal_counter=internal_counter,
            current_velocity_step_one=current_velocity_step_one,
        )
    )

    # Note that we output next_next_step_info and not next_step_info.
    return DiffusionStep(
        xt=new_xt,
        step_info=next_next_step_info,
        aux=aux,
    )

  @kt.typechecked
  def second_step(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
  ) -> DiffusionStep:
    """First step of the Heun sampler.

    This is the second internal step. It should be called when the internal
    counter is 1.

    Args:
      prediction: The prediction from the corruption process.
      current_step: The current step.
      next_step_info: The next step info.

    Returns:
      The next step.
    """
    aux = current_step.aux
    xt = current_step.xt

    current_update = aux["current_update"]
    old_time = current_update.step_info.time
    next_time = next_step_info.time
    old_time = utils.bcast_right(old_time, xt.ndim)
    next_time = utils.bcast_right(next_time, xt.ndim)

    old_velocity = aux["current_velocity_step_one"]
    intermediate_velocity = self.corruption_process.convert_predictions(
        prediction=prediction,
        xt=xt,
        time=next_time,
    )["velocity"]
    # Note that here we use next_step_info and not current_step_info.
    # This is because `first_step` outputs next_next_step_info.

    # Perform the final step.

    old_xt = current_update.xt
    dt = old_time - next_time
    new_xt = old_xt - dt * (old_velocity + intermediate_velocity) / 2

    # Update the internal counter and the current_update
    internal_counter = jnp.mod(
        aux["internal_counter"] + 1, self.num_internal_steps
    )
    current_update = DiffusionStep(
        xt=new_xt,
        step_info=next_step_info,
        aux=dict(),
    )
    aux.update(
        dict(
            internal_counter=internal_counter,
            current_update=current_update,
        )
    )

    return DiffusionStep(
        xt=new_xt,
        step_info=next_step_info,
        aux=aux,
    )

  @kt.typechecked
  def update(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
  ) -> DiffusionStep:
    """Update function for the Heun sampler.

    Depending on the internal counter, we call either `first_step` or
    `second_step`.

    Args:
      prediction: The prediction from the corruption process.
      current_step: The current step.
      next_step_info: The next step info.

    Returns:
      The next step.
    """

    return jax.lax.switch(
        current_step.aux["internal_counter"],
        [self.first_step, self.second_step],
        prediction,
        current_step,
        next_step_info,
    )

  @kt.typechecked
  def finalize(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      last_step_info: StepInfo,
  ) -> DiffusionStep:
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
    """Deterministic predictor-corrector kernel -- ``sigma_step = 0``.

    Heun's method is a two-stage deterministic ODE integrator with no
    driving noise.  Both the predictor (first_step) and the corrector
    (second_step) produce a deterministic ``xt_next`` from
    ``(xt, xhat_0)``, so the Gaussian log-ratio is identically zero
    regardless of which internal stage produced the transition.
    """
    del prediction_uncorrected, prediction_corrected, time_next
    zero = jnp.asarray(0.0, dtype=xt.dtype)
    return GaussianStepKernel(
        coeff_x0=zero, coeff_xt=zero, sigma_step=zero,
        x0_uncorrected=jnp.zeros_like(xt),
        x0_corrected=jnp.zeros_like(xt),
    )
