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

"""Analytic Gaussian-mixture target for posterior-bridge integration tests.

For a ``K``-component isotropic Gaussian mixture ``p_0 = sum_k w_k
N(m_k, sigma_0^2 I_D)``, the variance-preserving Gaussian bridge
``X_t = alpha_t X_0 + sigma_t Z``, ``alpha_t^2 + sigma_t^2 = 1``, has
a closed-form posterior at every ``x_t`` -- a Gaussian mixture with
the same number of components and analytic responsibilities.  This
module exposes:

  - :class:`GaussianMixture`: the endpoint distribution + closed-form
    log-pdf and sampler.
  - :class:`GaussianMixtureBridge`: the bridge wrapper with analytic
    ``p_{0|t}(. | x_t)``, posterior sampler, and the closed-form
    ``H_t(x_t) = E[L_y(X_0) | X_t = x_t]`` for Gaussian likelihood
    tilts ``L_y(x_0) = exp(-||x_0 - y||^2 / (2 sigma_y^2))``.
  - :func:`posterior_sampler_inference_fn`: wraps the analytic
    posterior sampler in the codebase's ``InferenceFn`` Protocol so
    framework code can drive it as a "ground-truth" model.

Math reference -- for one component ``k`` at state ``x_t``:

    v_t       = alpha_t^2 sigma_0^2 + sigma_t^2  (marginal variance)
    kappa_t   = alpha_t sigma_0^2 / v_t          (posterior gain)
    tau_bar^2 = sigma_0^2 sigma_t^2 / v_t        (posterior var)
    m_{k, t}(x_t) = m_k + kappa_t (x_t - alpha_t m_k)

Mixture responsibility:

    r_k(x_t) propto w_k * N(x_t; alpha_t m_k, v_t I_D),

so ``p_{0|t}(. | x_t) = sum_k r_k(x_t) N(m_{k, t}(x_t), tau_bar^2 I_D)``.

Gaussian-tilt potential (closed form via the Gaussian-Gaussian
convolution identity):

    H_t(x_t) = sum_k r_k(x_t) (tau_bar^2 / (tau_bar^2 + sigma_y^2))^(D/2)
                              exp(-||m_{k, t}(x_t) - y||^2
                                    / (2 (tau_bar^2 + sigma_y^2))).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp


@dataclasses.dataclass(kw_only=True, frozen=True)
class GaussianMixture:
  """Isotropic K-component Gaussian mixture in ``R^D``.

  Attributes:
    weights: ``[K]`` mixture weights, must sum to 1 and lie in [0, 1].
    means: ``[K, D]`` component means.
    component_var: shared scalar variance ``sigma_0^2`` of each
      component.
  """

  weights: jax.Array
  means: jax.Array
  component_var: float

  @property
  def num_components(self) -> int:
    return int(self.means.shape[0])

  @property
  def dim(self) -> int:
    return int(self.means.shape[-1])

  def log_pdf(self, x: jax.Array) -> jax.Array:
    """Log-density at ``x`` of shape ``[..., D]``.  Returns ``[...]``."""
    diff = x[..., None, :] - self.means                        # [..., K, D]
    sq = jnp.sum(diff ** 2, axis=-1)                           # [..., K]
    log_w = jnp.log(self.weights)                              # [K]
    log_norm = -0.5 * self.dim * jnp.log(
        2.0 * jnp.pi * self.component_var
    )
    log_pdf_per_k = log_w + log_norm - sq / (2.0 * self.component_var)
    return jax.nn.logsumexp(log_pdf_per_k, axis=-1)

  def sample(self, rng: jax.Array, n: int) -> jax.Array:
    """Draw ``n`` i.i.d. samples; return ``[n, D]``."""
    rng_k, rng_z = jax.random.split(rng)
    k_idx = jax.random.categorical(rng_k, jnp.log(self.weights), shape=(n,))
    centers = self.means[k_idx]                                # [n, D]
    noise = jax.random.normal(rng_z, (n, self.dim), dtype=self.means.dtype)
    return centers + jnp.sqrt(self.component_var) * noise


@dataclasses.dataclass(kw_only=True, frozen=True)
class GaussianMixtureBridge:
  """VP Gaussian bridge ``X_t = alpha_t X_0 + sigma_t Z`` over a GMM ``p_0``.

  ``alpha`` is supplied as a scalar callable ``alpha_fn(t)`` so this
  module is independent of the ``Schedule`` Protocol -- callers who
  have a Schedule can pass ``schedule.alpha`` after wrapping for the
  required input shape, see :func:`alpha_from_schedule`.
  """

  mixture: GaussianMixture
  alpha_fn: 'object'  # callable: scalar t -> scalar alpha(t)

  def _alpha(self, t: jax.Array) -> jax.Array:
    return self.alpha_fn(t)

  def _sigma_sq(self, t: jax.Array) -> jax.Array:
    return 1.0 - self._alpha(t) ** 2

  def corrupt(self, rng: jax.Array, x0: jax.Array, t: jax.Array) -> jax.Array:
    """Sample ``X_t = alpha_t X_0 + sigma_t Z`` for ``x0`` shape ``[B, D]``."""
    z = jax.random.normal(rng, x0.shape, dtype=x0.dtype)
    sigma_t = jnp.sqrt(self._sigma_sq(t))
    return self._alpha(t) * x0 + sigma_t * z

  def posterior_responsibilities(
      self, x_t: jax.Array, t: jax.Array,
  ) -> jax.Array:
    """Bayes responsibilities ``r_k(x_t)`` of shape ``[..., K]``."""
    alpha_t = self._alpha(t)
    sigma_sq = self._sigma_sq(t)
    v_t = alpha_t ** 2 * self.mixture.component_var + sigma_sq
    diff = x_t[..., None, :] - alpha_t * self.mixture.means    # [..., K, D]
    sq = jnp.sum(diff ** 2, axis=-1)                           # [..., K]
    log_w = jnp.log(self.mixture.weights)                      # [K]
    log_marginals = log_w - 0.5 * sq / v_t
    return jax.nn.softmax(log_marginals, axis=-1)              # [..., K]

  def posterior_component_means(
      self, x_t: jax.Array, t: jax.Array,
  ) -> jax.Array:
    """Posterior component means ``m_{k, t}(x_t)`` of shape ``[..., K, D]``."""
    alpha_t = self._alpha(t)
    sigma_sq = self._sigma_sq(t)
    v_t = alpha_t ** 2 * self.mixture.component_var + sigma_sq
    kappa_t = alpha_t * self.mixture.component_var / v_t
    # m_{k, t} = m_k + kappa_t (x_t - alpha_t m_k)
    return self.mixture.means + kappa_t * (
        x_t[..., None, :] - alpha_t * self.mixture.means
    )                                                          # [..., K, D]

  def posterior_component_var(self, t: jax.Array) -> jax.Array:
    """Posterior component variance ``tau_bar^2`` (shared, scalar)."""
    alpha_t = self._alpha(t)
    sigma_sq = self._sigma_sq(t)
    v_t = alpha_t ** 2 * self.mixture.component_var + sigma_sq
    return self.mixture.component_var * sigma_sq / v_t

  def posterior_mean(self, x_t: jax.Array, t: jax.Array) -> jax.Array:
    """Posterior mean ``E[X_0 | X_t = x_t]`` of shape ``[..., D]``."""
    r = self.posterior_responsibilities(x_t, t)                # [..., K]
    component_means = self.posterior_component_means(x_t, t)   # [..., K, D]
    return jnp.sum(r[..., None] * component_means, axis=-2)    # [..., D]

  def sample_posterior(
      self, rng: jax.Array, x_t: jax.Array, t: jax.Array,
  ) -> jax.Array:
    """Draw one posterior sample at each ``x_t`` of shape ``[B, D]``."""
    if x_t.ndim != 2:
      raise ValueError(
          f"sample_posterior expects x_t of shape [B, D]; got {x_t.shape}."
      )
    rng_k, rng_z = jax.random.split(rng)
    r = self.posterior_responsibilities(x_t, t)                # [B, K]
    k_idx = jax.random.categorical(rng_k, jnp.log(r))          # [B]
    component_means = self.posterior_component_means(x_t, t)   # [B, K, D]
    selected_means = jnp.take_along_axis(
        component_means, k_idx[:, None, None].astype(jnp.int32), axis=1,
    ).squeeze(axis=1)                                           # [B, D]
    z = jax.random.normal(rng_z, x_t.shape, dtype=x_t.dtype)
    return selected_means + jnp.sqrt(self.posterior_component_var(t)) * z

  def gaussian_tilt_log_potential(
      self,
      x_t: jax.Array,
      t: jax.Array,
      y: jax.Array,
      sigma_y: float,
  ) -> jax.Array:
    """Closed-form ``log H_t(x_t)`` for ``L_y(x_0) = exp(-||x_0 - y||^2
    / (2 sigma_y^2))``.

    H_t(x_t)
      = sum_k r_k(x_t) E[exp(-||X_0 - y||^2 / (2 sigma_y^2)) | X_t, S=k]
      = sum_k r_k(x_t) (sigma_y^2 / (tau_bar^2 + sigma_y^2))^(D/2)
                       exp(-||m_{k, t}(x_t) - y||^2
                            / (2 (tau_bar^2 + sigma_y^2))).

    Derivation: ``L_y = (2 pi sigma_y^2)^(D/2) phi(.; y, sigma_y^2 I)``,
    so ``E_{X_0 ~ N(mu, tau_bar^2 I)} L_y(X_0)`` is a Gaussian
    convolution and reduces to the displayed product.

    Returns log H_t shaped ``[...]`` (matching ``x_t`` leading axes).
    """
    r = self.posterior_responsibilities(x_t, t)                # [..., K]
    component_means = self.posterior_component_means(x_t, t)   # [..., K, D]
    tau_bar_sq = self.posterior_component_var(t)
    sigma_y_sq = float(sigma_y) ** 2
    denom = tau_bar_sq + sigma_y_sq
    D = self.mixture.dim
    # Normalisation: ``(sigma_y^2 / (tau_bar^2 + sigma_y^2))^(D/2)``,
    # not ``tau_bar^2 / (...)``.
    log_norm_per_k = (D / 2.0) * jnp.log(sigma_y_sq / denom)
    diff = component_means - y[..., None, :]                   # [..., K, D]
    sq = jnp.sum(diff ** 2, axis=-1)                           # [..., K]
    log_per_k = log_norm_per_k - sq / (2.0 * denom)            # [..., K]
    log_r = jnp.log(jnp.clip(r, 1e-30, None))                  # [..., K]
    return jax.nn.logsumexp(log_r + log_per_k, axis=-1)        # [...]


def posterior_sampler_inference_fn(bridge: GaussianMixtureBridge):
  """Wrap a :class:`GaussianMixtureBridge` as an ``InferenceFn``.

  Returns a callable matching the codebase's ``InferenceFn`` Protocol
  -- ``(time, xt, conditioning, rng)`` -> ``{"x0": x0_sample}`` -- so
  it can be used directly as the inference fn passed to
  :class:`ConditionalDiffusionSampler`.  Each call draws ONE posterior
  sample per batch element from the analytic posterior; calling it
  ``R`` times via :func:`make_posterior_cloud_fn` produces an
  ``R``-sample posterior cloud at the same ``x_t``.

  This is the "ground truth" inference fn used in integration tests:
  it lets the test isolate framework wiring from model error.
  """

  def inference_fn(time, xt, conditioning=None, rng=None):
    del conditioning
    if rng is None:
      raise ValueError(
          "posterior_sampler_inference_fn is stochastic; rng must be "
          "provided.  The sampler does this automatically when the "
          "inference fn declares an ``rng`` parameter."
      )
    # Time may be passed as scalar / 1-d / shaped-like-xt[..., :1];
    # the bridge expects a scalar to evaluate alpha.
    t_scalar = jnp.atleast_1d(time).reshape(-1)[0]
    x0 = bridge.sample_posterior(rng, xt, t_scalar)
    return {"x0": x0}

  return inference_fn


def alpha_from_schedule(schedule):
  """Wrap a Schedule's ``alpha`` method as a scalar callable.

  The bridge module does not import the Schedule Protocol; it just
  needs ``alpha_fn(t) -> scalar``.  This shim makes any ``Schedule``
  with a 1-D-array-typed ``alpha`` interoperable.
  """

  def alpha_fn(t):
    t1 = jnp.atleast_1d(jnp.asarray(t))
    return schedule.alpha(t1).reshape(())

  return alpha_fn
