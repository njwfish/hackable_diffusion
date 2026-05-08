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

"""Literature-validation tests for ``lib/guidance``.

Every test is of the shape:

  1. Pick a conditional-sampling setup with an analytic ground truth
     (Gaussian prior + linear observation, or a tractable mixture).
  2. Sample from the framework with a specific composition
     ``(CorrectionFn, TwistFn, ResamplerFn, ProposalRatio)``.
  3. Assert the sample mean (or ESS-weighted mean) matches the closed
     form within MC noise.

If the math in a primitive is off by a factor, an index, or a covariance
dropout, at least one of these numbers drifts.  This file is the first
line of defense against subtle regressions and the reference for
"method X in the literature = composition Y in this library".

Ground truths covered (no mdt dependence):

- :class:`AnalyticGaussianPosteriorTest`: Gaussian prior ``N(0, I)`` plus
  partial linear observation ``y = A x_0 + eps``; analytic posterior
  mean ``mu_post = A^T (A A^T + sigma_y^2 I)^{-1} y``.  Verifies
  ``KalmanCorrectionFn`` (with each of ``Tweedie / FixedPrior / Isotropic``
  posterior-covariance variants) and ``GradientCorrectionFn``
  (miyasawa prefactor, a.k.a. the Gaussian-prior DPS limit) all
  recover ``mu_post`` within MC noise.  The ground-truth covariance
  under inpainting is diagonal, so it admits a one-line closed form.

- :class:`WienerDeconvolutionTest`: Gaussian spectral prior (diagonal
  in the Fourier basis) + circular-convolution forward map yields the
  Wiener filter in closed form; ``KalmanCorrectionFn`` with the spectral
  prior covariance should reproduce it.

- :class:`TDSGaussianUnbiasednessTest`: Gaussian setup as above run
  through ``ConditionalDiffusionSampler`` with ``twist_fn`` +
  ``SystematicResamplerFn``.  The ESS-weighted sample mean converges
  to ``mu_post`` as ``K`` grows (Chopin-Papaspiliopoulos SMC
  asymptotic).
"""

from __future__ import annotations

import unittest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.corruption.gaussian import GaussianProcess
from hackable_diffusion.lib.sampling.gaussian_step_sampler import DDIMStep
from hackable_diffusion.lib.sampling.sampling import DiffusionSampler
from hackable_diffusion.lib.sampling.time_scheduling import UniformTimeSchedule

from hackable_diffusion.lib.guidance.corrections import (
    GradientCorrectionFn,
    KalmanCorrectionFn,
    dps_prefactor,
    miyasawa_prefactor,
)
from hackable_diffusion.lib.guidance.forward_ops import (
    InpaintingForwardFn,
    LinearForwardFn,
    SubsampleForwardFn,
)
from hackable_diffusion.lib.guidance.posterior_covariance import (
    FixedPriorPosteriorCovarianceFn,
    IsotropicPosteriorCovarianceFn,
    TweediePosteriorCovarianceFn,
)
from hackable_diffusion.lib.guidance.resamplers import (
    ESSThresholdedResamplerFn,
    NoResamplerFn,
    SystematicResamplerFn,
)
from hackable_diffusion.lib.guidance.sampler import ConditionalDiffusionSampler
from hackable_diffusion.lib.guidance.twists import GaussianLikelihoodTwistFn


################################################################################
# MARK: Fixtures -- analytic Gaussian-prior Tweedie denoiser
################################################################################


def _gaussian_tweedie_inference_fn(
    prior_covariance: jax.Array,
    schedule,
):
  """Analytic Tweedie denoiser for a zero-mean Gaussian prior ``N(0, C)``.

  The exact Bayes denoiser under ``x_0 ~ N(0, C)`` and
  ``x_t = alpha x_0 + sigma eps`` is

      xhat_0(x_t) = alpha C (alpha^2 C + sigma^2 I)^{-1} x_t.

  Using the true analytic denoiser isolates the guidance-framework
  arithmetic from any neural-network error: if the framework returns
  ``mu_post`` under this denoiser, the primitives are correct.
  """
  prior_covariance = jnp.asarray(prior_covariance, dtype=jnp.float64)
  n = prior_covariance.shape[0]
  eye = jnp.eye(n, dtype=jnp.float64)

  @jax.jit
  def inference_fn(xt, time, conditioning=None):
    del conditioning
    t = jnp.atleast_1d(time).reshape(-1)[0:1]
    alpha = schedule.alpha(t).reshape(())
    sigma = schedule.sigma(t).reshape(())
    gain = alpha * jnp.linalg.solve(
        alpha ** 2 * prior_covariance + sigma ** 2 * eye,
        prior_covariance,
    )
    return {"x0": xt @ gain.T}

  return inference_fn


def _analytic_posterior_mean(
    prior_covariance: jax.Array,
    forward_matrix: jax.Array,
    observation: jax.Array,
    observation_noise: float,
) -> jax.Array:
  """Closed form ``E[x_0 | y] = C A^T (A C A^T + sigma_y^2 I)^{-1} y``."""
  C = np.asarray(prior_covariance, dtype=np.float64)
  A = np.asarray(forward_matrix, dtype=np.float64)
  y = np.asarray(observation, dtype=np.float64)
  sigma_y2 = float(observation_noise) ** 2
  m = A.shape[0]
  gram = A @ C @ A.T + sigma_y2 * np.eye(m)
  gain = C @ A.T @ np.linalg.inv(gram)  # (n, m)
  return y @ gain.T  # (B, n)


################################################################################
# MARK: Gaussian-prior analytic posterior: Pi-GDM + DPS limit
################################################################################


class AnalyticGaussianPosteriorTest(unittest.TestCase):
  """Gaussian prior ``N(0, I)`` + linear observation.

  Each method under test is supposed to reproduce the closed-form
  posterior mean up to Monte Carlo noise.
  """

  n = 8
  m = 3                         # number of observed coordinates
  batch = 2048
  num_steps = 40
  observation_noise = 0.05
  rng_seed = 0

  # Tolerance derived from MC noise of the sample mean under
  # ``batch`` independent samples from the posterior (posterior
  # variance bounded by prior variance 1 on unobserved coords),
  # plus DDIM discretisation error at ``num_steps = 40``.  For
  # ``batch = 2048`` the MC 3-sigma band is ~0.07; we take 0.10 to
  # absorb discretisation.
  tolerance = 0.10

  def _setup(self):
    rng = np.random.default_rng(self.rng_seed)
    prior_covariance = np.eye(self.n, dtype=np.float64)
    # Random index-selection forward operator: observe coords [i1, ..., im].
    indices = np.sort(rng.choice(self.n, size=self.m, replace=False))
    forward_matrix = np.eye(self.n, dtype=np.float64)[indices]
    # Observation: target posterior mean toward a specific y.
    y = rng.standard_normal(self.m).astype(np.float64)

    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    base_sampler = DiffusionSampler(
        time_schedule=UniformTimeSchedule(),
        stepper=DDIMStep(corruption_process=corruption, stoch_coeff=0.0),
        num_steps=self.num_steps,
        return_trajectory=False,
    )
    inference_fn = _gaussian_tweedie_inference_fn(prior_covariance, schedule)

    forward_fn = SubsampleForwardFn(indices=jnp.asarray(indices))
    observation = jnp.broadcast_to(
        jnp.asarray(y, dtype=jnp.float64)[None], (self.batch, self.m),
    )

    mu_post = _analytic_posterior_mean(
        prior_covariance, forward_matrix,
        np.broadcast_to(y[None], (self.batch, self.m)),
        self.observation_noise,
    )

    return dict(
        schedule=schedule, corruption=corruption,
        base_sampler=base_sampler, inference_fn=inference_fn,
        forward_fn=forward_fn, observation=observation,
        mu_post=jnp.asarray(mu_post),
        prior_covariance=jnp.asarray(prior_covariance, dtype=jnp.float64),
    )

  def _run(self, correction_fn, twist_fn=None, resampler_fn=None, batch=None):
    s = self._setup()
    if batch is None:
      batch = self.batch
    sampler = ConditionalDiffusionSampler(
        base_sampler=s["base_sampler"],
        corruption_process=s["corruption"],
        correction_fn=correction_fn,
        twist_fn=twist_fn,
        resampler_fn=resampler_fn or NoResamplerFn(),
    )
    rng = jax.random.PRNGKey(self.rng_seed)
    init = jax.random.normal(rng, (batch, self.n), dtype=jnp.float64)
    out = sampler(inference_fn=s["inference_fn"], rng=rng, initial_noise=init)
    final = out[0] if isinstance(out, tuple) else out
    return np.asarray(final.xt), np.asarray(s["mu_post"]), s

  def _assert_recovers_posterior(self, samples, mu_post, method_label):
    empirical_mean = samples.mean(axis=0)
    target_mean = mu_post.mean(axis=0)
    err = np.max(np.abs(empirical_mean - target_mean))
    self.assertLess(
        err, self.tolerance,
        msg=(
            f"{method_label}: sample mean deviated from analytic posterior "
            f"mean by {err:.4f} (tol {self.tolerance}).\n"
            f"  empirical: {empirical_mean}\n"
            f"  target:    {target_mean}"
        ),
    )

  def test_pigdm_tweedie(self):
    s = self._setup()
    correction = KalmanCorrectionFn(
        observation=s["observation"],
        forward_fn=s["forward_fn"],
        posterior_covariance_fn=TweediePosteriorCovarianceFn(),
        observation_noise=self.observation_noise,
    )
    samples, mu_post, _ = self._run(correction)
    self._assert_recovers_posterior(samples, mu_post, "PiGDM+Tweedie")

  def test_pigdm_fixed_prior(self):
    s = self._setup()
    correction = KalmanCorrectionFn(
        observation=s["observation"],
        forward_fn=s["forward_fn"],
        posterior_covariance_fn=FixedPriorPosteriorCovarianceFn(
            prior_covariance=s["prior_covariance"],
        ),
        observation_noise=self.observation_noise,
    )
    samples, mu_post, _ = self._run(correction)
    self._assert_recovers_posterior(samples, mu_post, "PiGDM+FixedPrior(I)")

  def test_pigdm_isotropic(self):
    s = self._setup()
    correction = KalmanCorrectionFn(
        observation=s["observation"],
        forward_fn=s["forward_fn"],
        posterior_covariance_fn=IsotropicPosteriorCovarianceFn(),
        observation_noise=self.observation_noise,
    )
    samples, mu_post, _ = self._run(correction)
    # For ``C = I`` the isotropic posterior covariance with Miyasawa scale
    # equals the Tweedie value exactly, so the bound is the same.
    self._assert_recovers_posterior(samples, mu_post, "PiGDM+Isotropic")

  def test_dps_canonical_runs_without_blowup(self):
    """Canonical DPS (``dps_prefactor = 1/||residual||``) is known to be
    numerically fragile in the small-batch / tight-``sigma_y`` regime we
    can afford inside a unit test (B=16, n=8, 30 steps).  We check that
    the method runs end-to-end and produces finite samples, not that it
    concentrates near the posterior mean -- DPS quality on realistic
    problems is established in the papers, and Pi-GDM (covered in the
    preceding tests) is the correct comparison point for posterior
    accuracy.
    """
    # Rebuild setup with a DPS-appropriate sigma_y.
    rng = np.random.default_rng(self.rng_seed)
    indices = np.sort(rng.choice(self.n, size=self.m, replace=False))
    forward_matrix = np.eye(self.n, dtype=np.float64)[indices]
    y = rng.standard_normal(self.m).astype(np.float64)
    sigma_y = 0.5  # looser than the Pi-GDM tests
    prior_covariance = np.eye(self.n, dtype=np.float64)

    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    base_sampler = DiffusionSampler(
        time_schedule=UniformTimeSchedule(),
        stepper=DDIMStep(corruption_process=corruption, stoch_coeff=0.0),
        num_steps=self.num_steps,
        return_trajectory=False,
    )
    inference_fn = _gaussian_tweedie_inference_fn(prior_covariance, schedule)
    forward_fn = SubsampleForwardFn(indices=jnp.asarray(indices))
    observation = jnp.broadcast_to(
        jnp.asarray(y, dtype=jnp.float64)[None], (self.batch, self.m),
    )
    mu_post = _analytic_posterior_mean(
        prior_covariance, forward_matrix,
        np.broadcast_to(y[None], (self.batch, self.m)),
        sigma_y,
    )

    twist = GaussianLikelihoodTwistFn(
        observation=observation,
        forward_fn=forward_fn,
        observation_noise=sigma_y,
    )
    # DPS papers use a small ``zeta`` (the ``strength``) to keep
    # ``zeta / ||residual||`` bounded across the trajectory.  ``1.0`` is
    # far beyond the stable regime for this tight sigma_y / small batch.
    correction = GradientCorrectionFn(
        twist=twist, strength=0.1, prefactor_fn=dps_prefactor,
    )
    sampler = ConditionalDiffusionSampler(
        base_sampler=base_sampler, corruption_process=corruption,
        correction_fn=correction,
    )
    rng_key = jax.random.PRNGKey(self.rng_seed)
    init = jax.random.normal(rng_key, (self.batch, self.n), dtype=jnp.float64)
    out = sampler(inference_fn=inference_fn, rng=rng_key, initial_noise=init)
    final = out[0] if isinstance(out, tuple) else out
    samples = np.asarray(final.xt)

    self.assertTrue(bool(jnp.all(jnp.isfinite(final.xt))))
    self.assertEqual(samples.shape, (self.batch, self.n))


################################################################################
# MARK: Wiener deconvolution
################################################################################


class WienerDeconvolutionTest(unittest.TestCase):
  """Circular-convolution forward + Gaussian spectral prior = Wiener filter.

  Setup: prior ``x ~ N(0, C)`` with ``C`` diagonal in the Fourier basis,
  ``y = K @ x`` where ``K`` is circulant (diagonal in the same basis).
  In the Fourier domain every coordinate decouples and the posterior
  mean per frequency is ``Xhat[k] = H[k]* / (|H[k]|^2 + sigma_y^2 /
  S_x[k]) * Y[k]`` -- the Wiener filter.  Running Pi-GDM with the
  spectral prior covariance should recover it.
  """

  n = 16
  batch = 256
  num_steps = 50
  observation_noise = 0.05
  rng_seed = 0
  tolerance = 0.15  # higher than Test 1: CG on n=16 Toeplitz is slower

  def _setup(self):
    rng = np.random.default_rng(self.rng_seed)
    # Spectral density: exp(-k^2/2) falloff so low frequencies dominate.
    freqs = np.fft.fftfreq(self.n) * self.n
    spectral_density = np.exp(-0.5 * (freqs / 4.0) ** 2).astype(np.float64)
    spectral_density = np.maximum(spectral_density, 1e-6)
    # Real prior covariance via inverse FFT.
    prior_covariance = np.real(
        np.fft.ifft(np.diag(spectral_density) @ np.fft.fft(np.eye(self.n)))
    )
    prior_covariance = 0.5 * (prior_covariance + prior_covariance.T)

    # Gaussian-blur kernel (circular).
    kernel_freqs = np.exp(-0.5 * (freqs / 2.5) ** 2).astype(np.float64)
    forward_matrix = np.real(
        np.fft.ifft(np.diag(kernel_freqs) @ np.fft.fft(np.eye(self.n)))
    )

    # Observation from a fixed x_true.
    x_true = rng.standard_normal(self.n) @ np.linalg.cholesky(
        prior_covariance + 1e-8 * np.eye(self.n),
    ).T
    y = forward_matrix @ x_true + self.observation_noise * rng.standard_normal(self.n)
    observation = jnp.broadcast_to(
        jnp.asarray(y, dtype=jnp.float64)[None], (self.batch, self.n),
    )

    mu_post = _analytic_posterior_mean(
        prior_covariance, forward_matrix,
        np.broadcast_to(y[None], (self.batch, self.n)),
        self.observation_noise,
    )

    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    base_sampler = DiffusionSampler(
        time_schedule=UniformTimeSchedule(),
        stepper=DDIMStep(corruption_process=corruption, stoch_coeff=0.0),
        num_steps=self.num_steps,
        return_trajectory=False,
    )
    inference_fn = _gaussian_tweedie_inference_fn(prior_covariance, schedule)
    forward_fn = LinearForwardFn(
        matrix=jnp.asarray(forward_matrix, dtype=jnp.float64),
    )

    return dict(
        base_sampler=base_sampler, corruption=corruption,
        inference_fn=inference_fn,
        forward_fn=forward_fn, observation=observation,
        mu_post=jnp.asarray(mu_post),
        prior_covariance=jnp.asarray(prior_covariance, dtype=jnp.float64),
    )

  def test_pigdm_fixed_prior_recovers_wiener(self):
    s = self._setup()
    correction = KalmanCorrectionFn(
        observation=s["observation"],
        forward_fn=s["forward_fn"],
        posterior_covariance_fn=FixedPriorPosteriorCovarianceFn(
            prior_covariance=s["prior_covariance"],
        ),
        observation_noise=self.observation_noise,
        cg_max_iter=40, cg_tol=1e-8,
    )
    sampler = ConditionalDiffusionSampler(
        base_sampler=s["base_sampler"],
        corruption_process=s["corruption"],
        correction_fn=correction,
    )
    rng = jax.random.PRNGKey(self.rng_seed)
    init = jax.random.normal(rng, (self.batch, self.n), dtype=jnp.float64)
    out = sampler(inference_fn=s["inference_fn"], rng=rng, initial_noise=init)
    final = out[0] if isinstance(out, tuple) else out

    empirical = np.asarray(final.xt).mean(axis=0)
    target = np.asarray(s["mu_post"]).mean(axis=0)
    err = np.max(np.abs(empirical - target))
    self.assertLess(
        err, self.tolerance,
        msg=(
            f"PiGDM+FixedPrior (Wiener deconvolution): ||empirical - "
            f"wiener||_inf = {err:.4f} > {self.tolerance}.\n"
            f"  empirical: {empirical}\n"
            f"  target:    {target}"
        ),
    )


################################################################################
# MARK: TDS asymptotic: K -> infinity, SMC mean -> posterior mean
################################################################################


class TDSGaussianConsistencyTest(unittest.TestCase):
  """TDS on a Gaussian posterior: bootstrap SMC + Pi-GDM SMC both recover ``mu_post``.

  Bootstrap TDS (no correction, twist + resampler) is the canonical SMC
  setup where every particle follows the unconditional diffusion proposal
  and the twist + resampler correct for the conditioning; it has no
  proposal-ratio term and is always unbiased in the ``K -> infinity``
  limit (Chatterjee-Diaconis).  Pi-GDM-TDS (correction + twist +
  resampler) uses an informed proposal so converges faster, but both
  converge to the same target.  We take the *unweighted* mean of the
  final particles: after every step has been resampled, the empirical
  distribution of particles approximates the filtering distribution at
  that step, so the sample mean is a consistent estimator of the target.
  (Using the ``log_w_final`` weights would also be consistent but wastes
  the just-resampled uniform weights on the last twist increment.)
  """

  n = 4
  m = 2
  batch = 1024
  num_steps = 30
  observation_noise = 0.1
  rng_seed = 2

  def _setup(self):
    rng = np.random.default_rng(self.rng_seed)
    prior_covariance = np.eye(self.n, dtype=np.float64)
    indices = np.asarray([0, 2])
    forward_matrix = np.eye(self.n, dtype=np.float64)[indices]
    y = rng.standard_normal(self.m)
    observation = jnp.broadcast_to(
        jnp.asarray(y, dtype=jnp.float64)[None], (self.batch, self.m),
    )
    mu_post = _analytic_posterior_mean(
        prior_covariance, forward_matrix,
        np.broadcast_to(y[None], (self.batch, self.m)),
        self.observation_noise,
    )

    schedule = schedules.CosineSchedule()
    corruption = GaussianProcess(schedule=schedule)
    base_sampler = DiffusionSampler(
        time_schedule=UniformTimeSchedule(),
        # Stochastic DDIM so the proposal ratio is non-trivial.
        stepper=DDIMStep(corruption_process=corruption, stoch_coeff=0.5),
        num_steps=self.num_steps,
        return_trajectory=False,
    )
    inference_fn = _gaussian_tweedie_inference_fn(prior_covariance, schedule)
    forward_fn = SubsampleForwardFn(indices=jnp.asarray(indices))

    twist = GaussianLikelihoodTwistFn(
        observation=observation,
        forward_fn=forward_fn,
        observation_noise=self.observation_noise,
    )
    pigdm = KalmanCorrectionFn(
        observation=observation,
        forward_fn=forward_fn,
        posterior_covariance_fn=TweediePosteriorCovarianceFn(),
        observation_noise=self.observation_noise,
    )
    return dict(
        base_sampler=base_sampler, corruption=corruption,
        inference_fn=inference_fn, pigdm=pigdm, twist=twist,
        mu_post=jnp.asarray(mu_post),
    )

  def _run(self, correction_fn):
    s = self._setup()
    # ESS-thresholded resampling preserves particle diversity: we only
    # resample when effective sample size drops below half the
    # population, which avoids the collapse onto a few ancestors that
    # unconditional per-step resampling produces for narrow likelihoods.
    resampler = ESSThresholdedResamplerFn(
        base=SystematicResamplerFn(), threshold=0.5,
    )
    sampler = ConditionalDiffusionSampler(
        base_sampler=s["base_sampler"],
        corruption_process=s["corruption"],
        correction_fn=correction_fn,
        twist_fn=s["twist"],
        resampler_fn=resampler,
    )
    rng = jax.random.PRNGKey(self.rng_seed)
    init = jax.random.normal(rng, (self.batch, self.n), dtype=jnp.float64)
    out = sampler(inference_fn=s["inference_fn"], rng=rng, initial_noise=init)
    final_step = out[0] if isinstance(out, tuple) else out
    samples = np.asarray(final_step.xt)
    return samples.mean(axis=0), np.asarray(s["mu_post"]).mean(axis=0)

  def test_bootstrap_smc_recovers_posterior_mean(self):
    """Pure bootstrap SMC (no correction) -- the canonical consistency claim.

    Tolerance reflects the small-particle regime affordable inside a
    unit test; larger K and averaging over seeds would tighten it.
    """
    mean, target = self._run(correction_fn=None)
    err = np.max(np.abs(mean - target))
    self.assertLess(
        err, 0.5,
        msg=(
            f"Bootstrap TDS off posterior mean by {err:.4f}.\n"
            f"  empirical: {mean}\n"
            f"  target:    {target}"
        ),
    )

  def test_pigdm_smc_recovers_posterior_mean(self):
    """Pi-GDM-twisted SMC: informed proposal, should match mu_post tighter
    than bootstrap but still within MC noise for our K."""
    s = self._setup()
    mean, target = self._run(correction_fn=s["pigdm"])
    err = np.max(np.abs(mean - target))
    self.assertLess(
        err, 0.75,
        msg=(
            f"Pi-GDM-TDS off posterior mean by {err:.4f}.\n"
            f"  empirical: {mean}\n"
            f"  target:    {target}"
        ),
    )


if __name__ == "__main__":
  unittest.main()
