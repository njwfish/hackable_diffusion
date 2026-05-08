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

"""Training-side helpers for posterior-sampler diffusion (arXiv:2502.02483).

The training-time analogue of the sampling-time
:class:`hackable_diffusion.lib.inference.PosteriorSamplerInferenceFn`
and :func:`hackable_diffusion.lib.guidance.make_posterior_cloud_fn`:
all three call the same shape-preserving network ``M`` (or ``R``)
times with fresh ``xi`` draws to produce a posterior cloud
``[B, M, *data]``.

The network is **shape-preserving** under the ``[x_t, xi]``
doubled-last-axis input. That invariant is provided by posterior-cloud
backbone variants -- :class:`hackable_diffusion.lib.architecture.NoiseTrimBackbone`
for any backbone whose last axis is a clean feature dim (MLPs,
per-token transformers), or a dedicated subclass for backbones that
reshape the last axis into an image (e.g.
``mdt.model.unet_patch_distributional.DistributionalUNetPatch`` -- the
"distributional" naming there is the literature name of the training
method, kept for citation continuity).

With that invariant, :func:`posterior_cloud_apply` is just:
  1. draw ``M`` i.i.d. Gaussian ``xi``, each with the same shape as
     ``x_t``,
  2. inject each via a user-supplied ``xi_injector`` (default:
     channel-concat along the last axis),
  3. vmap the network over the ``M`` axis.

The returned predictions pytree has an extra axis of size ``M`` at
position 1 of every leaf (``[B, M, *data]``) -- the same shape as
:func:`hackable_diffusion.lib.guidance.make_posterior_cloud_fn`'s
sampling-time output, ready for
:class:`hackable_diffusion.lib.loss.EnergyScoreLoss` or any
posterior-aware downstream consumer.
"""

from typing import Any, Callable, Mapping

from hackable_diffusion.lib import hd_typing
import jax
import jax.numpy as jnp

################################################################################
# MARK: Type Aliases
################################################################################

PRNGKey = hd_typing.PRNGKey
PyTree = hd_typing.PyTree

Conditioning = hd_typing.Conditioning
DataArray = hd_typing.DataArray
TargetInfo = hd_typing.TargetInfo
TimeArray = hd_typing.TimeArray

XiInjector = Callable[[DataArray, DataArray], DataArray]

################################################################################
# MARK: Default xi injection
################################################################################


def channel_concat_xi(xt: DataArray, xi: DataArray) -> DataArray:
  """Concatenates xi to xt along the last axis (paper's default)."""
  return jnp.concatenate([xt, xi], axis=-1)


################################################################################
# MARK: Posterior cloud forward
################################################################################


def posterior_cloud_apply(
    apply_fn: Callable[..., TargetInfo],
    variables: PyTree,
    *,
    time: TimeArray,
    xt: DataArray,
    conditioning: Conditioning | None,
    xi_rng: PRNGKey,
    population_size: int,
    xi_injector: XiInjector = channel_concat_xi,
    apply_rngs: Mapping[str, PRNGKey] | None = None,
    **apply_kwargs: Any,
) -> TargetInfo:
  """Calls a shape-preserving diffusion network M times with fresh xi draws.

  Training-time counterpart of
  :func:`hackable_diffusion.lib.guidance.make_posterior_cloud_fn`: both
  produce an ``[B, M, *data]`` posterior cloud by ``vmap``-ing the
  network over ``M`` independent ``xi`` draws.  Use this in the energy
  score / scoring-rule training loop; use ``make_posterior_cloud_fn``
  inside a sampler when a cloud-aware twist or projection needs ``R``
  posterior samples at one ``x_t``.

  Pair this with a network whose output shape matches the original data
  shape -- i.e. one built around a posterior-cloud backbone variant
  (see the module docstring). Do not hand a raw backbone whose output
  has a doubled last axis; wrap it first.

  Args:
    apply_fn: The Flax apply fn of the diffusion network (``network.apply``).
    variables: Variable tree passed as the first positional arg to ``apply_fn``.
    time: Time array ``[B, ...]``.
    xt: Noisy data ``[B, *data]``. The network sees ``xi_injector(xt, xi)``;
      with the default injector that is the doubled-last-axis tensor that
      posterior-cloud backbones expect.
    conditioning: Optional conditioning pytree.
    xi_rng: PRNGKey used *only* to draw the xi noise tensor. Dropout / other
      RNGs the network may need are passed through ``apply_rngs``.
    population_size: M, the number of cloud members per example.
    xi_injector: How to combine ``xt`` and a single xi draw into the tensor the
      network consumes. Default is ``channel_concat_xi`` (concat along last
      axis), matching the paper.
    apply_rngs: Additional RNGs forwarded to ``apply_fn`` (e.g. ``{"dropout":
      key}``). Shared across the M cloud members -- downstream dropout
      patterns will therefore be identical per member, which matches the
      paper's setup. Split this yourself if you want per-member dropout.
    **apply_kwargs: Additional kwargs forwarded to ``apply_fn`` (e.g.
      ``is_training=True``).

  Returns:
    A predictions pytree with an extra axis of size ``M`` inserted at position
    1 of every leaf -- e.g. ``{"x0": [B, M, *data]}``.  Same shape as
    :func:`hackable_diffusion.lib.guidance.make_posterior_cloud_fn`'s
    sampling-time output.
  """
  if population_size < 1:
    raise ValueError(f"population_size must be >= 1, got {population_size}.")

  xi = jax.random.normal(
      xi_rng, (population_size,) + xt.shape, dtype=xt.dtype
  )                                                                 # [M, B, *data]

  def _one(xi_member: DataArray) -> TargetInfo:
    xt_ext = xi_injector(xt, xi_member)
    return apply_fn(
        variables,
        time=time,
        xt=xt_ext,
        conditioning=conditioning,
        rngs=apply_rngs,
        **apply_kwargs,
    )

  # out_axes=1 places the M axis at position 1 of every leaf, i.e. right
  # after the batch dim -- the shape the energy-score loss and any
  # cloud-aware downstream consumer expect.
  return jax.vmap(_one, in_axes=0, out_axes=1)(xi)
