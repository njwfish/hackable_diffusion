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

"""Helpers for distributional diffusion (arXiv:2502.02483).

Implements the ``x_theta(t, x_t, xi)`` ensemble forward:
  1. draw M i.i.d. Gaussian noise tensors ``xi`` with the same shape as ``x_t``,
  2. inject each into ``x_t`` by channel-concat along the last axis
     (paper's default for image U-Nets and the 2D MLP),
  3. call the network with ``jax.vmap`` over the M axis,
  4. slice the first ``c`` channels of each prediction to drop the xi-half
     of the doubled output.

The returned predictions dict has an extra axis of size ``M`` inserted at
position 1 (``[B, M, *data]``), ready for ``EnergyScoreLoss``.
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
OutputTrim = Callable[[TargetInfo], TargetInfo]

################################################################################
# MARK: Default xi injection
################################################################################


def channel_concat_xi(xt: DataArray, xi: DataArray) -> DataArray:
  """Concatenates xi to xt along the last axis (paper's default)."""
  return jnp.concatenate([xt, xi], axis=-1)


def make_channel_slice(keep_channels: int) -> OutputTrim:
  """Returns a trim fn that keeps the first ``keep_channels`` of the last dim.

  The network is expected to output ``2 * keep_channels`` along the final
  axis; the xi-half is dropped. Matches the paper's U-Net and 2D MLP setup.
  """
  def _trim(preds: TargetInfo) -> TargetInfo:
    return jax.tree.map(lambda y: y[..., :keep_channels], preds)
  return _trim


################################################################################
# MARK: Ensemble forward
################################################################################


def ensemble_apply(
    apply_fn: Callable[..., TargetInfo],
    variables: PyTree,
    *,
    time: TimeArray,
    xt: DataArray,
    conditioning: Conditioning | None,
    xi_rng: PRNGKey,
    population_size: int,
    xi_injector: XiInjector = channel_concat_xi,
    output_trim: OutputTrim | None = None,
    apply_rngs: Mapping[str, PRNGKey] | None = None,
    **apply_kwargs: Any,
) -> TargetInfo:
  """Calls a diffusion network ``population_size`` times with fresh xi draws.

  Args:
    apply_fn: The Flax apply fn of the diffusion network (``network.apply``).
    variables: Variable tree passed as the first positional arg to ``apply_fn``.
    time: Time array ``[B, ...]``.
    xt: Noisy data ``[B, *data]``.
    conditioning: Optional conditioning pytree.
    xi_rng: PRNGKey used *only* to draw the xi noise tensor. Dropout / other
      RNGs the network may need are passed through ``apply_rngs``.
    population_size: M, the number of ensemble members per example.
    xi_injector: How to combine ``xt`` and a single xi draw into the tensor the
      network consumes. Default is ``channel_concat_xi`` (concat along last
      axis), matching the paper.
    output_trim: Optional callable that trims the network output to the
      original data shape (drops the xi-half of a doubled-channel output). Use
      ``make_channel_slice(keep_channels=c)`` to slice last dim.
    apply_rngs: Additional RNGs forwarded to ``apply_fn`` (e.g. ``{"dropout":
      key}``). Shared across the M ensemble members — downstream dropout
      patterns will therefore be identical per member, which matches the
      paper's setup. Split this yourself if you want per-member dropout.
    **apply_kwargs: Additional kwargs forwarded to ``apply_fn`` (e.g.
      ``is_training=True``).

  Returns:
    A predictions pytree with an extra axis of size ``M`` inserted at position
    1 of every leaf — e.g. ``{"x0": [B, M, *data]}``.
  """
  if population_size < 1:
    raise ValueError(f"population_size must be >= 1, got {population_size}.")

  xi = jax.random.normal(
      xi_rng, (population_size,) + xt.shape, dtype=xt.dtype
  )                                                                 # [M, B, ...]

  def _one(xi_member: DataArray) -> TargetInfo:
    xt_ext = xi_injector(xt, xi_member)
    preds = apply_fn(
        variables,
        time=time,
        xt=xt_ext,
        conditioning=conditioning,
        rngs=apply_rngs,
        **apply_kwargs,
    )
    if output_trim is not None:
      preds = output_trim(preds)
    return preds

  # out_axes=1 places the M axis at position 1 of every leaf, i.e. right
  # after the batch dim — the shape the energy-score loss expects.
  return jax.vmap(_one, in_axes=0, out_axes=1)(xi)
