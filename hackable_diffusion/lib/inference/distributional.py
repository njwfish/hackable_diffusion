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

"""Inference fn for distributional diffusion (arXiv:2502.02483).

At each reverse step the network takes ``(t, x_t, xi)`` with ``xi ~ N(0, I)``,
and outputs an approximate *sample* from ``p_{0|t}(x_0 | x_t)`` rather than
its mean. The sampling-loop scan body passes us the step's rng; we use it to
draw ``xi``, concat into ``x_t``, call the network once, and slice the
xi-half off the output to match the expected prediction shape.
"""

import dataclasses
from typing import Callable

from hackable_diffusion.lib import distributional
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.inference import base
import flax.linen as nn
import jax
import jax.numpy as jnp
import kauldron.ktyping as kt

################################################################################
# MARK: Type Aliases
################################################################################

PyTree = hd_typing.PyTree

Conditioning = hd_typing.Conditioning
DataArray = hd_typing.DataArray
DataTree = hd_typing.DataTree
PRNGKey = hd_typing.PRNGKey
TargetInfoTree = hd_typing.TargetInfoTree
TimeTree = hd_typing.TimeTree

InferenceFn = base.InferenceFn

XiInjector = distributional.XiInjector
OutputTrim = distributional.OutputTrim

################################################################################
# MARK: DistributionalInferenceFn
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class DistributionalInferenceFn(InferenceFn):
  """Stochastic inference fn: calls the network with a per-step xi draw.

  Use case: sampling from a network trained with ``EnergyScoreLoss`` that
  expects a ``(x_t, xi)`` concatenated input. At every reverse step, the
  sampling loop passes the step's rng via ``rng=``; we deterministically
  derive an xi-specific key, draw ``xi``, forward the network, and strip the
  xi-half of the output.

  This is the sampling-time counterpart of ``lib/distributional.ensemble_apply``
  used at training time. Sampling uses a single xi draw per step (not a
  population) — the energy-score training is what bought us the ability to
  take big reverse steps with just one sample.

  Attributes:
    network: The trained Linen diffusion network.
    params: The trained parameters tree.
    keep_channels: Number of channels to keep from the last axis of every
      prediction leaf (to drop the xi-half of a doubled-channel output). Must
      match the per-sample data channel count the network was trained with.
    xi_injector: How to combine ``x_t`` and ``xi`` into the network's input.
      Default channel-concat along the last axis.
    rng_salt: Integer mixed into the step rng before drawing xi, so that any
      other consumer of ``step_info.rng`` in the same step (e.g. the SDE Z
      noise in the update) gets an independent stream.
  """

  network: nn.Module
  params: PyTree
  keep_channels: int
  xi_injector: XiInjector = dataclasses.field(
      default=distributional.channel_concat_xi
  )
  rng_salt: int = 0x1D01  # "distributional"

  @kt.typechecked
  def __call__(
      self,
      time: TimeTree,
      xt: DataTree,
      conditioning: Conditioning | None,
      rng: PRNGKey | None = None,
  ) -> TargetInfoTree:
    if rng is None:
      raise ValueError(
          "DistributionalInferenceFn requires an rng; the sampling loop "
          "should be passing one from step_info. If you're calling this fn "
          "directly, pass rng= explicitly."
      )
    xi_rng = jax.random.fold_in(rng, self.rng_salt)
    xi = jax.random.normal(xi_rng, xt.shape, dtype=xt.dtype)
    xt_ext = self.xi_injector(xt, xi)
    preds = self.network.apply(
        {"params": self.params},
        time=time,
        xt=xt_ext,
        conditioning=conditioning,
        is_training=False,
    )
    return jax.tree.map(lambda y: y[..., : self.keep_channels], preds)
