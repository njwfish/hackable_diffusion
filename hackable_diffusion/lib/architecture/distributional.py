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

"""Noise-injection backbone wrapper for distributional diffusion.

Implements the architecture recipe of De Bortoli et al., "Distributional
Diffusion Models with Scoring Rules" (arXiv:2502.02483): the network
receives ``[x_t, xi]`` concatenated along the last axis and emits an output
of the same doubled shape; we slice off the xi-half to recover the
data-shape prediction.

This is the canonical way to make any ``ConditionalBackbone`` whose input
and output share the same last-axis layout (MLPs, per-token transformers,
etc.) shape-preserving under the distributional recipe. Use it whenever you
configure a distributional training run against a non-U-Net backbone.

For U-Net-style backbones that reshape the last axis into an image before
applying convolutions, a last-axis token-level slice would interleave pixel
and channel indices. Those backbones need to split/concat/slice at the
**image** channel axis instead; provide a distributional subclass (see e.g.
``mdt.model.unet_patch_distributional.DistributionalUNetPatch``) rather
than wrapping with this module.
"""

from __future__ import annotations

import flax.linen as nn

from hackable_diffusion.lib.architecture import arch_typing

ConditionalBackbone = arch_typing.ConditionalBackbone


class NoiseTrimBackbone(nn.Module, ConditionalBackbone):
  """Shape-preserving wrapper: run ``base``, slice last axis to data size.

  The caller (typically
  ``hackable_diffusion.lib.distributional.ensemble_apply`` at train time or
  ``hackable_diffusion.lib.inference.DistributionalInferenceFn`` at sampling
  time) concatenates ``xi`` onto ``x_t`` along the last axis. The wrapped
  backbone sees this doubled input unchanged; we only trim its output.

  Attributes:
    base: Any ``ConditionalBackbone`` whose input and output share the same
      last-axis layout (e.g. ``hackable_diffusion.lib.architecture.mlp.
      ConditionalMLP``). Wrapping a backbone that internally reshapes
      channel axes (e.g. a patch-token U-Net) will produce wrong slicing
      semantics; supply a dedicated distributional subclass instead.
    keep_channels: Number of last-axis elements to keep from the output.
      If ``None`` (the default), inferred as half of the incoming last-axis
      size — the right setting whenever the caller doubled the input by
      channel-concatenating ``xi``, which is the standard path.
  """

  base: nn.Module
  keep_channels: int | None = None

  @nn.compact
  def __call__(self, x, conditioning_embeddings, is_training):
    y = self.base(
        x=x,
        conditioning_embeddings=conditioning_embeddings,
        is_training=is_training,
    )
    keep = self.keep_channels
    if keep is None:
      if x.shape[-1] % 2 != 0:
        raise ValueError(
            "NoiseTrimBackbone with keep_channels=None expects an even "
            f"last-axis size, got {x.shape[-1]}."
        )
      keep = x.shape[-1] // 2
    return y[..., :keep]
