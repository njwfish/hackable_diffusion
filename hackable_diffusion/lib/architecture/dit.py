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

"""DiT backbone."""

from flax import linen as nn
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import normalization
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
import jax.numpy as jnp

################################################################################
# MARK: Type Aliases
################################################################################

Float = hd_typing.Float
DType = hd_typing.DType

DataArray = hd_typing.DataArray

ConditionalBackbone = arch_typing.ConditionalBackbone
ConditioningMechanism = arch_typing.ConditioningMechanism
NormalizationType = arch_typing.NormalizationType

################################################################################
# MARK: DiT
################################################################################


class DiT(ConditionalBackbone):
  """DiT model.

  A Diffusion Transformer backbone based on:
  https://arxiv.org/abs/2212.09748

  Uses adaptive layer norm zero (adaLN-Zero) conditioning mechanism.
  The architecture consists of repeated DiT blocks with optional
  encoder/decoder and absolute positional encoding.

  Attributes:
    num_blocks: Number of DiT blocks.
    block: A DiT block module (e.g., DiTBlockAdaLNZero).
    encoder: Optional encoder module (e.g., Patchify for image inputs).
    decoder: Optional decoder module (e.g., DePatchify for image outputs).
    absolute_posenc: Optional absolute positional encoding module.
  """

  num_blocks: int

  block: nn.Module

  encoder: nn.Module | None = None
  decoder: nn.Module | None = None

  absolute_posenc: nn.Module | None = None

  cast_to_float32: bool = False
  dtype: DType = jnp.float32

  def setup(self):
    self.conditional_norm = normalization.NormalizationLayerFactory(
        normalization_method=NormalizationType.LAYER_NORM,
        dtype=self.dtype,
        use_bias=False,
        use_scale=False,
    ).conditional_norm_factory()

  @typechecked
  @nn.compact
  def __call__(
      self,
      x: DataArray,
      conditioning_embeddings: dict[ConditioningMechanism, Float["batch ..."]],
      is_training: bool,
  ) -> DataArray:
    adaptive_norm_emb = conditioning_embeddings.get(
        ConditioningMechanism.ADAPTIVE_NORM
    )
    if adaptive_norm_emb is None:
      raise ValueError("adaptive_norm_emb must be provided.")

    # Tokenize the input and add positional encoding.
    tokens = self.encoder(x) if self.encoder else x
    if self.absolute_posenc:
      tokens = tokens + self.absolute_posenc(tokens)

    # Apply DiT blocks.
    cond = adaptive_norm_emb
    for i in range(1, self.num_blocks + 1):
      tokens = self.block.copy(name=f"Block_{i}")(
          tokens, cond, is_training=is_training
      )

    tokens = self.conditional_norm(tokens, c=nn.silu(cond))

    # Decode the tokens to the output.
    if self.decoder:
      tokens = self.decoder(tokens, cond)

    if self.cast_to_float32:
      tokens = utils.optional_bf16_to_fp32(tokens)
    return tokens
