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

"""Canonical linear forward operators satisfying :class:`ForwardFn`.

These cover the inverse-problem benchmarks in the Pi-GDM / DPS /
MCGDiff literature.  Every class is a tiny frozen dataclass with a
single ``forward(x)`` method -- the adjoint is picked up automatically
by :func:`linear_adjoint` via ``jax.vjp``.

- :class:`LinearForwardFn`: dense matrix or user-supplied apply_fn.
- :class:`SubsampleForwardFn`: slice / stride along a spatial axis.
- :class:`InpaintingForwardFn`: select observed pixels under a mask.
- :class:`ConvForwardFn`: linear convolution via ``lax.conv_general_dilated``.
- :class:`ComposeForwardFn`: compose two linear forwards ``f2 o f1``.

All operate on a leading batch axis ``(B, *spatial)``.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp

from hackable_diffusion.lib.guidance.protocols import ForwardFn


@dataclasses.dataclass(kw_only=True, frozen=True)
class LinearForwardFn(ForwardFn):
  """Dense matrix ``y = x @ W^T`` (or user-supplied linear ``apply_fn``).

  ``matrix`` or ``apply_fn`` -- exactly one must be set.  Use
  ``apply_fn`` for structured operators (FFT-diagonal, sparse, random
  projection) that shouldn't be materialised.
  """

  matrix: jax.Array | None = None
  apply_fn: Callable[[jax.Array], jax.Array] | None = None

  def __post_init__(self):
    if (self.matrix is None) == (self.apply_fn is None):
      raise ValueError(
          "Exactly one of ``matrix`` or ``apply_fn`` must be set."
      )

  def forward(self, x: jax.Array) -> jax.Array:
    if self.apply_fn is not None:
      return self.apply_fn(x)
    x_flat = x.reshape(x.shape[0], -1)
    return x_flat @ self.matrix.T


@dataclasses.dataclass(kw_only=True, frozen=True)
class SubsampleForwardFn(ForwardFn):
  """Stride / slice along the last axis: ``y = x[..., indices]``.

  ``indices`` is a 1-D integer array of observed coordinates (e.g. every
  second pixel, a random subset, or an interior crop).  The adjoint
  implicit via VJP is "scatter-into-zeros" -- exactly the Pi-GDM
  measurement adjoint for subsampling.
  """

  indices: jax.Array

  def forward(self, x: jax.Array) -> jax.Array:
    return jnp.take(x, self.indices, axis=-1)


@dataclasses.dataclass(kw_only=True, frozen=True)
class InpaintingForwardFn(ForwardFn):
  """Mask observed pixels: ``y = mask * x`` (element-wise).

  ``mask`` broadcasts against the non-batch axes of ``x``; entries equal
  to 1.0 are observed, 0.0 are free.  The adjoint is the same mask
  applied to the cotangent -- under Pi-GDM this cleanly confines the
  Kalman update to the free pixels.
  """

  mask: jax.Array

  def forward(self, x: jax.Array) -> jax.Array:
    return self.mask * x


@dataclasses.dataclass(kw_only=True, frozen=True)
class ConvForwardFn(ForwardFn):
  """Linear convolution via ``lax.conv_general_dilated``.

  ``kernel`` shape matches JAX's NCHW / NHWC convention as selected by
  ``dimension_numbers``.  Defaults to 1-D (NCL) convolution with
  ``stride=1`` and SAME padding, useful for Gaussian-blur deblurring
  benchmarks.  Multi-dimensional setups (2-D image blurs) work with
  the same dataclass -- pass the appropriate ``dimension_numbers``
  triple.
  """

  kernel: jax.Array
  stride: tuple[int, ...] = (1,)
  padding: str = "SAME"
  dimension_numbers: tuple[str, str, str] = ("NCL", "OIL", "NCL")

  def forward(self, x: jax.Array) -> jax.Array:
    return jax.lax.conv_general_dilated(
        x, self.kernel, self.stride, self.padding,
        dimension_numbers=self.dimension_numbers,
    )


@dataclasses.dataclass(kw_only=True, frozen=True)
class ComposeForwardFn(ForwardFn):
  """Function composition ``(f2 o f1)(x) = f2(f1(x))``.

  Useful for pipelines like "blur then subsample" (conv + stride) that
  appear in image super-resolution benchmarks.  Linear when both
  components are linear; the VJP-based adjoint is
  ``f1^T o f2^T`` automatically.
  """

  first: ForwardFn
  second: ForwardFn

  def forward(self, x: jax.Array) -> jax.Array:
    return self.second.forward(self.first.forward(x))
