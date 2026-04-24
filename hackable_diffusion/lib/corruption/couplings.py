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

"""Concrete :class:`Source` and :class:`Coupling` implementations.

- :class:`StandardNormalSource`: ``x_1 ~ N(0, I)``.  The diffusion
  source; ``IndependentCoupling(StandardNormalSource())`` recovers the
  legacy ``GaussianProcess`` sampling behaviour byte-for-byte.
- :class:`UniformManifoldSource`: ``x_1 ~ Uniform(manifold)``.  The
  Riemannian source.
- :class:`DataloaderSource`: pulls pre-batched tensors from a queue.
  For data-to-data flow matching.

- :class:`IndependentCoupling`: ``x_1`` drawn from a source independently
  of ``x_0``.  Per-sample, vmap-friendly.
- :class:`DeterministicCoupling`: ``x_1 = map_fn(x_0)``.  Per-sample,
  vmap-friendly.  No ``marginal`` (depends on ``p_0``).
- :class:`MiniBatchOTCoupling`: deferred to M4.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, ClassVar

import jax

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.corruption import base

PRNGKey = hd_typing.PRNGKey
DataTree = hd_typing.DataTree

Source = base.Source
Coupling = base.Coupling


################################################################################
# MARK: Sources
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class StandardNormalSource(Source):
  """``x_1 ~ N(0, I)``.

  Byte-equivalent to legacy ``GaussianProcess.sample_from_invariant``
  when wrapped in :class:`IndependentCoupling`.
  """

  def sample(self, key: PRNGKey, data_spec: DataTree) -> DataTree:
    return jax.random.normal(key, shape=data_spec.shape)


@dataclasses.dataclass(kw_only=True, frozen=True)
class UniformManifoldSource(Source):
  """``x_1 ~ Uniform(manifold)``.

  Wraps :meth:`manifold.random_uniform`.  Byte-equivalent to legacy
  ``RiemannianProcess.sample_from_invariant``.
  """

  manifold: Any

  def sample(self, key: PRNGKey, data_spec: DataTree) -> DataTree:
    return self.manifold.random_uniform(key, data_spec.shape)


@dataclasses.dataclass(kw_only=True, frozen=True)
class DataloaderSource(Source):
  """Pulls a pre-batched tensor from a caller-provided ``pull`` callable.

  For data-to-data flow matching where ``x_1`` comes from a dataset
  rather than a closed-form distribution.  The ``key`` argument is
  ignored (the dataloader provides its own ordering); this is an
  impurity the framework tolerates because the alternative -- threading
  a pytree of dataset iterators through ``corrupt`` -- is much worse.
  """

  pull: Callable[[DataTree], DataTree]

  def sample(self, key: PRNGKey, data_spec: DataTree) -> DataTree:
    del key
    return self.pull(data_spec)


################################################################################
# MARK: Couplings
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class IndependentCoupling(Coupling):
  """``x_1 ~ source``, independent of ``x_0``.

  Per-sample in effect (``source.sample`` produces a fresh batch
  regardless of ``x_0``); safe to vmap.  Recovers the legacy diffusion
  coupling when ``source=StandardNormalSource()``.
  """

  source: Source
  is_batch_level: ClassVar[bool] = False

  @property
  def marginal(self) -> Source:
    return self.source

  def sample(self, key: PRNGKey, x0: DataTree) -> DataTree:
    return self.source.sample(key, x0)


@dataclasses.dataclass(kw_only=True, frozen=True)
class DeterministicCoupling(Coupling):
  """``x_1 = map_fn(x_0)``.

  Per-sample, vmap-friendly.  No well-defined ``x_1`` marginal (depends
  on ``p_0``), so ``marginal = None`` and ``sample_from_invariant``
  raises.
  """

  map_fn: Callable[[DataTree], DataTree]
  is_batch_level: ClassVar[bool] = False
  marginal: None = None

  def sample(self, key: PRNGKey, x0: DataTree) -> DataTree:
    del key
    return self.map_fn(x0)
