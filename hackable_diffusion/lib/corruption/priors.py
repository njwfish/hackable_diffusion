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

"""Concrete :class:`Prior` implementations.

A :class:`Prior` produces an ``x_1`` marginal sample.  Used inside
:meth:`InterpolantProcess.corrupt` when the caller doesn't pass
``x_1``, and inside :meth:`InterpolantProcess.sample_from_invariant`
at inference time.

- :class:`GaussianPrior`: ``x_1 ~ N(0, I)``; recovers legacy diffusion
  when paired with :class:`LinearInterpolant`.
- :class:`UniformManifoldPrior`: ``x_1 ~ Uniform(manifold)``; Riemannian
  case.
- :class:`DeterministicPrior`: ``x_1 = map_fn(x_0)``; blur/mask and
  other data-to-data priors where ``x_1`` is a fixed function of
  ``x_0``.  Strictly the only x0-functional prior shipped here -- a
  "joint sampler" disguised as a prior; paired with
  :class:`IndependentCoupling` it gives byte-equivalent semantics to
  the pre-refactor ``DeterministicCoupling``.

For a paired dataset, no prior is needed: pass ``x1=`` to
``corrupt(...)`` directly and the prior slot stays ``None``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import jax

from hackable_diffusion.lib import hd_typing

PRNGKey = hd_typing.PRNGKey
DataTree = hd_typing.DataTree


@dataclasses.dataclass(kw_only=True, frozen=True)
class GaussianPrior:
  """``x_1 ~ N(0, I)``; uses only the shape of ``x_0``."""

  def sample(self, key: PRNGKey, x0: DataTree) -> DataTree:
    return jax.random.normal(key, shape=x0.shape)


@dataclasses.dataclass(kw_only=True, frozen=True)
class UniformManifoldPrior:
  """``x_1 ~ Uniform(manifold)``; wraps :meth:`manifold.random_uniform`."""

  manifold: Any

  def sample(self, key: PRNGKey, x0: DataTree) -> DataTree:
    return self.manifold.random_uniform(key, x0.shape)


@dataclasses.dataclass(kw_only=True, frozen=True)
class DeterministicPrior:
  """``x_1 = map_fn(x_0)``; deterministic function of ``x_0``.

  The one prior that *does* depend on ``x_0`` values (not just its
  spec).  Useful for blur-deblur and other functional priors.
  ``sample_from_invariant`` will refuse to use this prior alone (no
  well-defined unconditional ``x_1`` marginal) -- the wrapping
  :class:`InterpolantProcess` raises if a caller tries.
  """

  map_fn: Callable[[DataTree], DataTree]

  def sample(self, key: PRNGKey, x0: DataTree) -> DataTree:
    del key
    return self.map_fn(x0)
