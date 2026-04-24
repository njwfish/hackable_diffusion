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

"""Concrete :class:`Coupling` implementations.

Every concrete coupling satisfies the single :class:`Coupling`
Protocol: ``sample(key, x0) -> x_1``, plus ``marginal`` (an
``x_0``-independent coupling usable at inference time) and
``is_batch_level`` (whether ``sample`` must see the whole batch).

- :class:`StandardNormalSource`: ``x_1 ~ N(0, I)``; ignores ``x_0``
  values (uses only its shape).  ``marginal = self``.  Recovers legacy
  diffusion behaviour when paired with :class:`LinearInterpolant`.
- :class:`UniformManifoldSource`: ``x_1 ~ Uniform(manifold)``.
  ``marginal = self``.  Riemannian case.
- :class:`DataloaderSource`: ``x_1 = pull(x_0)`` where ``pull`` is a
  caller-provided dataset callable (``key`` ignored -- dataloader
  provides its own ordering).  ``marginal = self``.
- :class:`DeterministicCoupling`: ``x_1 = map_fn(x_0)``.
  ``marginal = None`` (depends on ``p_0``); blur-deblur / data-to-data.
- :class:`MiniBatchOTCoupling`: entropic-OT matching via ott-jax
  Sinkhorn between a batch of ``x_0`` and an unmatched ``x_1 ~ source``
  batch.  ``marginal = source``.  ``is_batch_level = True``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, ClassVar

import jax
import jax.numpy as jnp

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.corruption import base

PRNGKey = hd_typing.PRNGKey
DataTree = hd_typing.DataTree

Coupling = base.Coupling


@dataclasses.dataclass(kw_only=True, frozen=True)
class StandardNormalSource(Coupling):
  """``x_1 ~ N(0, I)``; ignores ``x_0`` values, uses only its shape."""

  is_batch_level: ClassVar[bool] = False

  @property
  def marginal(self) -> 'StandardNormalSource':
    return self

  def sample(self, key: PRNGKey, x0: DataTree) -> DataTree:
    return jax.random.normal(key, shape=x0.shape)


@dataclasses.dataclass(kw_only=True, frozen=True)
class UniformManifoldSource(Coupling):
  """``x_1 ~ Uniform(manifold)``; wraps :meth:`manifold.random_uniform`."""

  manifold: Any
  is_batch_level: ClassVar[bool] = False

  @property
  def marginal(self) -> 'UniformManifoldSource':
    return self

  def sample(self, key: PRNGKey, x0: DataTree) -> DataTree:
    return self.manifold.random_uniform(key, x0.shape)


@dataclasses.dataclass(kw_only=True, frozen=True)
class DataloaderSource(Coupling):
  """``x_1 = pull(x_0)``.  Caller-provided dataset callable; ``key`` ignored.

  The ``key`` argument is ignored because the dataloader provides its
  own ordering.  This is an impurity the framework tolerates because
  the alternative -- threading a pytree of dataset iterators through
  ``corrupt`` -- is much worse.
  """

  pull: Callable[[DataTree], DataTree]
  is_batch_level: ClassVar[bool] = False

  @property
  def marginal(self) -> 'DataloaderSource':
    return self

  def sample(self, key: PRNGKey, x0: DataTree) -> DataTree:
    del key
    return self.pull(x0)


@dataclasses.dataclass(kw_only=True, frozen=True)
class DeterministicCoupling(Coupling):
  """``x_1 = map_fn(x_0)``; no well-defined marginal."""

  map_fn: Callable[[DataTree], DataTree]
  is_batch_level: ClassVar[bool] = False
  marginal: 'Coupling | None' = None

  def sample(self, key: PRNGKey, x0: DataTree) -> DataTree:
    del key
    return self.map_fn(x0)


@dataclasses.dataclass(kw_only=True, frozen=True)
class MiniBatchOTCoupling(Coupling):
  """Mini-batch entropic-OT coupling between ``x_0`` and ``x_1 ~ source``.

  Given a batch of ``x_0``, samples an unmatched batch of ``x_1`` from
  ``source`` and computes an optimal-transport plan between them via
  entropy-regularised Sinkhorn (ott-jax).  Each ``x_0[i]`` is paired
  with ``x_1[j]`` by sampling ``j ~ Categorical(plan[i, :] / sum(plan[i, :]))``.

  Batch-level: ``is_batch_level = True``.  Training loops must call
  ``corrupt`` on the whole batch at once, not vmap over it.

  Gradients do not flow through the coupling output
  (``jax.lax.stop_gradient``).  This matches the OT-CFM literature
  (Tong et al. 2024) where the plan is computed purely as a matching
  signal, not a differentiable term.

  Configuration surface is intentionally narrow: ``epsilon`` (Sinkhorn
  regularisation) and ``num_iters`` only.  Users who need other
  ott-jax knobs (unbalanced, low-rank, custom cost) can subclass and
  override ``_transport_plan``.
  """

  source: Coupling
  epsilon: float = 0.01
  num_iters: int = 100
  is_batch_level: ClassVar[bool] = True

  @property
  def marginal(self) -> Coupling:
    return self.source

  def _transport_plan(
      self, x0_flat: jax.Array, x1_flat: jax.Array,
  ) -> jax.Array:
    """Return the ``(B, B)`` Sinkhorn transport plan.  Override for
    custom solvers / costs."""
    from ott.geometry import pointcloud
    from ott.problems.linear import linear_problem
    from ott.solvers.linear import sinkhorn

    geom = pointcloud.PointCloud(
        x0_flat, x1_flat, epsilon=self.epsilon,
    )
    problem = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn(max_iterations=self.num_iters)
    output = solver(problem)
    return output.matrix

  def sample(self, key: PRNGKey, x0: DataTree) -> DataTree:
    key_source, key_match = jax.random.split(key)
    x1_unmatched = self.source.sample(key_source, x0)

    x0_flat = x0.reshape(x0.shape[0], -1)
    x1_flat = x1_unmatched.reshape(x1_unmatched.shape[0], -1)

    plan = self._transport_plan(x0_flat, x1_flat)
    row_sums = jnp.sum(plan, axis=-1, keepdims=True)
    row_log_probs = jnp.log(
        jnp.clip(plan / jnp.maximum(row_sums, 1e-30), 1e-30, None),
    )
    indices = jax.random.categorical(key_match, row_log_probs, axis=-1)

    x1_matched = x1_unmatched[indices]
    return jax.lax.stop_gradient(x1_matched)


def assert_vmappable(process: base.InterpolantProcess) -> None:
  """Raise ``ValueError`` if ``process.coupling.is_batch_level``.

  Training loops that wrap :meth:`InterpolantProcess.corrupt` in a
  per-sample ``jax.vmap`` must call the whole batch through ``corrupt``
  directly when the coupling is batch-level (e.g.
  :class:`MiniBatchOTCoupling`).
  """
  if process.coupling.is_batch_level:
    raise ValueError(
        f'{type(process.coupling).__name__} is batch-level and cannot be '
        'vmapped per-sample.  Call ``corruption_process.corrupt(key, '
        'x0_batch, time)`` on the whole batch directly.'
    )
