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
import jax.numpy as jnp

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

  source: Source
  epsilon: float = 0.01
  num_iters: int = 100
  is_batch_level: ClassVar[bool] = True

  @property
  def marginal(self) -> Source:
    return self.source

  def _transport_plan(
      self, x0_flat: jax.Array, x1_flat: jax.Array,
  ) -> jax.Array:
    """Return the ``(B, B)`` Sinkhorn transport plan.  Override for
    custom solvers / costs."""
    # Deferred imports: ott-jax is only needed inside this method.
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
    # Row-normalise to get per-row conditional assignments p(j | i).
    row_sums = jnp.sum(plan, axis=-1, keepdims=True)
    row_log_probs = jnp.log(
        jnp.clip(plan / jnp.maximum(row_sums, 1e-30), 1e-30, None),
    )
    indices = jax.random.categorical(key_match, row_log_probs, axis=-1)

    x1_matched = x1_unmatched[indices]
    return jax.lax.stop_gradient(x1_matched)


def assert_vmappable(process) -> None:
  """Assert that ``process``'s coupling is vmap-safe (per-sample).

  Raises ``ValueError`` with a clear message when the coupling is
  batch-level (e.g. :class:`MiniBatchOTCoupling`) -- training loops
  that wrap ``corrupt`` in a per-sample ``jax.vmap`` must call the
  whole batch through ``corrupt`` directly instead.
  """
  coupling = getattr(process, "coupling", None)
  if coupling is None:
    return
  if getattr(coupling, "is_batch_level", False):
    raise ValueError(
        f"{type(coupling).__name__} is batch-level and cannot be "
        "vmapped per-sample.  Call ``corruption_process.corrupt(key, "
        "x0_batch, time)`` on the whole batch directly."
    )
