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

A :class:`Coupling` re-pairs an ``(x_0, x_1)`` batch without producing
``x_1`` itself.  Production is the :class:`Prior`'s job (see
:mod:`priors`).

- :class:`IndependentCoupling`: identity; the trivial coupling.
  Re-exported from :mod:`base` -- defined there because it's
  :class:`InterpolantProcess`'s default and must avoid a circular
  import.
- :class:`MiniBatchOTCoupling`: entropic-OT matching via ott-jax
  Sinkhorn (Tong et al. 2024).  Permutes ``x_1`` within the batch so
  paired indices minimise transport cost.  ``is_batch_level = True``.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import jax
import jax.numpy as jnp

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.corruption import base

PRNGKey = hd_typing.PRNGKey
DataTree = hd_typing.DataTree

Coupling = base.Coupling
# Re-export the trivial identity coupling so callers can write
# ``from hackable_diffusion.lib.corruption import couplings;
#   couplings.IndependentCoupling()`` symmetrically with the OT one.
IndependentCoupling = base.IndependentCoupling


@dataclasses.dataclass(kw_only=True, frozen=True)
class MiniBatchOTCoupling(Coupling):
  """Mini-batch entropic-OT re-pairer for ``(x_0, x_1)``.

  Given an ``(x_0, x_1)`` batch, computes the entropy-regularised
  Sinkhorn transport plan between the two empirical distributions and
  re-samples ``x_0`` so that ``(x_0[i], x_1[i])`` minimises transport
  cost.  **``x_1`` order is preserved** -- any conditioning,
  embeddings, or aux metadata aligned with ``x_1`` in the dataset
  stays index-aligned with the post-pairing batch.

  Batch-level: ``is_batch_level = True``.  Training loops must call
  ``corrupt`` on the whole batch at once, not vmap over it.

  Gradients do not flow through the coupling output
  (``jax.lax.stop_gradient``).  This matches the OT-CFM literature
  (Tong et al. 2024) where the plan is computed purely as a matching
  signal, not a differentiable term.

  Configuration surface is intentionally narrow: ``epsilon`` (Sinkhorn
  regularisation) and ``num_iters`` only.  Users who need other
  ott-jax knobs (unbalanced, low-rank, custom cost) can subclass and
  override :meth:`_transport_plan`.
  """

  epsilon: float = 0.01
  num_iters: int = 100
  is_batch_level: ClassVar[bool] = True

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

  def __call__(
      self,
      key: PRNGKey,
      x0: DataTree,
      x1: DataTree,
  ) -> tuple[DataTree, DataTree]:
    x0_flat = x0.reshape(x0.shape[0], -1)
    x1_flat = x1.reshape(x1.shape[0], -1)

    # ``plan[i, j]`` is the OT mass from ``x_0[i]`` to ``x_1[j]``.
    # For each ``x_1[j]`` (the order we want to preserve), sample which
    # ``x_0[i]`` to pair with -- conditional Categorical over rows.
    plan = self._transport_plan(x0_flat, x1_flat)
    col_sums = jnp.sum(plan, axis=0, keepdims=True)
    col_log_probs = jnp.log(
        jnp.clip(plan / jnp.maximum(col_sums, 1e-30), 1e-30, None),
    )
    # ``jax.random.categorical`` samples over the last axis; transpose
    # so each "row" of the input is a distribution over ``x_0`` rows
    # for a given ``x_1`` index ``j``.
    indices = jax.random.categorical(key, col_log_probs.T, axis=-1)

    x0_matched = x0[indices]
    return jax.lax.stop_gradient(x0_matched), x1


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
