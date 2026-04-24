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

"""Base classes and wrappers for noise processes.

The generic data-to-data corruption process factors into three composable
primitives:

- :class:`Coupling`: samples ``x_1`` given a batch of ``x_0``.  Covers
  independent-source (diffusion; ``StandardNormalSource`` etc. are
  couplings that ignore ``x_0`` values and use only its shape),
  deterministic (blur/mask), and mini-batch-OT couplings.  Per-sample
  couplings are vmap-friendly; batch-level couplings (OT) are not --
  see :data:`Coupling.is_batch_level`.
- :class:`Interpolant`: deterministic path ``x_t = I(t, x_0, x_1)``,
  optionally augmented with ``+ gamma(t) z`` for stochastic interpolants.
- :class:`TargetAdapter`: emits the ``target_info`` dict from
  ``(x_0, x_1, z, x_t, dx_t/dt, t, interpolant)``.
  ``GaussianSourceTargets`` emits ``{x0, x1, score, velocity, v}`` --
  valid only when the coupling's ``x_1`` marginal is
  :class:`StandardNormalSource`.  ``VelocityOnlyTargets`` emits
  ``{x0, x1, velocity}`` for an arbitrary interpolant.

These compose inside :class:`InterpolantProcess`, which satisfies the
legacy :class:`CorruptionProcess` Protocol.  ``GaussianProcess`` and
``RiemannianProcess`` are kept as thin shim constructors that build the
equivalent :class:`InterpolantProcess`.

See ``docs/interpolant_refactor_plan.md`` for the full design.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar, Protocol

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
import jax
import kauldron.ktyping as kt

################################################################################
# MARK: Type Aliases
################################################################################

PyTree = hd_typing.PyTree
PRNGKey = hd_typing.PRNGKey

DataTree = hd_typing.DataTree
TargetInfoTree = hd_typing.TargetInfoTree
TimeTree = hd_typing.TimeTree
ScheduleInfoTree = hd_typing.ScheduleInfoTree

################################################################################
# MARK: Protocols
################################################################################


class CorruptionProcess(Protocol):
  """Base class for all corruption processes (continuous and discrete)."""

  def corrupt(
      self,
      key: PRNGKey,
      x0: DataTree,
      time: TimeTree,
  ) -> tuple[DataTree, TargetInfoTree]:
    """Corrupt x0 according to time, and return xt and targets info."""

  def sample_from_invariant(
      self,
      key: PRNGKey,
      data_spec: DataTree,
  ) -> DataTree:
    """Sample from the invariant distribution."""

  def convert_predictions(
      self,
      prediction: TargetInfoTree,
      xt: DataTree,
      time: TimeTree,
  ) -> TargetInfoTree:
    """Convert the prediction to the target type."""

  def get_schedule_info(self, time: TimeTree) -> ScheduleInfoTree:
    """Get the schedule info for the given time."""


################################################################################
# MARK: Interpolant-process primitives (Coupling / Interpolant / TargetAdapter)
################################################################################


class Coupling(Protocol):
  """Samples ``x_1`` given a batch of ``x_0``.

  MUST operate on whole batches: OT-style couplings are inherently set
  operations and cannot be vmapped per-sample.  Couplings that happen
  to ignore ``x_0`` (``StandardNormalSource``, ``UniformManifoldSource``,
  ``DataloaderSource``) or treat it per-sample
  (``DeterministicCoupling``) remain vmap-friendly.

  ``marginal`` returns an ``x_0``-independent ``Coupling`` giving the
  ``x_1`` distribution.  For couplings that already ignore ``x_0``
  (the "sources") it is ``self``.  For :class:`MiniBatchOTCoupling` it
  is the underlying source.  For :class:`DeterministicCoupling` it is
  ``None`` (no well-defined marginal).  :meth:`InterpolantProcess.sample_from_invariant`
  consults it at inference time.
  """

  is_batch_level: ClassVar[bool]
  marginal: 'Coupling | None'

  def sample(self, key: PRNGKey, x0: DataTree) -> DataTree: ...


class Interpolant(Protocol):
  """Deterministic path ``I(t, x_0, x_1)``, optionally augmented with ``gamma(t) z``.

  Full stochastic-interpolant form:

      x_t = I(t, x_0, x_1) + gamma(t) z,   z ~ N(0, I) indep. of (x_0, x_1)

  When the concrete interpolant has no noise augmentation,
  ``needs_noise`` is ``False`` and ``eval`` is called with ``z=None``.
  When ``True``, :class:`InterpolantProcess` draws ``z`` and threads it
  in.  The flag is a ``ClassVar`` so it's compile-time-determinate under
  ``jit`` -- no Python branching on array values inside ``corrupt``.

  ``schedule`` is whatever schedule object the concrete interpolant
  holds (``GaussianSchedule``, ``RiemannianSchedule``, a pair of
  ``(alpha, beta, gamma)`` callables for stochastic interpolants).
  Forwarded to samplers that peek.

  ``eval`` returns ``(x_t, dx_t/dt)`` as a tuple -- shared-work shortcut
  for interpolants whose path and velocity reuse schedule evaluations.
  ``TargetAdapter``s pull ``dx_t/dt`` from the second element.
  """

  schedule: object
  needs_noise: ClassVar[bool]

  def eval(
      self,
      x0: DataTree,
      x1: DataTree,
      t: TimeTree,
      z: DataTree | None = None,
  ) -> tuple[DataTree, DataTree]: ...


class TargetAdapter(Protocol):
  """Emits ``target_info`` from ``(x_0, x_1, z, x_t, dx_t/dt, t, interpolant)``.

  Different adapters are valid for different sources.  Each adapter
  declares the keys it emits via :attr:`emitted_keys`; downstream
  compatibility checks (e.g. ``DiffusionSampler``'s stepper wiring)
  consult this to fail-fast when a stepper asks for a key the adapter
  doesn't produce.

  ``dxt_dt`` is the velocity computed by :meth:`Interpolant.eval`,
  threaded through so adapters don't redundantly re-call ``eval``.

  ``convert`` replaces the legacy ``CorruptionProcess.convert_predictions``
  method -- takes a single-key prediction dict in any of the supported
  parameterisations and fans it out into the full set.
  """

  emitted_keys: ClassVar[frozenset[str]]

  def emit(
      self,
      *,
      x0: DataTree,
      x1: DataTree,
      z: DataTree | None,
      xt: DataTree,
      dxt_dt: DataTree,
      t: TimeTree,
      interpolant: 'Interpolant',
  ) -> TargetInfoTree: ...

  def convert(
      self,
      *,
      prediction: TargetInfoTree,
      xt: DataTree,
      t: TimeTree,
      interpolant: 'Interpolant',
  ) -> TargetInfoTree: ...


################################################################################
# MARK: InterpolantProcess -- the composed process
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class InterpolantProcess(CorruptionProcess):
  """Composed ``(coupling, interpolant, targets)`` corruption process.

  Satisfies the legacy :class:`CorruptionProcess` Protocol.  Every
  downstream consumer that expects a ``CorruptionProcess`` continues to
  work.

  Construction:

      GaussianProcess(schedule=CosineSchedule())
        => InterpolantProcess(
               coupling=StandardNormalSource(),
               interpolant=LinearInterpolant(schedule=CosineSchedule()),
               targets=GaussianSourceTargets(),
           )

  Data-to-data OT flow matching:

      InterpolantProcess(
          coupling=MiniBatchOTCoupling(source=DataloaderSource(...)),
          interpolant=LinearInterpolant(schedule=RFSchedule()),
          targets=VelocityOnlyTargets(),
      )

  Stochastic interpolant:

      InterpolantProcess(
          coupling=StandardNormalSource(),
          interpolant=StochasticInterpolant(alpha, beta, gamma),
          targets=VelocityOnlyTargets(),
      )
  """

  coupling: Coupling
  interpolant: Interpolant
  targets: TargetAdapter

  @property
  def schedule(self):
    # Downstream samplers and guidance primitives read ``corruption_process.schedule``
    # directly; expose the interpolant's schedule at the process level.  The
    # ``GaussianProcess`` / ``RiemannianProcess`` shims delegate (not subclass)
    # so their own ``schedule`` dataclass fields don't collide with this property.
    return self.interpolant.schedule

  def corrupt(
      self,
      key: PRNGKey,
      x0: DataTree,
      time: TimeTree,
  ) -> tuple[DataTree, TargetInfoTree]:
    # Key split is conditional on the interpolant needing z.  For the
    # shim Gaussian / Riemannian paths, LinearInterpolant /
    # GeodesicInterpolant both set needs_noise=False, so the original
    # key flows straight into the source -- byte-identical to legacy.
    if self.interpolant.needs_noise:
      key_coupling, key_z = jax.random.split(key)
      z = jax.random.normal(key_z, shape=x0.shape)
    else:
      key_coupling, z = key, None
    x1 = self.coupling.sample(key_coupling, x0)
    xt, dxt_dt = self.interpolant.eval(x0, x1, time, z)
    target_info = self.targets.emit(
        x0=x0, x1=x1, z=z, xt=xt, dxt_dt=dxt_dt,
        t=time, interpolant=self.interpolant,
    )
    return xt, target_info

  def sample_from_invariant(
      self,
      key: PRNGKey,
      data_spec: DataTree,
  ) -> DataTree:
    """Sample from the ``x_1`` marginal.

    Raises if the coupling has no well-defined marginal
    (``DeterministicCoupling``'s ``x_1`` depends on ``x_0``).
    """
    if self.coupling.marginal is None:
      raise ValueError(
          f"{type(self.coupling).__name__} has no well-defined x_1 marginal; "
          "``sample_from_invariant`` is not applicable.  Use an "
          "independent- or OT-coupling, or sample x_1 via the coupling's "
          "own sample() after providing an x_0 batch."
      )
    return self.coupling.marginal.sample(key, data_spec)

  def convert_predictions(
      self,
      prediction: TargetInfoTree,
      xt: DataTree,
      time: TimeTree,
  ) -> TargetInfoTree:
    return self.targets.convert(
        prediction=prediction, xt=xt, t=time, interpolant=self.interpolant,
    )

  def get_schedule_info(self, time: TimeTree) -> ScheduleInfoTree:
    return self.schedule.evaluate(time)


################################################################################
# MARK: NestedProcess
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class NestedProcess(CorruptionProcess):
  """Wrapper for a pytree of noise schedules that is mapped over the data.

  Enables using different noise schedules for different input modalities.
  E.g. a gaussian schedule for the image and a categorical schedule for the
  labels.
  """

  processes: PyTree[CorruptionProcess]

  @kt.typechecked
  def sample_from_invariant(
      self,
      key: PRNGKey,
      data_spec: DataTree,
  ) -> DataTree:
    """Sample from the invariant distribution."""
    return utils.tree_map_with_key(
        lambda k, process, data: process.sample_from_invariant(k, data),
        key,
        self.processes,
        data_spec,
    )

  @kt.typechecked
  def corrupt(
      self,
      key: PRNGKey,
      x0: DataTree,
      time: TimeTree,
  ) -> tuple[DataTree, TargetInfoTree]:
    x0_structure = jax.tree.structure(x0)
    time_structure = jax.tree.structure(time)
    if x0_structure != time_structure:
      raise ValueError(
          f'x0 and time must have the same structure. Got: {x0_structure=} and'
          f' {time_structure=}'
      )
    xt_and_targets = utils.tree_map_with_key(
        lambda k, process, x, t: process.corrupt(k, x, t),
        key,
        self.processes,
        x0,
        time,
    )
    # Unzip the tree (from a tree of tuples to a tuple of trees)
    xt = jax.tree.map(
        lambda x0, xt_and_targets: xt_and_targets[0], x0, xt_and_targets
    )
    target_info = jax.tree.map(
        lambda x0, xt_and_targets: xt_and_targets[1], x0, xt_and_targets
    )
    return xt, target_info

  @kt.typechecked
  def convert_predictions(
      self,
      prediction: TargetInfoTree,
      xt: DataTree,
      time: TimeTree,
  ) -> TargetInfoTree:
    """Convert the prediction to the target type."""
    return jax.tree.map(
        lambda process, pred, xt, time: process.convert_predictions(
            pred, xt, time
        ),
        self.processes,
        prediction,
        xt,
        time,
    )

  @kt.typechecked
  def get_schedule_info(self, time: TimeTree) -> ScheduleInfoTree:
    """Get the schedule info for the given time."""
    return jax.tree.map(
        lambda process, t: process.get_schedule_info(t),
        self.processes,
        time,
    )
