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

The generic data-to-data corruption process factors into four
composable primitives:

- :class:`Prior`: produces an ``x_1`` marginal from an ``x_0`` *spec*
  (used purely for shape/dtype).  Covers Gaussian noise
  (:class:`GaussianPrior`), uniform-on-manifold
  (:class:`UniformManifoldPrior`), and deterministic maps
  ``x_1 = f(x_0)`` (:class:`DeterministicPrior`, blur/mask).  A prior
  is *only* needed when the dataset does not yield ``x_1`` directly --
  e.g. classical diffusion, where ``x_1`` is invented by the process.
- :class:`Coupling`: re-pairs an ``(x_0, x_1)`` batch.  The default
  :class:`IndependentCoupling` is identity -- the batch order out is
  the batch order in.  :class:`MiniBatchOTCoupling` permutes ``x_1``
  within the batch via entropic-OT Sinkhorn.  Per-sample couplings are
  vmap-friendly; batch-level ones (OT) are not -- see
  :data:`Coupling.is_batch_level`.
- :class:`Interpolant`: deterministic path ``x_t = I(t, x_0, x_1)``,
  optionally augmented with ``+ gamma(t) z`` for stochastic interpolants.
- :class:`TargetAdapter`: emits the ``target_info`` dict from
  ``(x_0, x_1, z, x_t, dx_t/dt, t, interpolant)``.
  ``GaussianSourceTargets`` emits ``{x0, x1, score, velocity, v}`` --
  valid only when ``x_1`` is Gaussian (:class:`GaussianPrior`).
  ``VelocityOnlyTargets`` emits ``{x0, x1, velocity}`` for arbitrary
  interpolants.

These compose inside :class:`InterpolantProcess`, which satisfies the
legacy :class:`CorruptionProcess` Protocol.  ``GaussianProcess`` and
``RiemannianProcess`` are kept as thin shim constructors that build the
equivalent :class:`InterpolantProcess`.

The Prior/Coupling split makes joint-distribution training first-class:
when the dataset yields paired ``(x_0, x_1)`` (e.g. image + caption
embedding, blurry + sharp, paired timestamps), the user calls
``InterpolantProcess.corrupt(key, x0, time, x1=x1)`` directly -- the
prior is unused, the default :class:`IndependentCoupling` no-ops, and
the same interpolant + targets logic flows.  When the user wants OT
matching of dataset ``x_1`` to ``x_0`` within each batch, they swap in
:class:`MiniBatchOTCoupling`.  No special "DataloaderSource" class is
needed: a paired dataset is just data.

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
# MARK: Interpolant-process primitives (Prior / Coupling / Interpolant / TargetAdapter)
################################################################################


class Prior(Protocol):
  """Produces an ``x_1`` marginal sample given an ``x_0`` *spec*.

  ``x_0`` here is used only for shape/dtype information -- the prior
  itself is *unconditional* on data values (with the documented
  exception of :class:`DeterministicPrior`, which is x0-functional).
  Used in two places:

    1. :meth:`InterpolantProcess.corrupt` when the caller does not
       pass ``x_1`` (the dataset only yields ``x_0``; the process
       invents ``x_1``).
    2. :meth:`InterpolantProcess.sample_from_invariant` at inference
       time, where only an ``x_0`` spec is in scope.
  """

  def sample(self, key: PRNGKey, x0: DataTree) -> DataTree: ...


class Coupling(Protocol):
  """Re-pairs an ``(x_0, x_1)`` batch.

  The default :class:`IndependentCoupling` is the identity -- the
  batch order out is the batch order in, so ``(x_0[i], x_1[i])`` pairs
  are preserved.  :class:`MiniBatchOTCoupling` permutes ``x_0`` within
  the batch via entropic-OT Sinkhorn so paired indices minimise
  transport cost; this is the OT-CFM matching of Tong et al. 2024.

  Couplings that need to see the whole batch (OT) cannot be vmapped
  per-sample; flag them with ``is_batch_level = True``.  The default
  identity coupling is vmap-friendly.

  **Convention: ``x_1`` order is preserved by all shipped couplings**;
  re-pairing permutes ``x_0`` instead.  This keeps any conditioning,
  embeddings, or auxiliary metadata yielded alongside ``x_1`` in the
  dataset index-aligned with the post-pairing batch, so downstream
  code (loss heads, conditioning encoders, target adapters) can
  consume ``x_1``-aligned tensors without any extra bookkeeping.

  A coupling is a *pure re-pairer*: it does not produce ``x_1``.
  Production is the :class:`Prior`'s job (when the dataset doesn't
  supply ``x_1``).  This split makes joint-distribution training and
  OT matching strictly orthogonal -- the dataset chooses what ``x_1``
  is, the coupling chooses how ``x_0`` and ``x_1`` pair up.
  """

  is_batch_level: ClassVar[bool]

  def __call__(
      self,
      key: PRNGKey,
      x0: DataTree,
      x1: DataTree,
  ) -> tuple[DataTree, DataTree]: ...


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
# MARK: IndependentCoupling -- the trivial identity coupling
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class IndependentCoupling:
  """Identity re-pairer: returns ``(x_0, x_1)`` unchanged.

  The default coupling.  Use this whenever the ``(x_0, x_1)`` pairing
  is already correct -- either because the prior produced ``x_1`` from
  scratch (classical diffusion) or because the dataset yields paired
  ``(x_0, x_1)`` and you want to keep its order.
  """

  is_batch_level: ClassVar[bool] = False

  def __call__(
      self,
      key: PRNGKey,
      x0: DataTree,
      x1: DataTree,
  ) -> tuple[DataTree, DataTree]:
    del key
    return x0, x1


_DEFAULT_COUPLING = IndependentCoupling()


################################################################################
# MARK: InterpolantProcess -- the composed process
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class InterpolantProcess(CorruptionProcess):
  """Composed ``(prior, coupling, interpolant, targets)`` corruption process.

  Satisfies the legacy :class:`CorruptionProcess` Protocol -- every
  downstream consumer that expects a ``CorruptionProcess`` continues
  to work because :meth:`corrupt` adds ``x1`` as an *optional keyword*
  (Protocol callers still write ``process.corrupt(key, x0, time)``).

  Construction:

      # Classical Gaussian diffusion (x_1 invented by the process):
      GaussianProcess(schedule=CosineSchedule())
        => InterpolantProcess(
               prior=GaussianPrior(),
               coupling=IndependentCoupling(),    # default
               interpolant=LinearInterpolant(schedule=CosineSchedule()),
               targets=GaussianSourceTargets(),
           )

      # Data-to-data OT flow matching (x_1 from the dataset, OT match):
      InterpolantProcess(
          coupling=MiniBatchOTCoupling(),
          interpolant=LinearInterpolant(schedule=RFSchedule()),
          targets=VelocityOnlyTargets(),
      )
      # ... then at the training loop: process.corrupt(key, x0, time, x1=x1)

      # Stochastic interpolant on Gaussian noise:
      InterpolantProcess(
          prior=GaussianPrior(),
          interpolant=StochasticInterpolant(alpha, beta, gamma),
          targets=VelocityOnlyTargets(),
      )

  ``prior`` is only required when callers don't pass ``x_1`` to
  :meth:`corrupt`.  When the dataset supplies paired ``(x_0, x_1)``,
  ``prior`` can be omitted and the default :class:`IndependentCoupling`
  preserves the dataset pairing.
  """

  interpolant: Interpolant
  targets: TargetAdapter
  prior: 'Prior | None' = None
  coupling: Coupling = dataclasses.field(
      default_factory=lambda: _DEFAULT_COUPLING,
  )

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
      *,
      x1: DataTree | None = None,
  ) -> tuple[DataTree, TargetInfoTree]:
    """Run one corruption step.

    ``x1`` is an *optional* additive kwarg over the Protocol surface:
    Protocol-typed callers keep writing
    ``process.corrupt(key, x0, time)`` and the prior fills in ``x_1``;
    paired-data callers pass ``x1=x1`` and skip the prior.

    Key flow is a uniform three-way split into ``(prior, coupling, z)``
    consumers regardless of which path is taken; unused slots simply
    don't read their key.  This drops byte-parity with the pre-refactor
    coupling-as-source implementation but keeps the code branchless.
    """
    key_prior, key_coupling, key_z = jax.random.split(key, 3)
    z = (
        jax.random.normal(key_z, shape=x0.shape)
        if self.interpolant.needs_noise else None
    )
    if x1 is None:
      if self.prior is None:
        raise ValueError(
            'InterpolantProcess.corrupt: x1 was not provided and no prior '
            'is configured.  Either pass x1=... or construct the process '
            'with a prior (e.g. GaussianPrior()).'
        )
      x1 = self.prior.sample(key_prior, x0)
    x0, x1 = self.coupling(key_coupling, x0, x1)
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

    Delegates to the configured :class:`Prior`.  Raises if the process
    has no prior (paired-data setups where ``x_1`` only comes from the
    dataset have no well-defined unconditional marginal).
    """
    if self.prior is None:
      raise ValueError(
          'InterpolantProcess.sample_from_invariant: no prior configured. '
          'Paired-data processes (x_1 only from the dataset) have no '
          'standalone x_1 marginal.'
      )
    return self.prior.sample(key, data_spec)

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
