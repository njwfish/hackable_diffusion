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

"""Posterior-sampler inference fn (arXiv:2502.02483; posterior-bridge view).

At each reverse step the network takes ``(t, x_t, xi)`` with ``xi ~ N(0, I)``
and outputs an approximate *sample* from the clean-endpoint posterior
``p_{0|t}(x_0 | x_t)`` rather than its mean.  In the posterior-bridge
framework this is the local posterior the bridge step
``K_{s|0,t}(. | x_0, x_t)`` averages against; everything downstream
(twists, projection, SMC potentials) reads it as a posterior sample
rather than a denoiser point estimate.

The sampling-loop scan body passes us the step's rng; we use it to draw
``xi``, inject it via the same ``xi_injector`` used at training, and
call the network once per step.  Calling this fn ``R`` times with
independent rngs gives an ``R``-sample posterior cloud at the same
``x_t`` --  see :func:`make_posterior_cloud_fn` in
``hackable_diffusion.lib.guidance.denoisers`` for that helper.

The network is expected to be shape-preserving under the doubled input --
see :mod:`hackable_diffusion.lib.posterior` for the training-time
invariant (the literature name "distributional diffusion" is preserved
in citations) and
:class:`hackable_diffusion.lib.architecture.NoiseTrimBackbone` for the
canonical wrapper.
"""

import dataclasses

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import posterior
from hackable_diffusion.lib.inference import base
import flax.linen as nn
import jax
import kauldron.ktyping as kt

################################################################################
# MARK: Type Aliases
################################################################################

PyTree = hd_typing.PyTree

Conditioning = hd_typing.Conditioning
DataArray = hd_typing.DataArray
DataTree = hd_typing.DataTree
PRNGKey = hd_typing.PRNGKey
TargetInfoTree = hd_typing.TargetInfoTree
TimeTree = hd_typing.TimeTree

InferenceFn = base.InferenceFn

XiInjector = posterior.XiInjector

# Salt mixed into the per-step rng before drawing xi, so any other consumer
# of the same step rng (e.g. the Z-noise in a stochastic DDIM update) gets
# an independent stream.
_XI_RNG_SALT = 0x1D01  # distinctive constant for traceability in logs.


################################################################################
# MARK: PosteriorSamplerInferenceFn
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class PosteriorSamplerInferenceFn(InferenceFn):
  """Stochastic inference fn: calls the network with a per-step xi draw.

  Returns one approximate sample from the clean-endpoint posterior
  ``p_{0|t}(. | x_t)`` per call.  The trained network must be
  shape-preserving under ``[x_t, xi]`` doubled inputs -- use
  :class:`hackable_diffusion.lib.architecture.NoiseTrimBackbone` around
  any plain backbone whose I/O share a last-axis layout, or a dedicated
  posterior-cloud subclass for backbones that reshape the last axis into
  an image (e.g.
  ``mdt.model.unet_patch_distributional.DistributionalUNetPatch`` --
  "distributional" is the literature naming).
  Either way this class simply injects ``xi`` and forwards; it never
  trims output.

  One xi draw per reverse step -- not a population.  The energy-score
  training is what buys the ability to take big steps with a single
  sample.  Algorithms that need an ``R``-sample posterior cloud at the
  same ``x_t`` (SMC potential estimation, projection) call this fn
  ``R`` times with independent rngs; see :func:`make_posterior_cloud_fn`
  for that helper.

  Attributes:
    network: The trained Linen diffusion network.
    params: The trained parameters tree.
    xi_injector: How to combine ``x_t`` and ``xi`` into the network's input.
      Default channel-concat along the last axis. Must match whatever was
      used at training time (otherwise the learned function sees a
      different input distribution than it was trained on).
  """

  network: nn.Module
  params: PyTree
  xi_injector: XiInjector = dataclasses.field(
      default=posterior.channel_concat_xi
  )

  @kt.typechecked
  def __call__(
      self,
      time: TimeTree,
      xt: DataTree,
      conditioning: Conditioning | None,
      rng: PRNGKey | None = None,
  ) -> TargetInfoTree:
    if rng is None:
      raise ValueError(
          "PosteriorSamplerInferenceFn requires an rng; the sampling loop "
          "should be passing one from step_info. If you're calling this fn "
          "directly, pass rng= explicitly."
      )
    xi_rng = jax.random.fold_in(rng, _XI_RNG_SALT)
    xi = jax.random.normal(xi_rng, xt.shape, dtype=xt.dtype)
    xt_ext = self.xi_injector(xt, xi)
    return self.network.apply(
        {"params": self.params},
        time=time,
        xt=xt_ext,
        conditioning=conditioning,
        is_training=False,
    )
