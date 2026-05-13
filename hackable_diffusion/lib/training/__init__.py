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

"""API for diffusion losses + time sampling.

distnl-side extensions (energy / riesz / pseudolikelihood) re-exported
alongside the upstream loss surface so callers can switch their imports
from the pre-rename ``lib.loss`` to ``lib.training`` mechanically.
"""

# pylint: disable=g-importing-member
from hackable_diffusion.lib.training.base import DiffusionLoss
from hackable_diffusion.lib.training.base import WeightFn
from hackable_diffusion.lib.training.discrete_loss import compute_discrete_diffusion_loss
from hackable_diffusion.lib.training.discrete_loss import MD4Loss
from hackable_diffusion.lib.training.discrete_loss import NoWeightDiscreteLoss
from hackable_diffusion.lib.training.gaussian_loss import compute_continuous_diffusion_loss
from hackable_diffusion.lib.training.gaussian_loss import NoWeightGaussianLoss
from hackable_diffusion.lib.training.gaussian_loss import SiD2Loss
from hackable_diffusion.lib.training.scoring_rules import compute_energy_score_loss
from hackable_diffusion.lib.training.scoring_rules import EnergyScoreLoss
from hackable_diffusion.lib.training.spectral_riesz import (
    compute_riesz_energy_score_loss,
)
from hackable_diffusion.lib.training.spectral_riesz import (
    make_sphere_riesz_distance_fn,
)
from hackable_diffusion.lib.training.spectral_riesz import make_torus_modes
from hackable_diffusion.lib.training.spectral_riesz import (
    make_torus_riesz_distance_fn,
)
from hackable_diffusion.lib.training.spectral_riesz import RiemannianDistanceFn
from hackable_diffusion.lib.training.spectral_riesz import RieszEnergyScoreLoss
from hackable_diffusion.lib.training.sequence_pseudolikelihood import BiasSiteFn
from hackable_diffusion.lib.training.sequence_pseudolikelihood import (
    compute_masked_pseudolikelihood_loss,
)
from hackable_diffusion.lib.training.sequence_pseudolikelihood import (
    compute_masked_pseudolikelihood_nce_loss,
)
from hackable_diffusion.lib.training.sequence_pseudolikelihood import EnergySequenceFn
from hackable_diffusion.lib.training.sequence_pseudolikelihood import (
    MaskedPseudolikelihoodLoss,
)
from hackable_diffusion.lib.training.sequence_pseudolikelihood import (
    MaskedPseudolikelihoodNCELoss,
)

# pylint: enable=g-importing-member
