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

"""Tests for normalization layers."""

import flax.linen as nn
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import normalization
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Type Aliases
################################################################################

NormalizationType = arch_typing.NormalizationType
PyTree = hd_typing.PyTree

################################################################################
# MARK: Helpers
################################################################################


def _pad_to_shape(
    arr: jnp.ndarray, target_shape: tuple[int, ...]
) -> jnp.ndarray:
  """Pads an array to a target shape."""
  return (
      jnp.zeros(target_shape, dtype=arr.dtype)
      .at[: arr.shape[0], : arr.shape[1], : arr.shape[2], : arr.shape[3]]
      .set(arr)
  )


def _perturb_params(params: PyTree, key: jax.Array) -> PyTree:
  leaves, treedef = jax.tree_util.tree_flatten(params)
  keys_list = jax.random.split(key, len(leaves))
  key_tree = jax.tree_util.tree_unflatten(treedef, keys_list)
  return jax.tree_util.tree_map(
      lambda p, k: p + 0.5 * jax.random.normal(k, p.shape),
      params,
      key_tree,
  )


################################################################################
# MARK: Tests
################################################################################


class NormalizationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.x_shape = (2, 8, 8, 10)
    self.x = jax.random.normal(self.rng, self.x_shape)
    self.c_shape = (2, 32)
    self.c = jax.random.normal(self.rng, self.c_shape)
    self.num_groups = 5

    # Sequence lengths for testing padding invariance.
    unpadded_seq_len = 4
    small_seq_len = 6
    large_seq_len = 8
    x_shape_small = (
        self.x_shape[0],
        self.x_shape[1],
        small_seq_len,
        self.x_shape[3],
    )
    x_shape_large = (
        self.x_shape[0],
        self.x_shape[1],
        large_seq_len,
        self.x_shape[3],
    )

    x_slice = self.x[:, :, :unpadded_seq_len, :]

    self.x_small = _pad_to_shape(arr=x_slice, target_shape=x_shape_small)
    self.x_large = _pad_to_shape(arr=x_slice, target_shape=x_shape_large)
    self.unpadded_seq_len = unpadded_seq_len
    self.small_seq_len = small_seq_len
    self.large_seq_len = large_seq_len

  def test_unconditional_rmsnorm_at_init(self):
    """Tests unconditional RMSNorm at init."""
    norm_layer = normalization.NormalizationLayer(
        normalization_method=NormalizationType.RMS_NORM,
        conditional=False,
    )
    params = norm_layer.init(self.rng, self.x)

    output_new = norm_layer.apply(params, self.x)

    x2 = jnp.mean(self.x**2, -1, keepdims=True)
    output_ref = self.x * lax.rsqrt(x2 + norm_layer.epsilon)

    self.assertEqual(output_new.shape, self.x_shape)
    np.testing.assert_allclose(output_new, output_ref, rtol=1e-5, atol=1e-5)

  def test_conditional_rmsnorm_at_init(self):
    """Tests conditional normalization at init when scale=0 and shift=0."""
    norm_layer = normalization.NormalizationLayer(
        normalization_method=NormalizationType.RMS_NORM,
        conditional=True,
    )
    params = norm_layer.init(self.rng, self.x, self.c)
    output = norm_layer.apply(params, self.x, self.c)
    self.assertEqual(output.shape, self.x_shape)

    # At init, scale=0 and shift=0, so output is same as in unconditional.
    x2 = jnp.mean(self.x**2, -1, keepdims=True)
    output_ref = self.x * lax.rsqrt(x2 + norm_layer.epsilon)
    np.testing.assert_allclose(
        output,
        output_ref,
        rtol=1e-5,
        atol=1e-5,
        err_msg=(
            "Conditional output should be same as unconditional output at"
            " params init."
        ),
    )

  def test_conditional_rmsnorm_perturbed(self):
    """Tests conditional normalization when scale!=0 and shift!=0."""
    norm_layer = normalization.NormalizationLayer(
        normalization_method=NormalizationType.RMS_NORM,
        conditional=True,
    )
    params = norm_layer.init(self.rng, self.x, self.c)
    params_perturbed = _perturb_params(params=params, key=self.rng)
    output_perturbed = norm_layer.apply(params_perturbed, self.x, self.c)

    # Compute unconditional output for comparison.
    x2 = jnp.mean(self.x**2, -1, keepdims=True)
    output_ref = self.x * lax.rsqrt(x2 + norm_layer.epsilon)

    self.assertEqual(output_perturbed.shape, self.x_shape)
    self.assertFalse(
        np.allclose(output_perturbed, output_ref, rtol=1e-5, atol=1e-5),
        msg=(
            "Conditional output should be different from unconditional output"
            " after perturbing params."
        ),
    )

  def test_unconditional_groupnorm_at_init(self):
    """Tests unconditional GroupNorm at init."""
    norm_layer = normalization.NormalizationLayer(
        normalization_method=NormalizationType.GROUP_NORM,
        conditional=False,
        num_groups=self.num_groups,
    )
    params = norm_layer.init(self.rng, self.x)
    output_new = norm_layer.apply(params, self.x)

    norm_ref = nn.GroupNorm(num_groups=self.num_groups)
    params_ref = norm_ref.init(self.rng, self.x)
    output_ref = norm_ref.apply(params_ref, self.x)

    self.assertEqual(output_new.shape, self.x_shape)
    np.testing.assert_allclose(output_new, output_ref, rtol=1e-5, atol=1e-5)

  def test_conditional_groupnorm_at_init(self):
    """Tests conditional GroupNorm at init when scale=0 and shift=0."""
    norm_layer = normalization.NormalizationLayer(
        normalization_method=NormalizationType.GROUP_NORM,
        conditional=True,
        num_groups=self.num_groups,
    )
    params = norm_layer.init(self.rng, self.x, self.c)
    output_new = norm_layer.apply(params, self.x, self.c)

    # At init, scale=0 and shift=0, so output is same as in unconditional.
    norm_ref = nn.GroupNorm(num_groups=self.num_groups)
    params_ref = norm_ref.init(self.rng, self.x)
    output_ref = norm_ref.apply(params_ref, self.x)

    self.assertEqual(output_new.shape, self.x_shape)
    np.testing.assert_allclose(output_new, output_ref, rtol=1e-5, atol=1e-5)

  def test_conditional_groupnorm_perturbed(self):
    """Tests conditional GroupNorm when scale!=0 and shift!=0."""
    norm_layer = normalization.NormalizationLayer(
        normalization_method=NormalizationType.GROUP_NORM,
        conditional=True,
        num_groups=self.num_groups,
    )
    params = norm_layer.init(self.rng, self.x, self.c)
    params_perturbed = _perturb_params(params=params, key=self.rng)
    output = norm_layer.apply(params_perturbed, self.x, self.c)

    # Compute unconditional output for comparison.
    norm_ref = nn.GroupNorm(num_groups=self.num_groups)
    params_ref = norm_ref.init(self.rng, self.x)
    output_ref = norm_ref.apply(params_ref, self.x)

    self.assertEqual(output.shape, self.x_shape)
    self.assertFalse(
        np.allclose(output, output_ref, rtol=1e-5, atol=1e-5, equal_nan=True),
        "Conditional output should be different from unconditional output after"
        " perturbing params.",
    )

  def test_rmsnorm_padding_invariance(self):
    """Tests RMSNorm padding invariance.

    Here we test that RMSNorm is padding invariant.
    We consider two data points of shape (b, seq_len, seq_len, c) where n is the
    number of elements. One data point has seq_len=m and the other has
    seq_len=n, with m < n. The first k < m elements are the same in both data
    points, and the rest are 0. We then apply RMSNorm to both data points and
    check that the first k elements are the same in both outputs.
    """

    norm_layer = normalization.NormalizationLayer(
        normalization_method=NormalizationType.RMS_NORM,
        conditional=False,
    )
    params = norm_layer.init(self.rng, self.x_small)
    # Perturb the params.
    params_perturbed = _perturb_params(params=params, key=self.rng)

    out_small = norm_layer.apply(params_perturbed, self.x_small)
    out_large = norm_layer.apply(params_perturbed, self.x_large)
    np.testing.assert_allclose(
        out_small[:, :, : self.unpadded_seq_len, :],
        out_large[:, :, : self.unpadded_seq_len, :],
        atol=1e-5,
    )

  def test_groupnorm_padding_non_invariance(self):
    """Tests GroupNorm padding invariance."""

    norm_layer = normalization.NormalizationLayer(
        normalization_method=NormalizationType.GROUP_NORM,
        conditional=False,
        num_groups=self.num_groups,
    )
    params = norm_layer.init(self.rng, self.x_small)
    # Perturb the params.
    params_perturbed = _perturb_params(params=params, key=self.rng)

    out_small = norm_layer.apply(params_perturbed, self.x_small)
    out_large = norm_layer.apply(params_perturbed, self.x_large)
    # GroupNorm normalizes over spatial dims, so it is NOT padding invariant.
    self.assertFalse(
        np.allclose(
            out_small[:, :, : self.unpadded_seq_len, :],
            out_large[:, :, : self.unpadded_seq_len, :],
            atol=1e-5,
        )
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="rmsnorm",
          normalization_method=NormalizationType.RMS_NORM,
          num_groups=None,
          mask_dim=1,
      ),
      dict(
          testcase_name="groupnorm",
          normalization_method=NormalizationType.GROUP_NORM,
          num_groups=5,
          mask_dim=10,
      ),
  )
  def test_masked_padding_invariance(
      self, normalization_method, num_groups, mask_dim
  ):
    """Tests masked padding invariance."""

    mask_shape_small = (
        self.x_shape[0],
        self.x_shape[1],
        self.small_seq_len,
        mask_dim,
    )
    mask_shape_large = (
        self.x_shape[0],
        self.x_shape[1],
        self.large_seq_len,
        mask_dim,
    )

    mask_small = jnp.zeros(mask_shape_small, dtype=jnp.bool_)
    mask_small = mask_small.at[:, :, : self.unpadded_seq_len, :].set(True)

    mask_large = jnp.zeros(mask_shape_large, dtype=jnp.bool_)
    mask_large = mask_large.at[:, :, : self.unpadded_seq_len, :].set(True)

    norm_layer = normalization.NormalizationLayer(
        normalization_method=normalization_method,
        conditional=False,
        num_groups=num_groups,
    )
    params = norm_layer.init(self.rng, self.x_small, mask=mask_small)

    # Perturb the params.
    params_perturbed = _perturb_params(params=params, key=self.rng)

    out_small = norm_layer.apply(
        params_perturbed, self.x_small, mask=mask_small
    )
    out_large = norm_layer.apply(
        params_perturbed, self.x_large, mask=mask_large
    )
    np.testing.assert_allclose(
        out_small[:, :, : self.unpadded_seq_len, :],
        out_large[:, :, : self.unpadded_seq_len, :],
        atol=1e-5,
    )

  def test_groupnorm_broadcastable_mask_fails(self):
    """Tests GroupNorm with a broadcastable mask raises ValueError."""

    mask_shape_large = (self.x_shape[0], self.x_shape[1], self.large_seq_len, 1)

    mask_large = jnp.zeros(mask_shape_large, dtype=jnp.bool_)
    mask_large = mask_large.at[:, :, : self.unpadded_seq_len, :].set(True)

    norm_layer = normalization.NormalizationLayer(
        normalization_method=NormalizationType.GROUP_NORM,
        conditional=False,
        num_groups=self.num_groups,
    )

    with self.assertRaisesRegex(
        ValueError,
        "If using GroupNorm with a mask, the mask's last dimension must"
        " match the input's channel dimension. Otherwise, one cannot"
        " reshape the mask during the grouping operation.",
    ):
      norm_layer.init(self.rng, self.x_large, mask=mask_large)

  def test_rmsnorm_mask_equivalence(self):
    """Tests that RMSNorm produces same values for non-padding tokens with or without mask."""

    # Create a mask for those non-padding tokens.
    mask_large = jnp.zeros(
        (self.x_shape[0], self.x_shape[1], self.large_seq_len, 1),
        dtype=jnp.bool_,
    )
    mask_large = mask_large.at[:, : self.unpadded_seq_len, :].set(True)

    norm_layer = normalization.NormalizationLayer(
        normalization_method=NormalizationType.RMS_NORM,
        conditional=False,
    )
    params = norm_layer.init(self.rng, self.x_large)

    # Perturb the params.
    params_perturbed = _perturb_params(params=params, key=self.rng)

    # Run with and without mask.
    out_no_mask = norm_layer.apply(params_perturbed, self.x_large)
    out_masked = norm_layer.apply(
        params_perturbed, self.x_large, mask=mask_large
    )

    # Check that for the valid tokens, the results are identical.
    np.testing.assert_allclose(
        out_no_mask[:, : self.unpadded_seq_len, :],
        out_masked[:, : self.unpadded_seq_len, :],
        atol=1e-6,
        err_msg=(
            "RMSNorm output for valid tokens should be invariant to the mask."
        ),
    )


if __name__ == "__main__":
  absltest.main()
