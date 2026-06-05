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

"""Shared pytest configuration for the test suite.

``jax_enable_x64`` is a *process-global* flag.  A few test modules need 64-bit
precision (linear-algebra solvers, optimal-transport couplings) and enable it;
if they enable it at import time it silently leaks into every test collected
afterwards, changing float precision and PRNG bit-width.  That makes otherwise
hermetic tests order-dependent -- e.g. schedules / step-sampler tests that pin
float32 expected values start failing only because an x64-enabling module was
collected first.

This autouse fixture resets the flag to its default (disabled) before every
test, so the global default is deterministic regardless of collection order.
Modules that genuinely require 64-bit precision opt back in with their own
module-local ``autouse`` fixture, which runs after this one.
"""

import jax
import pytest


@pytest.fixture(autouse=True)
def _default_jax_x64():
  """Force the process-global ``jax_enable_x64`` flag to its default per test."""
  jax.config.update("jax_enable_x64", False)
  yield
