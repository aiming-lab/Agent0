# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from importlib.metadata import version, PackageNotFoundError

###
# [SUPPORT AMD:]
import torch
###


def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


package_name = 'vllm'
package_version = get_version(package_name)

###
# [SUPPORT AMD:] guard so import also works on CPU-only machines
try:
    _is_amd = torch.cuda.is_available() and "AMD" in torch.cuda.get_device_name()
except Exception:
    _is_amd = False

if _is_amd and package_version is not None:
    import re
    package_version = re.match(r'(\d+\.\d+\.?\d*)', package_version).group(1)
###

if package_version is not None and package_version <= '0.6.3':
    vllm_mode = 'customized'
    from .vllm_rollout import vLLMRollout
    from .fire_vllm_rollout import FIREvLLMRollout
else:
    # Default to SPMD mode (vllm >= 0.7 or vllm not yet installed)
    vllm_mode = 'spmd'
    try:
        from .vllm_rollout_spmd import vLLMRollout
    except ImportError:
        vLLMRollout = None

# Agent0-VL rollout (always available for SPMD mode)
try:
    from .vllm_agent0_rollout_spmd import vLLMAgent0Rollout
except ImportError:
    vLLMAgent0Rollout = None

__all__ = ['vLLMRollout', 'vLLMAgent0Rollout', 'vllm_mode']
