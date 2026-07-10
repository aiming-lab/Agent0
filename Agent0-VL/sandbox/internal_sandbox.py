# Copyright 2024 Agent0-VL Authors
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
"""
Default sandbox backend (import-compatibility shim).

Training code imports ``sandbox.internal_sandbox`` when no external sandbox
service is configured. In the open-source release this resolves to the local
subprocess sandbox, which requires no additional infrastructure.
"""

from sandbox.subprocess_sandbox import parallel_sandbox, single_sandbox

__all__ = ["parallel_sandbox", "single_sandbox"]
