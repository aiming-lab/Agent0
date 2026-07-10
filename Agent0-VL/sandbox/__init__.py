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
Sandboxed code execution for Agent0-VL tool-integrated reasoning.

Backends:
- ``sandbox.subprocess_sandbox``: local subprocess execution (default; no
  external service required).
- ``sandbox.local_sandbox``: HTTP client for a SandboxFusion-compatible
  service, enabled by setting ``SANDBOX_ENDPOINT``.
- ``sandbox.internal_sandbox``: alias of the subprocess backend kept for
  import compatibility with the training code.

Use :func:`get_parallel_sandbox` to pick the right backend automatically.
"""

import os


def get_parallel_sandbox():
    """Return the ``parallel_sandbox`` coroutine for the configured backend."""
    if os.getenv("SANDBOX_ENDPOINT", None) is not None:
        from sandbox.local_sandbox import parallel_sandbox
    else:
        from sandbox.subprocess_sandbox import parallel_sandbox
    return parallel_sandbox


__all__ = ["get_parallel_sandbox"]
