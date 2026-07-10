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
HTTP sandbox client (SandboxFusion-compatible).

Used when the ``SANDBOX_ENDPOINT`` environment variable points to a running
sandbox service (e.g. https://github.com/bytedance/SandboxFusion). For fully
local execution without any service, see ``sandbox.subprocess_sandbox``.

Adapted from SimpleTIR (Apache 2.0).
"""

import asyncio
import logging
import os
import random
from typing import List, Optional, Tuple

try:
    import aiohttp

    _AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    aiohttp = None
    _AIOHTTP_AVAILABLE = False


async def _post_snippet(endpoint: str, payload: dict, *, client_timeout: float = 30.0) -> dict:
    """Low-level HTTP POST to the sandbox and return parsed JSON."""
    timeout = aiohttp.ClientTimeout(total=client_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(endpoint, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def single_sandbox(
    code: str,
    stdin: str = "",
    endpoint: str = "",
    language: str = "python",
    compile_timeout: float = 1.0,
    run_timeout: float = 10.0,
    semaphore: Optional[asyncio.Semaphore] = None,
    max_attempts: int = 2,
) -> dict:
    """Submit one code snippet to the sandbox service with retries."""
    if semaphore is None:
        semaphore = asyncio.Semaphore(1)

    payload = {
        "code": code,
        "stdin": stdin if stdin is not None else "",
        "language": language,
        "compile_timeout": compile_timeout,
        "run_timeout": run_timeout,
    }

    response: dict = {"status": "runtime_error", "run_result": {"stdout": "", "stderr": ""}}
    for attempt in range(1, max_attempts + 1):
        try:
            async with semaphore:
                response = await _post_snippet(endpoint, payload)
            break
        except Exception as exc:  # noqa: BLE001
            logging.warning("single_sandbox attempt %s/%s failed: %s", attempt, max_attempts, exc)
            if attempt == max_attempts:
                response = {
                    "status": "runtime_error",
                    "run_result": {"stdout": "", "stderr": str(exc)},
                }
            else:
                await asyncio.sleep(0.5 * 2 ** (attempt - 1) + random.random() * 0.1)

    return response


async def parallel_sandbox(
    tasks: List[str],
    stdin_list: Optional[List[str]] = None,
    num_processes: int = 200,
    run_timeout: Optional[float] = None,
) -> Tuple[List[bool], List[str], List[str]]:
    """Execute snippets concurrently against the HTTP sandbox service.

    Falls back to the local subprocess sandbox when ``SANDBOX_ENDPOINT`` is
    unset or aiohttp is unavailable, so callers never silently lose tool
    execution.

    ``run_timeout`` overrides the per-snippet run limit; when ``None`` the
    ``SANDBOX_RUN_TIMEOUT`` environment default (10s) is used.
    """
    endpoint = os.getenv("SANDBOX_ENDPOINT", None)
    if endpoint is None or not _AIOHTTP_AVAILABLE:
        if endpoint is not None and not _AIOHTTP_AVAILABLE:
            logging.warning("SANDBOX_ENDPOINT is set but aiohttp is not installed; falling back to local subprocess sandbox")
        from sandbox.subprocess_sandbox import parallel_sandbox as local_parallel_sandbox

        return await local_parallel_sandbox(tasks, stdin_list=stdin_list, num_processes=num_processes,
                                            run_timeout=run_timeout)

    if run_timeout is None:
        run_timeout = float(os.getenv("SANDBOX_RUN_TIMEOUT", "10"))
    semaphore = asyncio.Semaphore(num_processes)
    if stdin_list is None:
        coros = [single_sandbox(code=code, endpoint=endpoint, semaphore=semaphore, run_timeout=run_timeout) for code in tasks]
    else:
        assert len(tasks) == len(stdin_list), f"len(tasks) ({len(tasks)}) != len(stdin_list) ({len(stdin_list)})"
        coros = [
            single_sandbox(code=code, stdin=stdin, endpoint=endpoint, semaphore=semaphore, run_timeout=run_timeout)
            for code, stdin in zip(tasks, stdin_list)
        ]
    results = await asyncio.gather(*coros, return_exceptions=True)

    ok_flags: List[bool] = []
    stdouts: List[str] = []
    stderrs: List[str] = []
    for r in results:
        if isinstance(r, Exception):
            ok_flags.append(False)
            stdouts.append("")
            stderrs.append(str(r))
            continue
        ok_flags.append(r.get("status") == "success")
        run_res = r.get("run_result", {}) if isinstance(r, dict) else {}
        stdouts.append(run_res.get("stdout", "") or "")
        stderrs.append(run_res.get("stderr", "") or "")
    return ok_flags, stdouts, stderrs
