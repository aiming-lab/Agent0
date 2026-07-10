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
Local subprocess-based code sandbox.

Executes Python snippets in isolated subprocesses with CPU-time, wall-clock
and memory limits. This is the default execution backend when no external
sandbox service (``SANDBOX_ENDPOINT``) is configured, so the repository works
out of the box on a single machine.

The public entry point mirrors the HTTP sandbox API used by SimpleTIR:

    parallel_sandbox(tasks, stdin_list=None, num_processes=...) ->
        (success_list, stdout_list, stderr_list)

Environment variables:
    SANDBOX_RUN_TIMEOUT   wall-clock seconds per snippet (default: 10)
    SANDBOX_CPU_TIMEOUT   CPU seconds per snippet          (default: 10)
    SANDBOX_MEM_LIMIT_MB  address-space limit in MB        (default: 1024)
"""

import asyncio
import os
import sys
from typing import List, Optional, Tuple

_RUN_TIMEOUT = float(os.getenv("SANDBOX_RUN_TIMEOUT", "10"))
_CPU_TIMEOUT = int(os.getenv("SANDBOX_CPU_TIMEOUT", "10"))
_MEM_LIMIT_MB = int(os.getenv("SANDBOX_MEM_LIMIT_MB", "1024"))
_MAX_OUTPUT_BYTES = 64 * 1024  # cap captured stdout/stderr per snippet


def _make_preexec_fn():
    """Build a preexec_fn that applies resource limits (POSIX only)."""
    if os.name != "posix":
        return None

    def _limit_resources():
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (_CPU_TIMEOUT, _CPU_TIMEOUT + 1))
            # RLIMIT_AS is unreliable on macOS; only enforce on Linux.
            if sys.platform.startswith("linux"):
                mem_bytes = _MEM_LIMIT_MB * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except Exception:
            # Never fail the exec because limits could not be applied.
            pass
        os.setsid()  # new process group so we can kill the whole tree

    return _limit_resources


async def single_sandbox(
    code: str,
    stdin: str = "",
    semaphore: Optional[asyncio.Semaphore] = None,
    run_timeout: Optional[float] = None,
) -> dict:
    """Execute one Python snippet in a subprocess.

    Returns a dict shaped like the HTTP sandbox response:
    ``{"status": "success"|..., "run_result": {"stdout": ..., "stderr": ...}}``
    """
    if semaphore is None:
        semaphore = asyncio.Semaphore(1)
    timeout = run_timeout if run_timeout is not None else _RUN_TIMEOUT

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")  # headless matplotlib
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")

    async with semaphore:
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                code,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                preexec_fn=_make_preexec_fn(),
            )
        except Exception as exc:  # spawn failure
            return {
                "status": "runtime_error",
                "run_result": {"stdout": "", "stderr": f"Failed to start sandbox process: {exc}"},
            }

        try:
            stdin_bytes = stdin.encode("utf-8", errors="replace") if stdin else None
            stdout, stderr = await asyncio.wait_for(proc.communicate(input=stdin_bytes), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                if os.name == "posix":
                    import signal

                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except (ProcessLookupError, PermissionError, OSError):
                pass
            # Reap the process to avoid zombies.
            try:
                await proc.wait()
            except Exception:
                pass
            return {
                "status": "timeout",
                "run_result": {"stdout": "", "stderr": ""},
            }
        except Exception as exc:
            return {
                "status": "runtime_error",
                "run_result": {"stdout": "", "stderr": str(exc)},
            }

    stdout_text = stdout[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
    stderr_text = stderr[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")
    status = "success" if proc.returncode == 0 else "runtime_error"
    return {
        "status": status,
        "run_result": {"stdout": stdout_text, "stderr": stderr_text},
    }


async def parallel_sandbox(
    tasks: List[str],
    stdin_list: Optional[List[str]] = None,
    num_processes: int = 256,
    run_timeout: Optional[float] = None,
) -> Tuple[List[bool], List[str], List[str]]:
    """Execute multiple snippets concurrently in local subprocesses.

    Returns three lists of length ``len(tasks)``:
    success flags, stdout texts, and stderr texts.

    ``run_timeout`` overrides the per-snippet wall-clock limit; when ``None`` the
    ``SANDBOX_RUN_TIMEOUT`` environment default (10s) is used.
    """
    # Bound concurrency by CPU count to avoid fork storms on large batches.
    cpu_bound = max(2, (os.cpu_count() or 4))
    semaphore = asyncio.Semaphore(max(1, min(num_processes, cpu_bound * 2)))

    if stdin_list is None:
        coros = [single_sandbox(code, semaphore=semaphore, run_timeout=run_timeout) for code in tasks]
    else:
        assert len(tasks) == len(stdin_list), f"len(tasks) ({len(tasks)}) != len(stdin_list) ({len(stdin_list)})"
        coros = [single_sandbox(code, stdin=stdin, semaphore=semaphore, run_timeout=run_timeout)
                 for code, stdin in zip(tasks, stdin_list)]

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
        run_res = r.get("run_result", {})
        stdouts.append(run_res.get("stdout", ""))
        stderrs.append(run_res.get("stderr", ""))
    return ok_flags, stdouts, stderrs


if __name__ == "__main__":
    demo = [
        "print(sum(range(10)))",
        "import time\ntime.sleep(30)\nprint('never')",
        "raise ValueError('boom')",
    ]
    ok, out, err = asyncio.run(parallel_sandbox(demo))
    for o, s, e in zip(ok, out, err):
        print(f"success={o} stdout={s!r} stderr={e[-80:]!r}")
