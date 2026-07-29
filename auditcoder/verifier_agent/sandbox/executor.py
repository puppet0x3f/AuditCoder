"""Sandboxed Python code executor with resource limits."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field


@dataclass
class ExecutionResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    time_ms: float = 0.0
    memory_kb: float = 0.0
    timed_out: bool = False


_IS_DARWIN = platform.system() == "Darwin"


def _ensure_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _build_wrapper(code_path: str, memory_limit_mb: int) -> str:
    """Build a Python wrapper script that sets resource limits then execs the target."""
    lines = [
        "import resource, sys, os",
        f"mem_bytes = {memory_limit_mb} * 1024 * 1024",
    ]
    if not _IS_DARWIN:
        lines.append("resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))")
    lines += [
        f'exec(open({code_path!r}, encoding="utf-8").read())',
    ]
    return "\n".join(lines)


class SandboxExecutor:
    """Execute Python code in a subprocess with time/memory limits."""

    def __init__(
        self,
        default_timeout_s: float = 5.0,
        default_memory_limit_mb: int = 256,
    ) -> None:
        self.default_timeout_s = default_timeout_s
        self.default_memory_limit_mb = default_memory_limit_mb

    def run(
        self,
        code: str,
        stdin: str = "",
        timeout_s: float | None = None,
        memory_limit_mb: int | None = None,
    ) -> ExecutionResult:
        timeout = timeout_s if timeout_s is not None else self.default_timeout_s
        mem_limit = memory_limit_mb if memory_limit_mb is not None else self.default_memory_limit_mb

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as code_file:
            code_file.write(code)
            code_path = code_file.name

        wrapper_code = _build_wrapper(code_path, mem_limit)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_wrapper.py", delete=False, encoding="utf-8"
        ) as wrapper_file:
            wrapper_file.write(wrapper_code)
            wrapper_path = wrapper_file.name

        result = ExecutionResult()
        try:
            start = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, wrapper_path],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed = time.perf_counter() - start

            result.stdout = proc.stdout
            result.stderr = proc.stderr
            result.exit_code = proc.returncode
            result.time_ms = elapsed * 1000

        except subprocess.TimeoutExpired as exc:
            result.timed_out = True
            result.stdout = _ensure_text(exc.stdout)
            result.stderr = _ensure_text(exc.stderr)
            result.exit_code = -1
            result.time_ms = timeout * 1000

        finally:
            for path in (code_path, wrapper_path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

        return result

    def run_function_call(
        self,
        function_code: str,
        function_name: str,
        args_code: str,
        timeout_s: float | None = None,
        memory_limit_mb: int | None = None,
    ) -> ExecutionResult:
        """Execute a function with given arguments and print the result."""
        full_code = (
            f"{function_code}\n\n"
            f"_result_ = {function_name}({args_code})\n"
            f"print(_result_)\n"
        )
        return self.run(full_code, timeout_s=timeout_s, memory_limit_mb=memory_limit_mb)
