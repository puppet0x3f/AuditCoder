"""Generated-code executors.

The default ``host`` backend preserves the historical subprocess behaviour for
local development and tests.  The opt-in ``container`` backend runs generated
code with Docker or Podman and applies the repository's isolation policy.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


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
    """Build a wrapper that applies available limits before executing the target.

    ``resource`` is POSIX-only.  The host backend is also used on Windows for
    local development and controlled experiments, so importing it
    unconditionally makes every generated program fail before it starts.
    Windows still keeps the subprocess timeout enforced by ``subprocess.run``;
    the address-space limit is simply unavailable there.
    """
    if os.name == "nt":
        lines = ["import sys, os"]
    else:
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


def _positive_number(value: str, *, name: str, integer: bool = False) -> float | int:
    try:
        parsed: float | int = int(value) if integer else float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive number")
    return parsed


def _container_user() -> str:
    """Return a non-root uid:gid that can read host-created temporary files."""
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if getuid is None or getgid is None:
        return "65532:65532"
    uid, gid = getuid(), getgid()
    if uid == 0:
        return "65532:65532"
    return f"{uid}:{gid}"


def _mount_spec(source: Path, target: str, *, read_only: bool) -> str:
    source_text = str(source)
    if "," in source_text or "," in target:
        raise ValueError("container bind-mount paths must not contain commas")
    fields = f"type=bind,source={source_text},target={target}"
    return f"{fields},readonly" if read_only else fields


def _build_container_command(
    *,
    runtime: str,
    image: str,
    code_directory: Path,
    cidfile: Path,
    memory_limit_mb: int,
    cpus: float,
    pids_limit: int,
    tmpfs_size_mb: int,
    workspace: Path | None = None,
    writable_workspace: bool = False,
) -> list[str]:
    """Build the fail-closed Docker/Podman invocation used by the backend."""
    command = [
        runtime,
        "run",
        "--rm",
        "--interactive",
        "--pull=never",
        f"--cidfile={cidfile}",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        f"--memory={memory_limit_mb}m",
        f"--memory-swap={memory_limit_mb}m",
        f"--cpus={cpus:g}",
        f"--pids-limit={pids_limit}",
        "--ulimit=nofile=256:256",
        "--stop-timeout=1",
        f"--tmpfs=/tmp:rw,noexec,nosuid,size={tmpfs_size_mb}m",
        f"--user={_container_user()}",
        "--env=PYTHONDONTWRITEBYTECODE=1",
        "--env=PYTHONHASHSEED=0",
        f"--mount={_mount_spec(code_directory, '/auditcoder', read_only=True)}",
    ]
    if workspace is None:
        command.append("--workdir=/tmp")
    else:
        workspace_target = str(workspace)
        command.extend(
            [
                f"--mount={_mount_spec(workspace, workspace_target, read_only=not writable_workspace)}",
                f"--workdir={workspace_target}",
            ]
        )
    command.extend([image, "/auditcoder/program.py"])
    return command


class SandboxExecutor:
    """Execute Python code using the configured host or container backend.

    ``AUDITCODER_EXECUTION_BACKEND=container`` enables Docker/Podman isolation.
    The default remains ``host`` so offline tests do not silently depend on a
    local container daemon.  Callers must treat the host backend as unsafe for
    untrusted generated programs.
    """

    def __init__(
        self,
        default_timeout_s: float = 5.0,
        default_memory_limit_mb: int = 256,
        *,
        backend: str | None = None,
        container_runtime: str | None = None,
        container_image: str | None = None,
    ) -> None:
        self.default_timeout_s = default_timeout_s
        self.default_memory_limit_mb = default_memory_limit_mb
        self.backend = (
            backend
            or os.getenv("AUDITCODER_EXECUTION_BACKEND", "host")
        ).strip().lower()
        if self.backend not in {"host", "container"}:
            raise ValueError(
                "AUDITCODER_EXECUTION_BACKEND must be 'host' or 'container'"
            )

        self.container_runtime = (
            container_runtime
            or os.getenv("AUDITCODER_CONTAINER_RUNTIME", "docker")
        ).strip()
        if Path(self.container_runtime).name not in {"docker", "podman"}:
            raise ValueError(
                "AUDITCODER_CONTAINER_RUNTIME must resolve to docker or podman"
            )
        self.container_image = (
            container_image
            or os.getenv("AUDITCODER_CONTAINER_IMAGE", "auditcoder-eval:aaai27")
        ).strip()
        if not self.container_image:
            raise ValueError("AUDITCODER_CONTAINER_IMAGE must not be empty")

        self.container_cpus = float(
            _positive_number(
                os.getenv("AUDITCODER_CONTAINER_CPUS", "2"),
                name="AUDITCODER_CONTAINER_CPUS",
            )
        )
        self.container_pids_limit = int(
            _positive_number(
                os.getenv("AUDITCODER_CONTAINER_PIDS_LIMIT", "256"),
                name="AUDITCODER_CONTAINER_PIDS_LIMIT",
                integer=True,
            )
        )
        self.container_tmpfs_size_mb = int(
            _positive_number(
                os.getenv("AUDITCODER_CONTAINER_TMPFS_MB", "256"),
                name="AUDITCODER_CONTAINER_TMPFS_MB",
                integer=True,
            )
        )

    def run(
        self,
        code: str,
        stdin: str = "",
        timeout_s: float | None = None,
        memory_limit_mb: int | None = None,
        *,
        workspace: str | os.PathLike[str] | None = None,
        writable_workspace: bool = False,
    ) -> ExecutionResult:
        timeout = timeout_s if timeout_s is not None else self.default_timeout_s
        mem_limit = memory_limit_mb if memory_limit_mb is not None else self.default_memory_limit_mb
        if timeout <= 0:
            raise ValueError("timeout_s must be positive")
        if mem_limit <= 0:
            raise ValueError("memory_limit_mb must be positive")

        workspace_path: Path | None = None
        if workspace is not None:
            workspace_path = Path(workspace).expanduser().resolve(strict=True)
            if not workspace_path.is_dir():
                raise ValueError("workspace must be an existing directory")

        if self.backend == "container":
            return self._run_container(
                code=code,
                stdin=stdin,
                timeout=timeout,
                memory_limit_mb=mem_limit,
                workspace=workspace_path,
                writable_workspace=writable_workspace,
            )
        return self._run_host(
            code=code,
            stdin=stdin,
            timeout=timeout,
            memory_limit_mb=mem_limit,
            workspace=workspace_path,
        )

    def _run_host(
        self,
        *,
        code: str,
        stdin: str,
        timeout: float,
        memory_limit_mb: int,
        workspace: Path | None,
    ) -> ExecutionResult:
        with tempfile.TemporaryDirectory(prefix="auditcoder-host-") as temp_dir:
            code_path = Path(temp_dir) / "program.py"
            wrapper_path = Path(temp_dir) / "wrapper.py"
            code_path.write_text(code, encoding="utf-8")
            wrapper_path.write_text(
                _build_wrapper(str(code_path), memory_limit_mb),
                encoding="utf-8",
            )

            result = ExecutionResult()
            try:
                start = time.perf_counter()
                proc = subprocess.run(
                    [sys.executable, str(wrapper_path)],
                    input=stdin,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=workspace,
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

            return result

    def _run_container(
        self,
        *,
        code: str,
        stdin: str,
        timeout: float,
        memory_limit_mb: int,
        workspace: Path | None,
        writable_workspace: bool,
    ) -> ExecutionResult:
        runtime = shutil.which(self.container_runtime)
        if runtime is None:
            return ExecutionResult(
                stderr=(
                    f"container runtime not found: {self.container_runtime}; "
                    "install Docker or Podman, or select the host backend explicitly"
                ),
                exit_code=127,
            )

        with tempfile.TemporaryDirectory(prefix="auditcoder-container-") as temp_dir:
            code_directory = Path(temp_dir).resolve()
            code_path = code_directory / "program.py"
            cidfile = code_directory / "container.cid"
            code_path.write_text(code, encoding="utf-8")
            command = _build_container_command(
                runtime=runtime,
                image=self.container_image,
                code_directory=code_directory,
                cidfile=cidfile,
                memory_limit_mb=memory_limit_mb,
                cpus=self.container_cpus,
                pids_limit=self.container_pids_limit,
                tmpfs_size_mb=self.container_tmpfs_size_mb,
                workspace=workspace,
                writable_workspace=writable_workspace,
            )

            result = ExecutionResult()
            try:
                start = time.perf_counter()
                proc = subprocess.run(
                    command,
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
                self._remove_timed_out_container(runtime, cidfile)
            return result

    @staticmethod
    def _remove_timed_out_container(runtime: str, cidfile: Path) -> None:
        try:
            container_id = cidfile.read_text(encoding="utf-8").strip()
        except OSError:
            return
        if not container_id:
            return
        try:
            subprocess.run(
                [runtime, "rm", "--force", container_id],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass

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
