#!/usr/bin/env python3
"""Portable evaluator for SWE-bench Verified Micro-5 patcher programs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "dataset_manifest.json"
DEFAULT_PYTHON = Path("/opt/anaconda3/envs/swe-micro5/bin/python")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def truncate(value: str, limit: int = 12000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n... [truncated {len(value) - limit} chars]"


def run(
    command: list[str],
    *,
    cwd: Path,
    timeout: float,
    env: dict[str, str] | None = None,
    stdin: str | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            input=stdin,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "timeout": False,
            "duration_seconds": round(time.monotonic() - started, 6),
            "stdout": truncate(completed.stdout),
            "stderr": truncate(completed.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "returncode": None,
            "timeout": True,
            "duration_seconds": round(time.monotonic() - started, 6),
            "stdout": truncate(exc.stdout or ""),
            "stderr": truncate(exc.stderr or ""),
        }


def clean_test_env(repo: Path, pythonpath: str) -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        env.pop(key, None)
    env.update(
        {
            "PYTHONPATH": pythonpath,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "LANG": "en_US.UTF-8",
            "LC_ALL": "C",
        }
    )
    return env


def normalize_test_ids(task: dict[str, Any], ids: list[str]) -> list[str]:
    if task["test_kind"] == "django":
        normalized = []
        for value in ids:
            match = re.fullmatch(r"(\w+) \(([^)]+)\)", value)
            if not match:
                raise RuntimeError(f"unexpected Django test id: {value}")
            normalized.append(f"{match.group(2)}.{match.group(1)}")
        return normalized
    if task["test_kind"] == "pytest-bare-names":
        return [f"{task['test_file']}::{value}" for value in ids]
    return ids


def write_pytest_version(repo: Path) -> None:
    destination = repo / "src/_pytest/_version.py"
    destination.write_text(
        "# generated portable-evaluation compatibility file\n"
        "__version__ = version = '7.2.0'\n"
        "__version_tuple__ = version_tuple = (7, 2, 0)\n"
        "__commit_id__ = commit_id = None\n"
    )


def run_group(
    *,
    task: dict[str, Any],
    repo: Path,
    python: Path,
    ids: list[str],
    group: str,
    timeout: float,
) -> dict[str, Any]:
    normalized = normalize_test_ids(task, ids)
    if task["repo"] == "pytest-dev/pytest":
        write_pytest_version(repo)
        pythonpath = str(repo / "src")
    else:
        pythonpath = str(repo)
    env = clean_test_env(repo, pythonpath)
    portable_scope = "exact official test identifiers"
    if task["test_kind"] == "django":
        command = [
            str(python),
            "tests/runtests.py",
            *normalized,
            "--verbosity",
            "1",
            "--settings=test_sqlite",
            "--parallel",
            "1",
        ]
    elif task["repo"] == "psf/requests" and group == "PASS_TO_PASS":
        # Several parameterized Requests P2P identifiers in the frozen
        # dataset are display strings rather than addressable pytest node IDs.
        # Run the complete target file while deselecting the two F2P nodes.
        # This is a conservative superset of the official P2P group.
        command = [
            str(python),
            "-m",
            "pytest",
            "-q",
            task["test_file"],
            *[
                f"--deselect={value}"
                for value in task["fail_to_pass"]
            ],
            "--disable-warnings",
        ]
        portable_scope = "target-file superset with F2P nodes deselected"
    else:
        command = [
            str(python),
            "-m",
            "pytest",
            "-q",
            *normalized,
            "--disable-warnings",
        ]
    result = run(command, cwd=repo, timeout=timeout, env=env)
    result.update(
        {
            "group": group,
            "official_test_count": len(ids),
            "portable_test_scope": portable_scope,
            "all_non_failing": result["returncode"] == 0
            and not result["timeout"],
        }
    )
    return result


def leakage_scan(code: str) -> dict[str, Any]:
    patterns = {
        "private_artifact": r"(?:gold_patch|test_patch|FAIL_TO_PASS|PASS_TO_PASS)",
        "private_path": r"(?:^|[/'\"]+)private(?:[/'\"]+)",
        "network": r"(?:https?://|urllib\.request|requests\.(?:get|post)|socket\.)",
        "git_invocation": r"(?:subprocess|os\.system).*?\bgit\b",
        "test_modification": r"(?:/tests?/|test_[A-Za-z0-9_]+\.py)",
    }
    matches = {
        name: bool(re.search(pattern, code, re.I | re.S))
        for name, pattern in patterns.items()
    }
    return {"violation": any(matches.values()), "matches": matches}


def status_paths(repo: Path) -> list[dict[str, str]]:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z"],
        cwd=repo,
        stdout=subprocess.PIPE,
        check=True,
    )
    values = completed.stdout.decode(errors="replace").split("\0")
    rows = []
    for value in values:
        if not value:
            continue
        rows.append({"status": value[:2], "path": value[3:]})
    return rows


def evaluate(
    *,
    task_id: str,
    repo_cache_root: Path,
    python: Path,
    patcher_code: str | None,
    model_patch: str | None,
    timeout: float,
    allow_empty_patch: bool,
) -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text())
    public = {item["task_id"]: item for item in manifest["tasks"]}[task_id]
    task = json.loads(
        (ROOT / "tasks" / task_id / "private/upstream.json").read_text()
    )
    cache = repo_cache_root / task["repo"].replace("/", "__")
    if not (cache / ".git").is_dir():
        raise RuntimeError(f"repository cache missing: {cache}")

    with tempfile.TemporaryDirectory(prefix=f"swe-micro5-{task_id}-") as temp:
        temp_root = Path(temp)
        repo = temp_root / "repo"
        clone = run(
            ["git", "clone", "-q", "--shared", str(cache), str(repo)],
            cwd=temp_root,
            timeout=180,
        )
        if clone["returncode"] != 0:
            return {
                "task_id": task_id,
                "task_resolved": False,
                "environment_failure": True,
                "failure_class": "repository_clone_failure",
                "clone": clone,
            }
        checkout = run(
            ["git", "checkout", "-q", "--detach", task["base_commit"]],
            cwd=repo,
            timeout=60,
        )
        if checkout["returncode"] != 0:
            return {
                "task_id": task_id,
                "task_resolved": False,
                "environment_failure": True,
                "failure_class": "base_checkout_failure",
                "checkout": checkout,
            }

        candidate_execution: dict[str, Any] | None = None
        leakage = {"violation": False, "matches": {}}
        if patcher_code is not None:
            leakage = leakage_scan(patcher_code)
            try:
                ast.parse(patcher_code)
                syntax_valid = True
                syntax_error = None
            except SyntaxError as exc:
                syntax_valid = False
                syntax_error = str(exc)
            if not syntax_valid or leakage["violation"]:
                return {
                    "task_id": task_id,
                    "task_resolved": False,
                    "environment_failure": False,
                    "patch_build_success": False,
                    "failure_class": (
                        "patcher_syntax_error"
                        if not syntax_valid
                        else "candidate_leakage_violation"
                    ),
                    "syntax_error": syntax_error,
                    "leakage_scan": leakage,
                }
            patcher = temp_root / "candidate_patcher.py"
            patcher.write_text(patcher_code)
            env = clean_test_env(repo, str(repo))
            env["REPO_ROOT"] = str(repo)
            candidate_execution = run(
                [str(python), str(patcher)],
                cwd=repo,
                timeout=timeout,
                env=env,
            )
            if (
                candidate_execution["returncode"] != 0
                or candidate_execution["timeout"]
            ):
                return {
                    "task_id": task_id,
                    "task_resolved": False,
                    "environment_failure": False,
                    "patch_build_success": False,
                    "failure_class": "patcher_execution_failure",
                    "candidate_execution": candidate_execution,
                    "leakage_scan": leakage,
                }
        elif model_patch is not None:
            if model_patch.strip():
                apply_model = run(
                    ["git", "apply", "--whitespace=nowarn", "-"],
                    cwd=repo,
                    timeout=60,
                    stdin=model_patch,
                )
                if apply_model["returncode"] != 0:
                    return {
                        "task_id": task_id,
                        "task_resolved": False,
                        "environment_failure": False,
                        "patch_build_success": False,
                        "failure_class": "model_patch_apply_failure",
                        "patch_application": apply_model,
                    }
            elif not allow_empty_patch:
                return {
                    "task_id": task_id,
                    "task_resolved": False,
                    "environment_failure": False,
                    "patch_build_success": False,
                    "failure_class": "empty_model_patch",
                }
        else:
            raise ValueError("patcher_code or model_patch is required")

        changed = status_paths(repo)
        allowed = set(public["oracle_files"])
        changed_paths = {item["path"] for item in changed}
        allowed_paths_only = changed_paths <= allowed
        diff_result = run(
            ["git", "diff", "--binary", "--", *public["oracle_files"]],
            cwd=repo,
            timeout=30,
        )
        generated_patch = diff_result["stdout"]
        nonempty_patch = bool(generated_patch.strip())
        patch_build_success = bool(
            allowed_paths_only and (nonempty_patch or allow_empty_patch)
        )
        if not patch_build_success:
            return {
                "task_id": task_id,
                "task_resolved": False,
                "environment_failure": False,
                "patch_build_success": False,
                "failure_class": (
                    "disallowed_repository_change"
                    if not allowed_paths_only
                    else "empty_model_patch"
                ),
                "changed_paths": changed,
                "allowed_files": sorted(allowed),
                "generated_patch_sha256": sha256_text(generated_patch),
                "generated_patch": generated_patch,
                "candidate_execution": candidate_execution,
                "leakage_scan": leakage,
            }

        test_patch_apply = run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=repo,
            timeout=60,
            stdin=task["test_patch"],
        )
        if test_patch_apply["returncode"] != 0:
            return {
                "task_id": task_id,
                "task_resolved": False,
                "environment_failure": True,
                "patch_build_success": patch_build_success,
                "failure_class": "test_patch_apply_failure",
                "test_patch_application": test_patch_apply,
            }

        f2p = run_group(
            task=task,
            repo=repo,
            python=python,
            ids=task["fail_to_pass"],
            group="FAIL_TO_PASS",
            timeout=timeout,
        )
        p2p = run_group(
            task=task,
            repo=repo,
            python=python,
            ids=task["pass_to_pass"],
            group="PASS_TO_PASS",
            timeout=timeout,
        )
        resolved = (
            patch_build_success
            and f2p["all_non_failing"]
            and p2p["all_non_failing"]
        )
        failure_class = None
        if not resolved:
            if not f2p["all_non_failing"]:
                failure_class = "fail_to_pass_failure"
            elif not p2p["all_non_failing"]:
                failure_class = "regression_failure"
        return {
            "task_id": task_id,
            "repo": task["repo"],
            "base_commit": task["base_commit"],
            "portable_evaluation": True,
            "official_docker_evaluation": False,
            "environment_failure": False,
            "patch_build_success": patch_build_success,
            "allowed_paths_only": allowed_paths_only,
            "changed_paths": changed,
            "generated_patch": generated_patch,
            "generated_patch_sha256": sha256_text(generated_patch),
            "generated_patch_nonblank_lines": sum(
                bool(line.strip()) for line in generated_patch.splitlines()
            ),
            "candidate_execution": candidate_execution,
            "leakage_scan": leakage,
            "fail_to_pass": f2p,
            "pass_to_pass": p2p,
            "task_resolved": resolved,
            "failure_class": failure_class,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--patcher", type=Path)
    source.add_argument("--model-patch", type=Path)
    parser.add_argument("--repo-cache-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--allow-empty-patch", action="store_true")
    args = parser.parse_args()
    patcher_code = args.patcher.read_text() if args.patcher else None
    model_patch = args.model_patch.read_text() if args.model_patch else None
    result = evaluate(
        task_id=args.task,
        repo_cache_root=args.repo_cache_root,
        python=args.python,
        patcher_code=patcher_code,
        model_patch=model_patch,
        timeout=args.timeout,
        allow_empty_patch=args.allow_empty_patch,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("task_resolved") else 1


if __name__ == "__main__":
    raise SystemExit(main())
