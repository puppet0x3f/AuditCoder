#!/usr/bin/env python3
"""Evaluate one generated module against frozen official SciCode tests."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


DATASET_ROOT = Path(__file__).resolve().parents[1]
TARGETS_PATH = DATASET_ROOT / "private/targets.h5"
TOOLS_ROOT = DATASET_ROOT / "tools"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def truncate(value: str, limit: int = 8000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n... [truncated {len(value) - limit} chars]"


def run(command: list[str], *, cwd: Path, timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(TOOLS_ROOT),
            },
        )
        return {
            "returncode": completed.returncode,
            "timeout": False,
            "duration_seconds": round(time.monotonic() - started, 6),
            "stdout": truncate(completed.stdout),
            "stderr": truncate(completed.stderr),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "timeout": True,
            "duration_seconds": round(time.monotonic() - started, 6),
            "stdout": truncate(exc.stdout or ""),
            "stderr": truncate(exc.stderr or ""),
        }


def compatibility_preamble() -> str:
    return """import scipy.integrate as _scipy_integrate
if not hasattr(_scipy_integrate, "simps"):
    _scipy_integrate.simps = _scipy_integrate.simpson
"""


def dependency_probe(spec: dict[str, Any], workdir: Path) -> dict[str, Any]:
    probe = workdir / "dependency_probe.py"
    probe.write_text(compatibility_preamble() + "\n" + spec["required_dependencies"])
    return run([sys.executable, str(probe)], cwd=workdir, timeout=60)


def leakage_scan(code: str) -> dict[str, Any]:
    patterns = {
        "private_artifact": r"(?:targets\.h5|task_spec\.json|/private/)",
        "target_global": r"\btarget\b",
        "frame_introspection": r"(?:inspect\.(?:currentframe|stack)|sys\._getframe)",
        "hdf5_access": r"\bh5py\b",
    }
    matches = {
        name: bool(re.search(pattern, code, re.IGNORECASE))
        for name, pattern in patterns.items()
    }
    return {"violation": any(matches.values()), "patterns": matches}


def render_test(task_id: str, test_index: int, test_case: str) -> str:
    adjusted = test_case.replace(
        "from scicode.compare.cmp import cmp_tuple_or_list",
        "from scicode_compat import cmp_tuple_or_list",
    )
    return f"""{compatibility_preamble()}
import numpy as np
from candidate import *
from scicode_compat import process_hdf5_to_tuple

target = process_hdf5_to_tuple({task_id!r}, {test_index + 1}, {str(TARGETS_PATH)!r})[{test_index}]

{adjusted}
"""


def evaluate(task_id: str, code_path: Path, timeout: float) -> dict[str, Any]:
    spec_path = DATASET_ROOT / "tasks" / task_id / "private/task_spec.json"
    if not spec_path.is_file():
        raise ValueError(f"unknown task: {task_id}")
    spec = json.loads(spec_path.read_text())
    code = code_path.read_text()
    result: dict[str, Any] = {
        "task_id": task_id,
        "candidate_sha256": sha256_text(code),
        "entry_point": spec["entry_point"],
        "official_test_count": spec["official_test_count"],
        "leakage_scan": leakage_scan(code),
    }
    try:
        tree = ast.parse(code)
        result["syntax_valid"] = True
        result["entry_point_present"] = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == spec["entry_point"]
            for node in tree.body
        )
    except SyntaxError as exc:
        result.update(
            {
                "syntax_valid": False,
                "entry_point_present": False,
                "syntax_error": str(exc),
                "tests": [],
                "official_tests_passed": 0,
                "build_import_success": False,
                "environment_failure": False,
                "task_resolved": False,
                "failure_class": "candidate_syntax_error",
            }
        )
        return result

    with tempfile.TemporaryDirectory(prefix=f"scicode-{task_id}-") as temp:
        workdir = Path(temp)
        candidate = workdir / "candidate.py"
        candidate.write_text(code)
        dependency = dependency_probe(spec, workdir)
        result["dependency_probe"] = dependency
        if dependency["returncode"] != 0 or dependency["timeout"]:
            result.update(
                {
                    "tests": [],
                    "official_tests_passed": 0,
                    "build_import_success": False,
                    "environment_failure": True,
                    "task_resolved": False,
                    "failure_class": "environment_failure",
                }
            )
            return result

        compile_result = run(
            [sys.executable, "-m", "py_compile", str(candidate)],
            cwd=workdir,
            timeout=60,
        )
        result["compile"] = compile_result
        import_probe = workdir / "import_probe.py"
        import_probe.write_text(
            compatibility_preamble()
            + f"\nimport candidate\nassert hasattr(candidate, {spec['entry_point']!r})\n"
        )
        import_result = run(
            [sys.executable, str(import_probe)], cwd=workdir, timeout=60
        )
        result["import_probe"] = import_result
        result["build_import_success"] = bool(
            compile_result["returncode"] == 0
            and not compile_result["timeout"]
            and import_result["returncode"] == 0
            and not import_result["timeout"]
            and result["entry_point_present"]
        )

        tests = []
        for index, test_case in enumerate(spec["official_test_cases"]):
            test_path = workdir / f"official_test_{index + 1}.py"
            test_path.write_text(render_test(task_id, index, test_case))
            execution = run(
                [sys.executable, str(test_path)], cwd=workdir, timeout=timeout
            )
            tests.append(
                {
                    "test_index": index + 1,
                    "passed": execution["returncode"] == 0
                    and not execution["timeout"],
                    "execution": execution,
                }
            )
        passed = sum(test["passed"] for test in tests)
        resolved = bool(
            result["build_import_success"]
            and passed == spec["official_test_count"]
            and not result["leakage_scan"]["violation"]
        )
        result.update(
            {
                "tests": tests,
                "official_tests_passed": passed,
                "environment_failure": False,
                "task_resolved": resolved,
                "failure_class": (
                    None
                    if resolved
                    else (
                        "leakage_violation"
                        if result["leakage_scan"]["violation"]
                        else (
                            "build_import_failure"
                            if not result["build_import_success"]
                            else (
                                "timeout"
                                if any(test["execution"]["timeout"] for test in tests)
                                else "wrong_answer_or_runtime_error"
                            )
                        )
                    )
                ),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.task, args.code, args.timeout)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["task_resolved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
