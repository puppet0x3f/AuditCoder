#!/usr/bin/env python3
"""Evaluate one AeroComm candidate against frozen private pytest cases."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "dataset_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def truncate(text: str, limit: int = 10000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def run(command: list[str], cwd: Path, timeout: float) -> dict[str, Any]:
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
                "PYTHONPATH": str(cwd),
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


def parse_junit(path: Path) -> dict[str, int]:
    if not path.is_file():
        return {"collected": 0, "passed": 0, "failed": 0, "errors": 1, "skipped": 0}
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    collected = sum(int(s.attrib.get("tests", 0)) for s in suites)
    failed = sum(int(s.attrib.get("failures", 0)) for s in suites)
    errors = sum(int(s.attrib.get("errors", 0)) for s in suites)
    skipped = sum(int(s.attrib.get("skipped", 0)) for s in suites)
    return {
        "collected": collected,
        "passed": collected - failed - errors - skipped,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
    }


def leakage_scan(code: str) -> dict[str, Any]:
    patterns = {
        "private_path": r"(?:/private/|private/test_official|private/oracle)",
        "hidden_test_name": r"\btest_official\.py\b",
        "frame_introspection": r"(?:inspect\.(?:currentframe|stack)|sys\._getframe)",
    }
    matches = {
        name: bool(re.search(pattern, code, re.IGNORECASE))
        for name, pattern in patterns.items()
    }
    return {"violation": any(matches.values()), "patterns": matches}


def evaluate(task_id: str, code_path: Path, timeout: float) -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text())
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    if task_id not in tasks:
        raise ValueError(f"unknown task: {task_id}")
    task = tasks[task_id]
    code = code_path.read_text()
    result: dict[str, Any] = {
        "task_id": task_id,
        "candidate_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "entry_point": task["entry_point"],
        "official_test_count": task["official_test_count"],
        "private_test_sha256": task["private_tests_sha256"],
        "leakage_scan": leakage_scan(code),
    }
    try:
        tree = ast.parse(code)
        result["syntax_valid"] = True
        result["entry_point_present"] = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == task["entry_point"]
            for node in tree.body
        )
    except SyntaxError as exc:
        result.update(
            {
                "syntax_valid": False,
                "entry_point_present": False,
                "syntax_error": str(exc),
                "official_tests_passed": 0,
                "build_import_success": False,
                "environment_failure": False,
                "task_resolved": False,
                "failure_class": "candidate_syntax_error",
            }
        )
        return result

    with tempfile.TemporaryDirectory(prefix=f"aerocomm-{task_id}-") as temp:
        workdir = Path(temp)
        candidate = workdir / "candidate.py"
        tests = workdir / "test_official.py"
        candidate.write_text(code)
        shutil.copy2(ROOT / task["private_tests"], tests)
        compile_result = run(
            [sys.executable, "-m", "py_compile", str(candidate)], workdir, 30
        )
        probe = workdir / "probe.py"
        probe.write_text(
            f"import candidate\nassert callable(getattr(candidate, {task['entry_point']!r}, None))\n"
        )
        import_result = run([sys.executable, str(probe)], workdir, 30)
        build_success = bool(
            compile_result["returncode"] == 0
            and not compile_result["timeout"]
            and import_result["returncode"] == 0
            and not import_result["timeout"]
            and result["entry_point_present"]
        )
        junit = workdir / "junit.xml"
        execution = run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                str(tests),
                f"--junitxml={junit}",
                "--disable-warnings",
            ],
            workdir,
            timeout,
        )
        counts = parse_junit(junit)
        environment_failure = bool(
            "No module named pytest" in execution["stderr"]
            or "No module named numpy" in execution["stderr"]
        )
        resolved = bool(
            build_success
            and counts["collected"] == task["official_test_count"]
            and counts["passed"] == task["official_test_count"]
            and not result["leakage_scan"]["violation"]
            and not environment_failure
        )
        result.update(
            {
                "compile": compile_result,
                "import_probe": import_result,
                "test_execution": execution,
                "test_counts": counts,
                "official_tests_passed": counts["passed"],
                "build_import_success": build_success,
                "environment_failure": environment_failure,
                "task_resolved": resolved,
                "failure_class": (
                    None
                    if resolved
                    else (
                        "environment_failure"
                        if environment_failure
                        else (
                            "leakage_violation"
                            if result["leakage_scan"]["violation"]
                            else (
                                "build_import_failure"
                                if not build_success
                                else (
                                    "timeout"
                                    if execution["timeout"]
                                    else "wrong_answer_or_runtime_error"
                                )
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
    parser.add_argument("--timeout", type=float, default=60)
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
