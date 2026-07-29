#!/usr/bin/env python3
"""Validate public/private separation and frozen Micro-5 artifacts."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    errors = []
    manifest = json.loads((ROOT / "dataset_manifest.json").read_text())
    tasks = manifest.get("tasks", [])
    if len(tasks) != 5:
        errors.append(f"expected 5 tasks, found {len(tasks)}")
    if sum(task["official_test_count"] for task in tasks) != 261:
        errors.append("official test count must be 261")
    for task in tasks:
        task_id = task["task_id"]
        public_dir = ROOT / "tasks" / task_id / "public"
        private = ROOT / "tasks" / task_id / "private/upstream.json"
        for path in (public_dir / "problem.md", public_dir / "task.json", private):
            if not path.is_file():
                errors.append(f"missing {path.relative_to(ROOT)}")
        if private.is_file():
            hidden = json.loads(private.read_text())
            for field in ("gold_patch", "test_patch", "fail_to_pass", "pass_to_pass"):
                if not hidden.get(field):
                    errors.append(f"{task_id}: missing private {field}")
        problem = public_dir / "problem.md"
        if problem.is_file():
            text = problem.read_text()
            for forbidden in ("diff --git ", "FAIL_TO_PASS", "PASS_TO_PASS"):
                if forbidden in text:
                    errors.append(f"{task_id}: public leakage marker {forbidden}")
        for relative, expected in manifest["public_artifact_sha256"].items():
            path = ROOT / relative
            if path.is_file() and sha256_file(path) != expected:
                errors.append(f"public hash mismatch: {relative}")

    for path in sorted((ROOT / "tools").glob("*.py")):
        try:
            ast.parse(path.read_text())
        except SyntaxError as exc:
            errors.append(f"{path.name}: {exc}")

    result = {
        "dataset": manifest["dataset_name"],
        "valid": not errors,
        "task_count": len(tasks),
        "official_test_count": sum(
            task["official_test_count"] for task in tasks
        ),
        "public_private_separation": not any(
            "leakage" in error for error in errors
        ),
        "errors": errors,
    }
    report = ROOT / "reports/dataset_validation.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
