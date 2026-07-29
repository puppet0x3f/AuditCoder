#!/usr/bin/env python3
"""Validate the frozen QF-Small6 dataset without importing AuditCoder."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TASKS = {
    "earnings-surprise-calculator",
    "corporate-action-adjustment",
    "historical-var-data-prep",
    "interest-rate-cap-floor",
    "zero-coupon-bootstrapping",
    "pca-factor-portfolio",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def static_test_count(path: Path) -> int:
    tree = ast.parse(path.read_text())
    return sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in tree.body
    )


def validate_checksums(errors: list[str]) -> int:
    checksum_path = ROOT / "UPSTREAM_SHA256SUMS"
    if not checksum_path.exists():
        errors.append("missing UPSTREAM_SHA256SUMS")
        return 0
    checked = 0
    for line in checksum_path.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"checksum target missing: {relative}")
            continue
        actual = sha256(path)
        if actual != expected:
            errors.append(
                f"checksum mismatch: {relative}: expected={expected} actual={actual}"
            )
        checked += 1
    return checked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    manifest_path = ROOT / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    tasks = manifest["tasks"]
    task_ids = {task["task_id"] for task in tasks}
    if task_ids != EXPECTED_TASKS:
        errors.append(
            f"task set mismatch: missing={sorted(EXPECTED_TASKS - task_ids)} "
            f"extra={sorted(task_ids - EXPECTED_TASKS)}"
        )
    if manifest["source"]["revision"] != "024921eb507fcc0c4ffe3e0a96802724be1ae84a":
        errors.append("unexpected source revision")
    if manifest["source"]["license"] != "CC BY-NC 4.0":
        errors.append("unexpected license")

    static_tests = 0
    public_files = 0
    private_files = 0
    for task in tasks:
        task_id = task["task_id"]
        base = ROOT / "tasks" / task_id
        required = [
            base / "public/problem.md",
            base / "public/data",
            base / "private/tests/test_outputs.py",
            base / "private/tests/test_contract_additions.py",
            base / "private/oracle/solve.sh",
            base / "metadata/task.toml",
            base / "metadata/Dockerfile",
            base / "metadata/upstream_test.sh",
        ]
        for path in required:
            if not path.exists():
                errors.append(f"missing required path: {path.relative_to(ROOT)}")
        if (base / "public/private").exists():
            errors.append(f"invalid public/private nesting: {task_id}")
        if list((base / "public").rglob("solve.sh")):
            errors.append(f"oracle leaked into public files: {task_id}")
        if list((base / "public").rglob("test_*.py")):
            errors.append(f"tests leaked into public files: {task_id}")

        official_count = static_test_count(base / "private/tests/test_outputs.py")
        addition_count = static_test_count(
            base / "private/tests/test_contract_additions.py"
        )
        if official_count != task["upstream_static_test_functions"]:
            errors.append(
                f"{task_id}: official static tests={official_count}, "
                f"manifest={task['upstream_static_test_functions']}"
            )
        if addition_count != task["contract_addition_test_functions"]:
            errors.append(
                f"{task_id}: addition static tests={addition_count}, "
                f"manifest={task['contract_addition_test_functions']}"
            )
        static_tests += official_count + addition_count
        public_files += sum(path.is_file() for path in (base / "public").rglob("*"))
        private_files += sum(path.is_file() for path in (base / "private").rglob("*"))

    checksums_checked = validate_checksums(errors)
    report = {
        "dataset": manifest["dataset_name"],
        "source_revision": manifest["source"]["revision"],
        "tasks": len(tasks),
        "public_files": public_files,
        "private_files": private_files,
        "static_test_functions": static_tests,
        "upstream_checksums_checked": checksums_checked,
        "valid": not errors,
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        output = args.report if args.report.is_absolute() else ROOT / args.report
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
