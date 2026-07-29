#!/usr/bin/env python3
"""Validate AeroComm-Small8 structure, hashes, and public/private isolation."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IDS = [
    "AC-AERO-01",
    "AC-AERO-02",
    "AC-AERO-03",
    "AC-AERO-04",
    "AC-COMM-01",
    "AC-COMM-02",
    "AC-COMM-03",
    "AC-COMM-04",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads((ROOT / "dataset_manifest.json").read_text())
    errors: list[str] = []
    ids = [task["task_id"] for task in manifest["tasks"]]
    if ids != EXPECTED_IDS:
        errors.append(f"task order mismatch: {ids}")
    if manifest["task_count"] != 8:
        errors.append("task_count must be 8")
    counted_tests = 0
    public_hashes = {}
    private_hashes = {}
    for task in manifest["tasks"]:
        problem = ROOT / task["public_problem"]
        metadata = ROOT / task["public_metadata"]
        tests = ROOT / task["private_tests"]
        oracle = ROOT / task["private_oracle"]
        for path in (problem, metadata, tests, oracle):
            if not path.is_file():
                errors.append(f"missing artifact: {path.relative_to(ROOT)}")
        if errors:
            continue
        public_text = problem.read_text() + metadata.read_text()
        if "private/test_official" in public_text or "private/oracle" in public_text:
            errors.append(f"private path leaked in {task['task_id']} public files")
        public_hashes[str(problem.relative_to(ROOT))] = sha256_file(problem)
        public_hashes[str(metadata.relative_to(ROOT))] = sha256_file(metadata)
        private_hashes[str(tests.relative_to(ROOT))] = sha256_file(tests)
        private_hashes[str(oracle.relative_to(ROOT))] = sha256_file(oracle)
        if sha256_file(problem) != task["public_problem_sha256"]:
            errors.append(f"problem hash mismatch: {task['task_id']}")
        if sha256_file(tests) != task["private_tests_sha256"]:
            errors.append(f"test hash mismatch: {task['task_id']}")
        if sha256_file(oracle) != task["private_oracle_sha256"]:
            errors.append(f"oracle hash mismatch: {task['task_id']}")
        tree = ast.parse(tests.read_text())
        test_count = sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
            for node in tree.body
        )
        counted_tests += test_count
        if test_count != task["official_test_count"]:
            errors.append(
                f"test count mismatch: {task['task_id']} "
                f"{test_count}!={task['official_test_count']}"
            )
    if counted_tests != manifest["official_test_count"]:
        errors.append(
            f"total tests mismatch: {counted_tests}!={manifest['official_test_count']}"
        )
    report = {
        "dataset": manifest["dataset_name"],
        "dataset_version": manifest["dataset_version"],
        "dataset_revision": manifest["source"]["dataset_revision"],
        "valid": not errors,
        "errors": errors,
        "task_count": len(ids),
        "official_test_count": counted_tests,
        "domain_counts": manifest["domains"],
        "public_private_separation": not any("leaked" in error for error in errors),
        "public_artifact_hashes": public_hashes,
        "private_artifact_hashes": private_hashes,
        "manifest_sha256": sha256_file(ROOT / "dataset_manifest.json"),
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "dataset_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
