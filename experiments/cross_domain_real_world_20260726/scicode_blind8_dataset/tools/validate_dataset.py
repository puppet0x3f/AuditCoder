#!/usr/bin/env python3
"""Validate public/private separation and frozen SciCode target completeness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IDS = ["13.1", "14.1", "25.1", "28.1", "35.1", "37.1", "45.1", "77.1"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads((ROOT / "dataset_manifest.json").read_text())
    errors = []
    ids = [task["task_id"] for task in manifest["tasks"]]
    if ids != EXPECTED_IDS:
        errors.append(f"task order/selection mismatch: {ids}")
    if manifest["task_count"] != 8 or manifest["official_test_count"] != 24:
        errors.append("manifest task/test totals must be 8/24")

    public_hashes = {}
    for task_id in EXPECTED_IDS:
        public_dir = ROOT / "tasks" / task_id / "public"
        private_spec = ROOT / "tasks" / task_id / "private/task_spec.json"
        for path in (public_dir / "problem.md", public_dir / "task.json"):
            if not path.is_file():
                errors.append(f"missing public artifact: {path}")
                continue
            text = path.read_text()
            if "test_cases" in text or "official_test_cases" in text:
                errors.append(f"test leakage in public artifact: {path}")
            public_hashes[str(path.relative_to(ROOT))] = sha256_file(path)
        if not private_spec.is_file():
            errors.append(f"missing private spec: {private_spec}")
            continue
        spec = json.loads(private_spec.read_text())
        if len(spec["official_test_cases"]) != 3:
            errors.append(f"{task_id} must have 3 official tests")

    target_path = ROOT / "private/targets.h5"
    target_counts = {}
    with h5py.File(target_path, "r") as handle:
        groups = list(handle.keys())
        if groups != sorted(EXPECTED_IDS):
            errors.append(f"target groups mismatch: {groups}")
        for task_id in EXPECTED_IDS:
            count = len(handle[task_id].keys()) if task_id in handle else 0
            target_counts[task_id] = count
            if count != 3:
                errors.append(f"{task_id} target count is {count}, expected 3")

    report = {
        "dataset": manifest["dataset_name"],
        "dataset_version": manifest["dataset_version"],
        "valid": not errors,
        "errors": errors,
        "task_count": len(ids),
        "official_test_count": sum(target_counts.values()),
        "target_counts": target_counts,
        "target_subset_sha256": sha256_file(target_path),
        "public_artifact_hashes": public_hashes,
        "public_private_separation": not any("leakage" in error for error in errors),
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
