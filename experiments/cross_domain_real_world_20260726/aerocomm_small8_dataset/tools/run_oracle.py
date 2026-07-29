#!/usr/bin/env python3
"""Evaluate all frozen reference implementations with the external judge."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "tools/evaluate_candidate.py"


def main() -> int:
    manifest = json.loads((ROOT / "dataset_manifest.json").read_text())
    rows = []
    for task in manifest["tasks"]:
        completed = subprocess.run(
            [
                sys.executable,
                str(EVALUATOR),
                "--task",
                task["task_id"],
                "--code",
                str(ROOT / task["private_oracle"]),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=180,
        )
        result = json.loads(completed.stdout)
        rows.append(
            {
                "task_id": task["task_id"],
                "resolved": result["task_resolved"],
                "passed": result["official_tests_passed"],
                "total": result["official_test_count"],
                "candidate_sha256": result["candidate_sha256"],
                "failure_class": result["failure_class"],
            }
        )
    report = {
        "dataset": manifest["dataset_name"],
        "all_oracles_pass": all(row["resolved"] for row in rows),
        "resolved_tasks": sum(row["resolved"] for row in rows),
        "task_count": len(rows),
        "tests_passed": sum(row["passed"] for row in rows),
        "tests_total": sum(row["total"] for row in rows),
        "rows": rows,
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "oracle_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_oracles_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
