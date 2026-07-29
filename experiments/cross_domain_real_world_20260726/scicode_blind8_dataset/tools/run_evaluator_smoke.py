#!/usr/bin/env python3
"""Run independent clear-spec implementations through the frozen evaluator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "tools/evaluate_candidate.py"
ORACLES = ROOT / "private/oracle_smoke"
TASKS = ["13.1", "14.1", "25.1", "35.1", "45.1", "77.1"]


def main() -> int:
    results = []
    for task_id in TASKS:
        completed = subprocess.run(
            [
                sys.executable,
                str(EVALUATOR),
                "--task",
                task_id,
                "--code",
                str(ORACLES / f"{task_id}.py"),
                "--timeout",
                "180",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=900,
            check=False,
        )
        result = json.loads(completed.stdout)
        results.append(
            {
                "task_id": task_id,
                "returncode": completed.returncode,
                "resolved": result["task_resolved"],
                "tests_passed": result["official_tests_passed"],
                "failure_class": result["failure_class"],
            }
        )
    report = {
        "purpose": (
            "evaluator smoke validation on six tasks with unambiguous "
            "independently implemented specifications; not model results"
        ),
        "tasks": len(TASKS),
        "resolved": sum(item["resolved"] for item in results),
        "tests_passed": sum(item["tests_passed"] for item in results),
        "tests_total": len(TASKS) * 3,
        "results": results,
    }
    path = ROOT / "reports/evaluator_smoke.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["resolved"] == len(TASKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
