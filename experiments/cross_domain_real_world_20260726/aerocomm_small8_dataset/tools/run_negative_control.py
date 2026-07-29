#!/usr/bin/env python3
"""Confirm that trivial importable stubs do not resolve dataset tasks."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "tools/evaluate_candidate.py"


def main() -> int:
    manifest = json.loads((ROOT / "dataset_manifest.json").read_text())
    rows = []
    with tempfile.TemporaryDirectory(prefix="aerocomm-negative-") as temp:
        base = Path(temp)
        for task in manifest["tasks"]:
            code = base / f"{task['task_id']}.py"
            code.write_text(
                f"def {task['entry_point']}(*args, **kwargs):\n    return None\n"
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(EVALUATOR),
                    "--task",
                    task["task_id"],
                    "--code",
                    str(code),
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
                    "failure_class": result["failure_class"],
                }
            )
    report = {
        "dataset": manifest["dataset_name"],
        "negative_control_resolved_tasks": sum(row["resolved"] for row in rows),
        "task_count": len(rows),
        "acceptance_condition_met": not any(row["resolved"] for row in rows),
        "rows": rows,
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "negative_control.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["acceptance_condition_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
