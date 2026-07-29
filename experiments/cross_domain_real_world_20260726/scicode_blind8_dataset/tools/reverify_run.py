#!/usr/bin/env python3
"""Deterministically re-evaluate every frozen final candidate in a run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "tools/evaluate_candidate.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stored = json.loads((args.run / "summary.json").read_text())
    rows = []
    for old in stored["results"]:
        task_id = old["task_id"]
        code = args.run / "tasks" / task_id / "final_code.py"
        completed = subprocess.run(
            [
                sys.executable,
                str(EVALUATOR),
                "--task",
                task_id,
                "--code",
                str(code),
                "--timeout",
                "180",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=900,
            check=False,
        )
        current = json.loads(completed.stdout)
        previous = old["external_evaluation"]
        rows.append(
            {
                "task_id": task_id,
                "candidate_sha256": current["candidate_sha256"],
                "stored_resolved": previous["task_resolved"],
                "reverified_resolved": current["task_resolved"],
                "stored_tests_passed": previous["official_tests_passed"],
                "reverified_tests_passed": current["official_tests_passed"],
                "verdict_match": (
                    previous["task_resolved"] == current["task_resolved"]
                    and previous["official_tests_passed"]
                    == current["official_tests_passed"]
                ),
                "failure_class": current["failure_class"],
            }
        )
    report = {
        "run_id": stored["run_id"],
        "reverified_at": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(rows),
        "all_verdicts_match": all(row["verdict_match"] for row in rows),
        "resolved_tasks": sum(row["reverified_resolved"] for row in rows),
        "official_tests_passed": sum(
            row["reverified_tests_passed"] for row in rows
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["all_verdicts_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
