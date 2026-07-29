#!/usr/bin/env python3
"""Run base negative controls and gold-patch positive controls."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "tools/evaluate_patch_script.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("micro5_evaluator", EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-cache-root", type=Path, required=True)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path("/opt/anaconda3/envs/swe-micro5/bin/python"),
    )
    args = parser.parse_args()
    evaluator = load_evaluator()
    manifest = json.loads((ROOT / "dataset_manifest.json").read_text())
    rows = []
    valid = True
    for task in manifest["tasks"]:
        task_id = task["task_id"]
        private = json.loads(
            (ROOT / "tasks" / task_id / "private/upstream.json").read_text()
        )
        base = evaluator.evaluate(
            task_id=task_id,
            repo_cache_root=args.repo_cache_root,
            python=args.python,
            patcher_code=None,
            model_patch="",
            timeout=300,
            allow_empty_patch=True,
        )
        gold = evaluator.evaluate(
            task_id=task_id,
            repo_cache_root=args.repo_cache_root,
            python=args.python,
            patcher_code=None,
            model_patch=private["gold_patch"],
            timeout=300,
            allow_empty_patch=False,
        )
        expected_base = (
            not base.get("fail_to_pass", {}).get("all_non_failing", True)
            and base.get("pass_to_pass", {}).get("all_non_failing", False)
        )
        row_valid = expected_base and gold.get("task_resolved", False)
        valid = valid and row_valid
        rows.append(
            {
                "task_id": task_id,
                "base_negative_control": base,
                "gold_positive_control": gold,
                "base_f2p_fails_and_p2p_passes": expected_base,
                "oracle_valid": row_valid,
            }
        )
        print(
            f"{task_id}: base_control={expected_base} "
            f"gold={gold.get('task_resolved')} valid={row_valid}",
            flush=True,
        )
    report = {
        "dataset": manifest["dataset_name"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "portable_environment": str(args.python),
        "official_docker_evaluation": False,
        "valid": valid,
        "tasks": rows,
    }
    destination = ROOT / "reports/portable_oracle.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(destination)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
