#!/usr/bin/env python3
"""Run the unchanged AuditCoder pipeline on AeroComm-Small8."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
QF_ADAPTER = (
    REPO_ROOT
    / "experiments/cross_domain_real_world_20260726/qf_small6_dataset/tools"
    / "run_auditcoder_qf_small6.py"
)
MANIFEST = ROOT / "dataset_manifest.json"
EVALUATOR = ROOT / "tools/evaluate_candidate.py"
DEFAULT_EVAL_PYTHON = Path(sys.executable).resolve()
DEFAULT_RUN_ID = "auditcoder_deepseek_v4_flash_20260729"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reset_public_app(_task: dict[str, Any], app_path: Path) -> None:
    if app_path.exists():
        import shutil

        shutil.rmtree(app_path)
    app_path.mkdir(parents=True)


def build_problem_text(
    public_problem: str,
    _app_path: Path,
    _required_inputs: list[str],
    _required_outputs: list[str],
) -> str:
    return (
        """# AuditCoder AeroComm library-task protocol

This is a short function implementation task, not a stdin/stdout contest
program. The final answer must be one complete importable Python module.
Preserve the exact function name and signature. Do not add stdin parsing,
demonstration calls, file I/O, network access, or private-test access.

AuditCoder's internal verifier can only check the generated module without the hidden
numeric assertions. The frozen hidden tests are applied exactly once after the
pipeline returns its final code and are never fed back into AuditCoder.

"""
        + public_problem.strip()
        + "\n"
    )


def call_evaluator(
    task: dict[str, Any], task_run_dir: Path, code: str
) -> dict[str, Any]:
    code_path = task_run_dir / "final_code.py"
    code_path.write_text(code)
    eval_python = Path(
        os.getenv("AEROCOMM_EVAL_PYTHON", str(DEFAULT_EVAL_PYTHON))
    )
    completed = subprocess.run(
        [
            str(eval_python),
            str(EVALUATOR),
            "--task",
            task["task_id"],
            "--code",
            str(code_path),
            "--timeout",
            "90",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
        check=False,
    )
    if not completed.stdout.strip():
        raise RuntimeError(
            f"external evaluator emitted no JSON: rc={completed.returncode}; "
            f"stderr={completed.stderr[-2000:]}"
        )
    result = json.loads(completed.stdout)
    result["evaluator_returncode"] = completed.returncode
    result["evaluator_stderr"] = completed.stderr
    return result


def summarize(
    run_dir: Path,
    config: dict[str, Any],
    results: list[dict[str, Any]],
    base: Any,
) -> dict[str, Any]:
    completed = [item for item in results if not item.get("error")]
    resolved = sum(
        item["external_evaluation"]["task_resolved"] for item in completed
    )
    passed = sum(
        item["external_evaluation"]["official_tests_passed"]
        for item in completed
    )
    total = sum(
        item["external_evaluation"]["official_test_count"] for item in completed
    )
    builds = sum(
        item["external_evaluation"]["build_import_success"] for item in completed
    )
    env_failures = sum(
        item["external_evaluation"]["environment_failure"] for item in completed
    )
    calls = sum(
        item["token_usage"]["total"].get("calls", 0) for item in completed
    )
    tokens = sum(
        item["token_usage"]["total"].get("total_tokens", 0) for item in completed
    )
    domain = {}
    for domain_name in ("aerospace", "communications"):
        prefix = "AC-AERO" if domain_name == "aerospace" else "AC-COMM"
        items = [item for item in completed if item["task_id"].startswith(prefix)]
        domain[domain_name] = {
            "resolved": sum(
                item["external_evaluation"]["task_resolved"] for item in items
            ),
            "tasks": len(items),
        }
    summary = {
        "run_id": config["run_id"],
        "generated_at": utc_now(),
        "method": "AuditCoder full",
        "model": config["model"],
        "dataset": config["dataset"],
        "tasks_requested": len(config["task_ids"]),
        "tasks_completed": len(completed),
        "pipeline_errors": len(results) - len(completed),
        "resolved_tasks": resolved,
        "resolved_rate": (
            resolved / len(config["task_ids"]) if config["task_ids"] else None
        ),
        "domain_results": domain,
        "official_tests_passed": passed,
        "official_tests_total": total,
        "build_import_successes": builds,
        "environment_failures": env_failures,
        "token_usage": {"calls": calls, "total_tokens": tokens},
        "total_pipeline_seconds": round(
            sum(item["pipeline_seconds"] for item in completed), 6
        ),
        "results": results,
    }
    base.write_json(run_dir / "summary.json", summary)
    lines = [
        f"# AeroComm-Small8 AuditCoder: {config['run_id']}",
        "",
        f"- Model: `{config['model']['name']}`",
        f"- Resolved@1: `{resolved}/{len(config['task_ids'])}`",
        f"- Aerospace: `{domain['aerospace']['resolved']}/{domain['aerospace']['tasks']}`",
        f"- Communications: `{domain['communications']['resolved']}/{domain['communications']['tasks']}`",
        f"- Hidden tests: `{passed}/{total}`",
        f"- Build/import: `{builds}/{len(completed)}`",
        f"- Environment failures: `{env_failures}`",
        f"- LLM calls / tokens: `{calls}` / `{tokens}`",
        "",
        "| Task | Resolved | Tests | Build/import | Failure | LOC | Tokens |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    for item in results:
        if item.get("error"):
            lines.append(
                f"| `{item['task_id']}` | ERROR | - | - | "
                f"`{item['error']['type']}` | - | - |"
            )
            continue
        evaluation = item["external_evaluation"]
        lines.append(
            f"| `{item['task_id']}` | "
            f"{'PASS' if evaluation['task_resolved'] else 'FAIL'} | "
            f"{evaluation['official_tests_passed']}/"
            f"{evaluation['official_test_count']} | "
            f"{'PASS' if evaluation['build_import_success'] else 'FAIL'} | "
            f"{evaluation['failure_class'] or ''} | "
            f"{item['final_code_nonblank_loc']} | "
            f"{item['token_usage']['total'].get('total_tokens', 0)} |"
        )
    lines.extend(
        [
            "",
            "The frozen external tests determine functional correctness. AuditCoder's "
            "internal verifier result is retained only as process evidence.",
            "",
        ]
    )
    (run_dir / "SUMMARY.md").write_text("\n".join(lines))
    base.write_checksums(run_dir)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env", override=True)
    manifest = json.loads(MANIFEST.read_text())
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    selected = args.task or list(tasks)
    unknown = set(selected) - set(tasks)
    if unknown:
        raise SystemExit(f"unknown task(s): {sorted(unknown)}")

    base = load_module("aerocomm_auditcoder_base", QF_ADAPTER)
    base.DATASET_ROOT = ROOT
    base.MANIFEST_PATH = MANIFEST
    base.reset_public_app = reset_public_app
    base.task_required_inputs = lambda _task: []
    base.build_problem_text = build_problem_text
    base.evaluate_candidate = call_evaluator

    run_dir = ROOT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_id": args.run_id,
        "created_at": utc_now(),
        "dataset": {
            "name": manifest["dataset_name"],
            "version": manifest["dataset_version"],
            "revision": manifest["source"]["dataset_revision"],
            **base.dataset_snapshot(),
        },
        "task_ids": selected,
        "model": {
            "provider": "deepseek",
            "name": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            "temperature": 0,
            "api_key_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
        },
        "protocol": {
            "attempts_per_task": 1,
            "hidden_test_feedback": False,
            "functional_judge": "all frozen AeroComm hidden pytest cases",
            "candidate_contract": "importable Python module with specified function",
        },
        "runtime": {
            "pipeline_python": sys.executable,
            "evaluator_python": os.getenv(
                "AEROCOMM_EVAL_PYTHON", str(DEFAULT_EVAL_PYTHON)
            ),
        },
        "external_evaluator": {
            "path": str(EVALUATOR),
            "sha256": base.sha256_file(EVALUATOR),
        },
        "source_snapshot": base.source_snapshot(),
    }
    base.write_json(run_dir / "run_config.json", config)
    audit_module = base.load_module(
        "aerocomm_auditcoder_audit_runner", base.AUDIT_RUNNER_PATH
    )
    subtree_module = audit_module._load_subtree_replan()

    results = []
    for task_id in selected:
        result_path = run_dir / "tasks" / task_id / "result.json"
        if result_path.is_file() and not args.force:
            print(f"{task_id}: resume existing result", flush=True)
            results.append(json.loads(result_path.read_text()))
            continue
        print(f"{task_id}: AuditCoder pipeline start", flush=True)
        try:
            result = base.run_one_task(
                tasks[task_id],
                run_dir,
                audit_module=audit_module,
                subtree_module=subtree_module,
            )
            evaluation = result["external_evaluation"]
            print(
                f"{task_id}: resolved={evaluation['task_resolved']} "
                f"tests={evaluation['official_tests_passed']}/"
                f"{evaluation['official_test_count']}",
                flush=True,
            )
        except Exception as exc:
            result = {
                "task_id": task_id,
                "timestamp": utc_now(),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
            print(f"{task_id}: ERROR {type(exc).__name__}: {exc}", flush=True)
        base.write_json(result_path, result)
        results.append(result)
        summarize(run_dir, config, results, base)

    summary = summarize(run_dir, config, results, base)
    print(
        f"summary: resolved={summary['resolved_tasks']}/"
        f"{summary['tasks_requested']} report={run_dir / 'SUMMARY.md'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
