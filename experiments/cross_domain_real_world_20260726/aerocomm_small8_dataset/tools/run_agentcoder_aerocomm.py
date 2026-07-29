#!/usr/bin/env python3
"""Run the AgentCoder protocol on AeroComm-Small8."""

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
    / "run_agentcoder_qf_small6.py"
)
SCICODE_ADAPTER = (
    REPO_ROOT
    / "experiments/cross_domain_real_world_20260726/scicode_blind8_dataset/tools"
    / "run_agentcoder_scicode_blind8.py"
)
MANIFEST = ROOT / "dataset_manifest.json"
EVALUATOR = ROOT / "tools/evaluate_candidate.py"
DEFAULT_EVAL_PYTHON = Path(sys.executable).resolve()
DEFAULT_RUN_ID = "agentcoder_deepseek_v4_flash_nothinking_20260729"


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
    llm: Any,
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
    domain = {}
    for domain_name in ("aerospace", "communications"):
        items = [
            item
            for item in completed
            if item["task_id"].startswith(
                "AC-AERO" if domain_name == "aerospace" else "AC-COMM"
            )
        ]
        domain[domain_name] = {
            "resolved": sum(
                item["external_evaluation"]["task_resolved"] for item in items
            ),
            "tasks": len(items),
        }
    usage = llm.usage_summary()
    summary = {
        "run_id": config["run_id"],
        "generated_at": utc_now(),
        "method": "AgentCoder",
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
        "token_usage": usage,
        "total_pipeline_seconds": round(
            sum(item["pipeline_seconds"] for item in completed), 6
        ),
        "results": results,
    }
    base.write_json(run_dir / "summary.json", summary)
    base.write_json(run_dir / "llm_calls.json", llm.records)
    lines = [
        f"# AeroComm-Small8 AgentCoder: {config['run_id']}",
        "",
        f"- Model: `{config['model']['name']}`",
        f"- Resolved@1: `{resolved}/{len(config['task_ids'])}`",
        f"- Aerospace: `{domain['aerospace']['resolved']}/{domain['aerospace']['tasks']}`",
        f"- Communications: `{domain['communications']['resolved']}/{domain['communications']['tasks']}`",
        f"- Hidden tests: `{passed}/{total}`",
        f"- Build/import: `{builds}/{len(completed)}`",
        f"- Environment failures: `{env_failures}`",
        f"- LLM calls / tokens: `{usage['calls']}` / `{usage['total_tokens']}`",
        "",
        "| Task | Resolved | Tests | Generated-test score | Build/import | Failure | LOC | Tokens |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for item in results:
        if item.get("error"):
            lines.append(
                f"| `{item['task_id']}` | ERROR | - | - | - | "
                f"`{item['error']['type']}` | - | - |"
            )
            continue
        evaluation = item["external_evaluation"]
        lines.append(
            f"| `{item['task_id']}` | "
            f"{'PASS' if evaluation['task_resolved'] else 'FAIL'} | "
            f"{evaluation['official_tests_passed']}/"
            f"{evaluation['official_test_count']} | "
            f"{item['agentcoder']['selected_generated_test_score']}/10 | "
            f"{'PASS' if evaluation['build_import_success'] else 'FAIL'} | "
            f"{evaluation['failure_class'] or ''} | "
            f"{item['final_code_nonblank_loc']} | "
            f"{item['token_usage']['total_tokens']} |"
        )
    lines.extend(
        [
            "",
            "The frozen external tests determine functional correctness. "
            "Generated tester programs are used only for AgentCoder candidate selection.",
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
    parser.add_argument("--max-pools", type=int, default=6)
    parser.add_argument("--executor-workers", type=int, default=5)
    parser.add_argument("--program-timeout", type=float, default=90)
    parser.add_argument("--test-timeout", type=float, default=60)
    args = parser.parse_args()
    if not 1 <= args.max_pools <= 6:
        raise SystemExit("--max-pools must be between 1 and 6")

    load_dotenv(REPO_ROOT / ".env", override=True)
    manifest = json.loads(MANIFEST.read_text())
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    selected = args.task or list(tasks)
    unknown = set(selected) - set(tasks)
    if unknown:
        raise SystemExit(f"unknown task(s): {sorted(unknown)}")

    base = load_module("aerocomm_agentcoder_base", QF_ADAPTER)
    helper = load_module("aerocomm_agentcoder_helper", SCICODE_ADAPTER)
    base.DATASET_ROOT = ROOT
    base.MANIFEST_PATH = MANIFEST
    base.reset_public_app = helper.reset_public_app
    base.build_public_problem = helper.build_public_problem
    base.build_tester_prompt = helper.build_tester_prompt
    base.tester_static_check = helper.tester_static_check
    base.execute_pair = helper.execute_pair
    base.evaluate_candidate = call_evaluator

    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not configured")
    llm = base.LLMClient(
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=int(
            os.getenv("DEEPSEEK_MAX_TOKENS")
            or os.getenv("MAX_TOKENS")
            or 8192
        ),
        temperature=0,
    )
    run_dir = ROOT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_id": args.run_id,
        "created_at": utc_now(),
        "dataset": {
            "name": manifest["dataset_name"],
            "version": manifest["dataset_version"],
            "revision": manifest["source"]["dataset_revision"],
            "manifest_sha256": base.sha256_file(MANIFEST),
            "official_test_count": manifest["official_test_count"],
        },
        "task_ids": selected,
        "model": {
            "provider": "deepseek",
            "name": model,
            "base_url": base_url,
            "temperature": 0,
            "reasoning_mode": "disabled",
        },
        "protocol": {
            "upstream_variant": "src/test_executor_humaneval.py",
            "programmer_samples_per_pool": base.PROGRAMMER_SAMPLES,
            "tester_samples_per_pool": base.TESTER_SAMPLES,
            "freeze_threshold": base.FREEZE_THRESHOLD,
            "max_pools": args.max_pools,
            "programmer_tester_independence": True,
            "hidden_test_feedback": False,
            "functional_judge": "all frozen AeroComm hidden pytest cases",
            "attempts_per_task": 1,
        },
        "agentcoder_source": {
            **base.source_snapshot(base.AGENTCODER_ROOT),
            **base.git_snapshot(base.AGENTCODER_ROOT),
        },
        "external_evaluator": {
            "path": str(EVALUATOR),
            "sha256": base.sha256_file(EVALUATOR),
        },
    }
    base.write_json(run_dir / "run_config.json", config)

    results = []
    for task_id in selected:
        result_path = run_dir / "tasks" / task_id / "result.json"
        if result_path.is_file() and not args.force:
            print(f"{task_id}: resume existing result", flush=True)
            results.append(json.loads(result_path.read_text()))
            continue
        print(f"{task_id}: AgentCoder start", flush=True)
        try:
            result = base.run_one_task(
                tasks[task_id],
                run_dir,
                llm=llm,
                max_pools=args.max_pools,
                executor_workers=args.executor_workers,
                program_timeout=args.program_timeout,
                test_timeout=args.test_timeout,
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
        summarize(run_dir, config, results, base, llm)

    summary = summarize(run_dir, config, results, base, llm)
    print(
        f"summary: resolved={summary['resolved_tasks']}/"
        f"{summary['tasks_requested']} report={run_dir / 'SUMMARY.md'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
