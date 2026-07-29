#!/usr/bin/env python3
"""Run the public AgentCoder protocol on SciCode Blind-Small-8."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


DATASET_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DATASET_ROOT.parents[2]
QF_ADAPTER = (
    REPO_ROOT
    / "experiments/cross_domain_real_world_20260726/qf_small6_dataset/tools"
    / "run_agentcoder_qf_small6.py"
)
MANIFEST_PATH = DATASET_ROOT / "dataset_manifest.json"
EVALUATOR = DATASET_ROOT / "tools/evaluate_candidate.py"
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


def reset_public_app(_task: dict[str, Any], app_path: Path) -> None:
    if app_path.exists():
        shutil.rmtree(app_path)
    app_path.mkdir(parents=True)


def build_public_problem(problem: str, _app_path: Path) -> str:
    return problem


def build_tester_prompt(problem: str, outputs: list[str]) -> str:
    entry_point = outputs[0]
    return f"""**Role**: You are an independent software test designer.

Create ONE self-contained Python unit-test module for the scientific function
below. Import `{entry_point}` using:

```python
from candidate import {entry_point}
```

Use only the public specification to derive expected values or scientific
invariants. Include at least one meaningful `assert`. Do not copy an
implementation of the function, access files/network/private targets, use
`target`, HDF5, frame inspection, or execute shell commands. Return only one
complete Python code block. You cannot see any programmer candidate.

**Public task**:

{problem}
"""


def tester_static_check(code: str, outputs: list[str]) -> dict[str, Any]:
    entry_point = outputs[0]
    result = {
        "syntax_valid": False,
        "assert_count": 0,
        "imports_candidate": False,
        "mentions_entry_point": entry_point in code,
        "leakage_scan": {
            "violation": bool(
                re.search(
                    r"(?:targets\.h5|/private/|\bh5py\b|\btarget\b|"
                    r"inspect\.(?:currentframe|stack)|sys\._getframe)",
                    code,
                    re.IGNORECASE,
                )
            )
        },
        "valid": False,
    }
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        result["syntax_error"] = str(exc)
        return result
    result["syntax_valid"] = True
    result["assert_count"] = sum(
        isinstance(node, ast.Assert)
        or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr.startswith("assert")
        )
        for node in ast.walk(tree)
    )
    result["imports_candidate"] = any(
        (
            isinstance(node, ast.Import)
            and any(alias.name == "candidate" for alias in node.names)
        )
        or (isinstance(node, ast.ImportFrom) and node.module == "candidate")
        for node in ast.walk(tree)
    )
    result["valid"] = bool(
        result["assert_count"] > 0
        and result["imports_candidate"]
        and result["mentions_entry_point"]
        and not result["leakage_scan"]["violation"]
    )
    return result


def execute_pair(
    task: dict[str, Any],
    *,
    visible_app_path: Path,
    candidate_index: int,
    tester_index: int,
    candidate_code: str,
    tester_code: str,
    tester_check: dict[str, Any],
    program_timeout: float,
    test_timeout: float,
) -> dict[str, Any]:
    del visible_app_path, program_timeout
    if not candidate_code.strip():
        return {
            "candidate_index": candidate_index,
            "tester_index": tester_index,
            "passed": False,
            "reason": "empty_candidate",
        }
    if not tester_check["valid"]:
        return {
            "candidate_index": candidate_index,
            "tester_index": tester_index,
            "passed": False,
            "reason": "invalid_generated_test",
            "tester_static_check": tester_check,
        }
    try:
        ast.parse(candidate_code)
    except SyntaxError as exc:
        return {
            "candidate_index": candidate_index,
            "tester_index": tester_index,
            "passed": False,
            "reason": "candidate_syntax_error",
            "syntax_error": str(exc),
        }
    with tempfile.TemporaryDirectory(prefix="agentcoder-scicode-pair-") as temp:
        workdir = Path(temp)
        (workdir / "candidate.py").write_text(candidate_code)
        (workdir / "agent_test.py").write_text(tester_code)
        (workdir / "sitecustomize.py").write_text(
            "import scipy.integrate as _si\n"
            "if not hasattr(_si, 'simps'):\n"
            "    _si.simps = _si.simpson\n"
        )
        runtime_env = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(workdir),
        }
        started = time.monotonic()
        try:
            completed = subprocess.run(
                [sys.executable, str(workdir / "agent_test.py")],
                cwd=workdir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=test_timeout,
                env=runtime_env,
                check=False,
            )
            execution = {
                "returncode": completed.returncode,
                "timeout": False,
                "duration_seconds": round(time.monotonic() - started, 6),
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            }
        except subprocess.TimeoutExpired as exc:
            execution = {
                "returncode": None,
                "timeout": True,
                "duration_seconds": round(time.monotonic() - started, 6),
                "stdout": (exc.stdout or "")[-4000:],
                "stderr": (exc.stderr or "")[-4000:],
            }
        passed = execution["returncode"] == 0 and not execution["timeout"]
        return {
            "candidate_index": candidate_index,
            "tester_index": tester_index,
            "passed": passed,
            "reason": "generated_test_passed" if passed else "generated_test_failed",
            "tester_execution": execution,
        }


def call_evaluator(
    task: dict[str, Any], task_run_dir: Path, code: str
) -> dict[str, Any]:
    code_path = task_run_dir / "final_code.py"
    code_path.write_text(code)
    eval_python = Path(os.getenv("SCICODE_EVAL_PYTHON", str(DEFAULT_EVAL_PYTHON)))
    completed = subprocess.run(
        [
            str(eval_python),
            str(EVALUATOR),
            "--task",
            task["task_id"],
            "--code",
            str(code_path),
            "--timeout",
            "180",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=900,
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
    resolved = sum(item["external_evaluation"]["task_resolved"] for item in completed)
    tests_passed = sum(
        item["external_evaluation"]["official_tests_passed"] for item in completed
    )
    build_passed = sum(
        item["external_evaluation"]["build_import_success"] for item in completed
    )
    env_failures = sum(
        item["external_evaluation"]["environment_failure"] for item in completed
    )
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
        "resolved_rate": resolved / len(config["task_ids"]),
        "official_tests_passed": tests_passed,
        "official_tests_total": 3 * len(config["task_ids"]),
        "build_import_successes": build_passed,
        "environment_failures": env_failures,
        "token_usage": usage,
        "results": results,
    }
    base.write_json(run_dir / "summary.json", summary)
    base.write_json(run_dir / "llm_calls.json", llm.records)
    lines = [
        f"# SciCode Blind-Small-8 AgentCoder: {config['run_id']}",
        "",
        f"- Model: `{config['model']['name']}`",
        f"- Resolved@1: `{resolved}/{len(config['task_ids'])}`",
        f"- Official tests: `{tests_passed}/{3 * len(config['task_ids'])}`",
        f"- Build/import: `{build_passed}/{len(config['task_ids'])}`",
        f"- Environment failures: `{env_failures}`",
        f"- LLM calls / tokens: `{usage['calls']}` / `{usage['total_tokens']}`",
        "",
        "| Task | Resolved | Tests | Generated-test score | Build/import | Failure | Tokens |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for item in results:
        if item.get("error"):
            lines.append(
                f"| `{item['task_id']}` | ERROR | 0/3 | - | - | "
                f"`{item['error']['type']}` | - |"
            )
            continue
        evaluation = item["external_evaluation"]
        protocol = item["agentcoder"]
        lines.append(
            f"| `{item['task_id']}` | "
            f"{'PASS' if evaluation['task_resolved'] else 'FAIL'} | "
            f"{evaluation['official_tests_passed']}/3 | "
            f"{protocol['selected_generated_test_score']}/10 | "
            f"{'PASS' if evaluation['build_import_success'] else 'FAIL'} | "
            f"{evaluation['failure_class'] or ''} | "
            f"{item['token_usage']['total_tokens']} |"
        )
    (run_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    base.write_checksums(run_dir)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-pools", type=int, default=6)
    parser.add_argument("--executor-workers", type=int, default=5)
    parser.add_argument("--program-timeout", type=float, default=180)
    parser.add_argument("--test-timeout", type=float, default=120)
    args = parser.parse_args()
    if not 1 <= args.max_pools <= 6:
        raise SystemExit("--max-pools must be between 1 and 6")

    load_dotenv(REPO_ROOT / ".env", override=True)
    manifest = json.loads(MANIFEST_PATH.read_text())
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    selected = args.task or list(tasks)
    unknown = set(selected) - set(tasks)
    if unknown:
        raise SystemExit(f"unknown task(s): {sorted(unknown)}")

    base = load_module("scicode_agentcoder_qf_adapter", QF_ADAPTER)
    base.DATASET_ROOT = DATASET_ROOT
    base.MANIFEST_PATH = MANIFEST_PATH
    base.reset_public_app = reset_public_app
    base.build_public_problem = build_public_problem
    base.build_tester_prompt = build_tester_prompt
    base.tester_static_check = tester_static_check
    base.execute_pair = execute_pair
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
    run_dir = DATASET_ROOT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_id": args.run_id,
        "created_at": utc_now(),
        "dataset": {
            "name": manifest["dataset_name"],
            "version": manifest["dataset_version"],
            "revision": manifest["source"]["dataset_revision"],
            "manifest_sha256": base.sha256_file(MANIFEST_PATH),
            "target_subset_sha256": manifest["artifacts"][
                "target_subset_sha256"
            ],
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
            "functional_judge": "3 frozen official SciCode numeric tests",
            "prompt_setting": "with scientist-provided background",
        },
        "agentcoder_source": {
            **base.source_snapshot(base.AGENTCODER_ROOT),
            **base.git_snapshot(base.AGENTCODER_ROOT),
        },
        "external_evaluator": {
            "path": str(EVALUATOR),
            "sha256": base.sha256_file(EVALUATOR),
            "compat_sha256": base.sha256_file(
                DATASET_ROOT / "tools/scicode_compat.py"
            ),
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
                f"tests={evaluation['official_tests_passed']}/3",
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
