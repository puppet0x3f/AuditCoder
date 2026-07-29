#!/usr/bin/env python3
"""Run AuditCoder on QF-Small6 with an explicit file-execution contract.

The adapter changes the runtime `/app` prefix in the public prompt so that
AuditCoder's subprocess verifier can access a task-specific local sandbox. Hidden
tests and oracle files are never included in the model prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


DATASET_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DATASET_ROOT.parents[2]
MANIFEST_PATH = DATASET_ROOT / "dataset_manifest.json"
AUDIT_RUNNER_PATH = REPO_ROOT / "tests/run_apps_audit_metrics.py"
SUBTREE_RUNNER_PATH = REPO_ROOT / "tests/run_apps_subtree_replan.py"
DEFAULT_RUN_ID = "auditcoder_verifier_contract_fix_deepseek_v4_flash_20260728"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    )


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_snapshot() -> dict[str, Any]:
    paths = sorted((REPO_ROOT / "auditcoder").rglob("*.py"))
    paths.extend([AUDIT_RUNNER_PATH, SUBTREE_RUNNER_PATH, Path(__file__).resolve()])
    digest = hashlib.sha256()
    file_hashes = {}
    for path in paths:
        relative = path.relative_to(REPO_ROOT).as_posix()
        file_hash = sha256_file(path)
        file_hashes[relative] = file_hash
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")

    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return completed.stdout.strip()

    status = git("status", "--porcelain=v1")
    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status_line_count": len(status.splitlines()) if status else 0,
        "git_status_sha256": sha256_text(status),
        "source_tree_sha256": digest.hexdigest(),
        "source_file_count": len(paths),
        "source_file_hashes": file_hashes,
    }


def dataset_snapshot() -> dict[str, Any]:
    """Hash the exact prompts, fixtures, oracle artifacts, and frozen tests."""
    paths = [MANIFEST_PATH, DATASET_ROOT / "UPSTREAM_SHA256SUMS"]
    paths.extend(sorted((DATASET_ROOT / "tasks").rglob("*")))
    files = [path for path in paths if path.is_file()]
    digest = hashlib.sha256()
    file_hashes: dict[str, str] = {}
    for path in files:
        relative = path.relative_to(DATASET_ROOT).as_posix()
        file_hash = sha256_file(path)
        file_hashes[relative] = file_hash
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return {
        "dataset_tree_sha256": digest.hexdigest(),
        "dataset_file_count": len(files),
        "dataset_file_hashes": file_hashes,
    }


def write_checksums(run_dir: Path) -> None:
    checksum_path = run_dir / "CHECKSUMS.sha256"
    lines = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path == checksum_path:
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(run_dir).as_posix()}")
    checksum_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def build_problem_text(
    public_problem: str,
    app_path: Path,
    required_inputs: list[str],
    required_outputs: list[str],
) -> str:
    adapted = public_problem.replace("/app", str(app_path))
    execution_contract = {
        "schema_version": "1.0",
        "io_mode": "file",
        "explicit": True,
        "app_dir": str(app_path),
        "required_inputs": required_inputs,
        "required_outputs": required_outputs,
    }
    rendered_contract = (
        "<AUDITCODER_EXECUTION_CONTRACT>\n"
        + json.dumps(execution_contract, ensure_ascii=False, indent=2)
        + "\n</AUDITCODER_EXECUTION_CONTRACT>"
    )
    protocol = f"""

# AuditCoder QF-Small6 Execution Protocol

This is a file-processing program, not a stdin/stdout programming-contest task.
The complete generated Python source will be executed once with no stdin.
It MUST read the input files at the absolute paths stated below, create the
required output directory and files, and finish with exit code 0.

The runtime application directory for this task is:
`{app_path}`

If a generic assembler instruction mentions parsing stdin or printing a return
value, this task's explicit file input/output contract takes precedence.
Include an `if __name__ == "__main__":` entry point that performs the complete
calculation. Do not embed input data or expected outputs in the solution.
""".strip()
    return f"{rendered_contract}\n\n{protocol}\n\n{adapted.strip()}\n"


def reset_public_app(task: dict[str, Any], app_path: Path) -> None:
    if app_path.exists():
        shutil.rmtree(app_path)
    app_path.mkdir(parents=True)
    source = DATASET_ROOT / "tasks" / task["task_id"] / "public/data"
    mount = app_path / task["data_mount"]
    mount.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, mount, dirs_exist_ok=True)


def task_required_inputs(task: dict[str, Any]) -> list[str]:
    source = DATASET_ROOT / "tasks" / task["task_id"] / "public/data"
    mount = Path(task["data_mount"])
    return [
        (mount / path.relative_to(source)).as_posix()
        for path in sorted(source.rglob("*"))
        if path.is_file()
    ]


def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout: float,
    stdin_data: str | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        run_kwargs: dict[str, Any] = {}
        if stdin_data is not None:
            run_kwargs["input"] = stdin_data
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            **run_kwargs,
        )
        return {
            "returncode": completed.returncode,
            "timeout": False,
            "duration_seconds": round(time.monotonic() - started, 6),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "timeout": True,
            "duration_seconds": round(time.monotonic() - started, 6),
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }


def parse_junit(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "collected": 0,
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "skipped": 0,
            "junit_present": False,
        }
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    collected = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failed = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    return {
        "collected": collected,
        "passed": collected - failed - errors - skipped,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "junit_present": True,
    }


def materialize_test(source: Path, destination: Path, app_path: Path) -> None:
    destination.write_text(source.read_text().replace("/app", str(app_path)))


def run_pytest(
    test_file: Path,
    junit_path: Path,
    *,
    cwd: Path,
) -> dict[str, Any]:
    result = run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(test_file),
            f"--junitxml={junit_path}",
            "--disable-warnings",
        ],
        cwd=cwd,
        timeout=300,
    )
    result.update(parse_junit(junit_path))
    result["test_sha256"] = sha256_file(test_file)
    return result


def leakage_scan(code: str) -> dict[str, Any]:
    patterns = {
        "private_path": r"(?:^|[/'\"]+)private(?:[/'\"]+)",
        "upstream_test_file": r"test_outputs\.py",
        "contract_test_file": r"test_contract_additions\.py",
        "oracle_path": r"(?:^|[/'\"]+)oracle(?:[/'\"]+)",
    }
    matches = {
        name: bool(re.search(pattern, code, re.IGNORECASE | re.MULTILINE))
        for name, pattern in patterns.items()
    }
    return {
        "violation": any(matches.values()),
        "patterns": matches,
    }


def evaluate_candidate(
    task: dict[str, Any],
    task_run_dir: Path,
    code: str,
) -> dict[str, Any]:
    app_path = task_run_dir / "app"
    reset_public_app(task, app_path)
    code_path = task_run_dir / "final_code.py"
    code_path.write_text(code)
    execution = run_command(
        [sys.executable, str(code_path)],
        cwd=app_path,
        timeout=300,
        stdin_data="",
    )

    evaluation_dir = task_run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    task_source = DATASET_ROOT / "tasks" / task["task_id"] / "private/tests"
    official_test = evaluation_dir / "test_outputs.py"
    addition_test = evaluation_dir / "test_contract_additions.py"
    materialize_test(task_source / "test_outputs.py", official_test, app_path)
    materialize_test(
        task_source / "test_contract_additions.py", addition_test, app_path
    )
    official = run_pytest(
        official_test,
        evaluation_dir / "official-junit.xml",
        cwd=app_path,
    )
    additions = run_pytest(
        addition_test,
        evaluation_dir / "additions-junit.xml",
        cwd=app_path,
    )
    leakage = leakage_scan(code)
    required_outputs = {
        relative: (
            sha256_file(app_path / relative)
            if (app_path / relative).is_file()
            else None
        )
        for relative in task["outputs"]
    }
    resolved = (
        execution["returncode"] == 0
        and not execution["timeout"]
        and official["returncode"] == 0
        and additions["returncode"] == 0
        and not leakage["violation"]
    )
    return {
        "candidate_execution": execution,
        "upstream_tests": official,
        "contract_addition_tests": additions,
        "required_output_hashes": required_outputs,
        "leakage_scan": leakage,
        "task_resolved": resolved,
    }


def safe_pipeline_record(pipeline_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "ir": pipeline_result.get("ir", {}),
        "code_snippets": pipeline_result.get("code_snippets", {}),
        "verification_reports": pipeline_result.get("verification_reports", {}),
        "final_report": pipeline_result.get("final_report", {}),
        "execution_result": pipeline_result.get("execution_result", {}),
    }


def run_one_task(
    task: dict[str, Any],
    run_dir: Path,
    *,
    audit_module: Any,
    subtree_module: Any,
) -> dict[str, Any]:
    task_id = task["task_id"]
    task_run_dir = run_dir / "tasks" / task_id
    task_run_dir.mkdir(parents=True, exist_ok=True)
    app_path = task_run_dir / "app"
    reset_public_app(task, app_path)

    public_problem = (
        DATASET_ROOT / "tasks" / task_id / "public/problem.md"
    ).read_text()
    problem_text = build_problem_text(
        public_problem,
        app_path,
        task_required_inputs(task),
        task["outputs"],
    )
    (task_run_dir / "prompt.txt").write_text(problem_text)

    audit_module.TOKEN_TRACKER.reset()
    started = time.monotonic()
    with audit_module.TOKEN_TRACKER.scoped("auditcoder_pipeline"):
        pipeline_result = subtree_module.run_pipeline(
            problem_text,
            problem_id=None,
        )
    pipeline_seconds = round(time.monotonic() - started, 6)

    final_code = pipeline_result.get("final_code", "")
    (task_run_dir / "final_code.py").write_text(final_code)
    pipeline_record = safe_pipeline_record(pipeline_result)
    write_json(task_run_dir / "pipeline_artifacts.json", pipeline_record)

    dctc = audit_module.compute_dctc(
        pipeline_result.get("ir", {}),
        pipeline_result.get("code_snippets", {}),
        pipeline_result.get("verification_reports", {}),
        final_code,
    )
    evaluation = evaluate_candidate(task, task_run_dir, final_code)
    token_usage = audit_module.TOKEN_TRACKER.snapshot()
    return {
        "task_id": task_id,
        "timestamp": utc_now(),
        "dataset_revision": json.loads(MANIFEST_PATH.read_text())["source"][
            "revision"
        ],
        "prompt_sha256": sha256_text(problem_text),
        "final_code_sha256": sha256_text(final_code),
        "final_code_nonblank_loc": sum(
            bool(line.strip()) for line in final_code.splitlines()
        ),
        "pipeline_seconds": pipeline_seconds,
        "token_usage": token_usage,
        "dctc": dctc,
        "auditcoder_internal": {
            "final_report": pipeline_result.get("final_report", {}),
            "execution_result": pipeline_result.get("execution_result", {}),
            "note": (
                "The current AuditCoder internal verifier executes the file task but "
                "has no hidden QF assertions. Functional correctness is decided "
                "only by the external frozen tests below."
            ),
        },
        "external_evaluation": evaluation,
    }


def result_row(result: dict[str, Any]) -> str:
    if result.get("error"):
        return (
            f"| `{result['task_id']}` | ERROR | 0/0 | 0/0 | "
            f"- | - | `{result['error']['type']}` |"
        )
    evaluation = result["external_evaluation"]
    upstream = evaluation["upstream_tests"]
    additions = evaluation["contract_addition_tests"]
    internal = result["auditcoder_internal"]["final_report"]
    tokens = result["token_usage"]["total"].get("total_tokens", 0)
    return (
        f"| `{result['task_id']}` | "
        f"{'PASS' if evaluation['task_resolved'] else 'FAIL'} | "
        f"{upstream['passed']}/{upstream['collected']} | "
        f"{additions['passed']}/{additions['collected']} | "
        f"{'PASS' if internal.get('all_passed') else 'FAIL'} | "
        f"{tokens} |  |"
    )


def build_summary(
    run_dir: Path,
    config: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    completed = [result for result in results if not result.get("error")]
    resolved = sum(
        bool(result["external_evaluation"]["task_resolved"])
        for result in completed
    )
    pipeline_errors = sum(bool(result.get("error")) for result in results)
    official_passed = sum(
        result["external_evaluation"]["upstream_tests"]["passed"]
        for result in completed
    )
    official_collected = sum(
        result["external_evaluation"]["upstream_tests"]["collected"]
        for result in completed
    )
    additions_passed = sum(
        result["external_evaluation"]["contract_addition_tests"]["passed"]
        for result in completed
    )
    additions_collected = sum(
        result["external_evaluation"]["contract_addition_tests"]["collected"]
        for result in completed
    )
    total_tokens = sum(
        result["token_usage"]["total"].get("total_tokens", 0)
        for result in completed
    )
    total_calls = sum(
        result["token_usage"]["total"].get("calls", 0)
        for result in completed
    )
    summary = {
        "run_id": config["run_id"],
        "generated_at": utc_now(),
        "dataset": config["dataset"],
        "source_snapshot": config["source_snapshot"],
        "model": config["model"],
        "tasks_requested": len(config["task_ids"]),
        "tasks_with_pipeline_result": len(completed),
        "pipeline_errors": pipeline_errors,
        "resolved_tasks": resolved,
        "resolved_rate": (
            resolved / len(config["task_ids"]) if config["task_ids"] else None
        ),
        "upstream_tests": {
            "passed": official_passed,
            "collected": official_collected,
        },
        "contract_addition_tests": {
            "passed": additions_passed,
            "collected": additions_collected,
        },
        "token_usage": {
            "calls": total_calls,
            "total_tokens": total_tokens,
        },
        "results": results,
    }
    write_json(run_dir / "summary.json", summary)

    lines = [
        f"# QF-Small6 AuditCoder Run: {config['run_id']}",
        "",
        f"- Model: `{config['model']['name']}`",
        f"- Dataset revision: `{config['dataset']['revision']}`",
        f"- AuditCoder source tree SHA-256: `{config['source_snapshot']['source_tree_sha256']}`",
        f"- Resolved: `{resolved}/{len(config['task_ids'])}`",
        f"- Pipeline errors: `{pipeline_errors}`",
        f"- Upstream tests: `{official_passed}/{official_collected}`",
        f"- Contract additions: `{additions_passed}/{additions_collected}`",
        f"- LLM calls / tokens: `{total_calls}` / `{total_tokens}`",
        "",
        "| Task | External verdict | Upstream tests | Additions | AuditCoder internal execution | Tokens | Error |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    lines.extend(result_row(result) for result in results)
    lines.extend(
        [
            "",
            "AuditCoder internal execution is not the functional judge for these file tasks; "
            "the frozen external tests determine `task_resolved`.",
            "",
        ]
    )
    (run_dir / "SUMMARY.md").write_text("\n".join(lines))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task id to run; repeat for multiple tasks. Default: all six.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Re-evaluate existing generated code without invoking AuditCoder or the LLM.",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env", override=True)
    manifest = json.loads(MANIFEST_PATH.read_text())
    task_map = {task["task_id"]: task for task in manifest["tasks"]}
    selected_ids = args.task or [task["task_id"] for task in manifest["tasks"]]
    unknown = set(selected_ids) - set(task_map)
    if unknown:
        raise SystemExit(f"unknown task(s): {sorted(unknown)}")

    run_dir = DATASET_ROOT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_id": args.run_id,
        "created_at": utc_now(),
        "dataset": {
            "name": manifest["dataset_name"],
            "revision": manifest["source"]["revision"],
            **dataset_snapshot(),
        },
        "task_ids": selected_ids,
        "model": {
            "provider": "deepseek",
            "name": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            "max_tokens": os.getenv("DEEPSEEK_MAX_TOKENS")
            or os.getenv("MAX_TOKENS"),
            "api_key_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
        },
        "runtime": {
            "python": sys.executable,
            "python_version": sys.version,
            "pythonpath": os.getenv("PYTHONPATH", ""),
            "official_harbor_docker": False,
        },
        "protocol": {
            "attempts_per_task": 1,
            "hidden_test_feedback_to_auditcoder": False,
            "candidate_stdin": "closed EOF",
            "execution_contract": (
                "explicit file mode with app_dir and required_outputs"
            ),
            "public_prompt_path_rewrite": "/app -> task-specific absolute app path",
            "functional_judge": "upstream tests AND contract-addition tests",
        },
        "source_snapshot": source_snapshot(),
    }
    config_path = run_dir / "run_config.json"
    if args.evaluate_only and config_path.is_file():
        original_config = json.loads(config_path.read_text())
        if original_config.get("task_ids") != selected_ids:
            raise SystemExit(
                "evaluate-only task selection must match the original run config"
            )
        config = original_config
    else:
        write_json(config_path, config)

    if args.evaluate_only:
        results = []
        for task_id in selected_ids:
            result_path = run_dir / "tasks" / task_id / "result.json"
            code_path = run_dir / "tasks" / task_id / "final_code.py"
            if not result_path.is_file() or not code_path.is_file():
                raise SystemExit(f"missing existing result/code for {task_id}")
            result = json.loads(result_path.read_text())
            previous = result.get("external_evaluation", {})
            result.setdefault("external_evaluation_history", []).append(
                {
                    "superseded_at": utc_now(),
                    "reason": (
                        "deterministic independent re-evaluation of the same "
                        "generated source with stdin closed to EOF"
                    ),
                    "candidate_returncode": previous.get(
                        "candidate_execution", {}
                    ).get("returncode"),
                    "candidate_timeout": previous.get(
                        "candidate_execution", {}
                    ).get("timeout"),
                    "upstream_passed": previous.get("upstream_tests", {}).get(
                        "passed"
                    ),
                    "upstream_collected": previous.get("upstream_tests", {}).get(
                        "collected"
                    ),
                    "addition_passed": previous.get(
                        "contract_addition_tests", {}
                    ).get("passed"),
                    "addition_collected": previous.get(
                        "contract_addition_tests", {}
                    ).get("collected"),
                    "task_resolved": previous.get("task_resolved"),
                }
            )
            result["external_evaluation"] = evaluate_candidate(
                task_map[task_id],
                run_dir / "tasks" / task_id,
                code_path.read_text(),
            )
            result["external_evaluation_revision"] = {
                "evaluated_at": utc_now(),
                "protocol": (
                    "same generated source; public app reset; candidate stdin "
                    "closed to EOF; frozen tests rerun"
                ),
                "llm_or_auditcoder_rerun": False,
            }
            write_json(result_path, result)
            results.append(result)
            evaluation = result["external_evaluation"]
            print(
                f"{task_id}: re-evaluated resolved={evaluation['task_resolved']} "
                f"returncode={evaluation['candidate_execution']['returncode']} "
                f"timeout={evaluation['candidate_execution']['timeout']}",
                flush=True,
            )
        summary = build_summary(run_dir, config, results)
        write_checksums(run_dir)
        print(
            f"summary: resolved={summary['resolved_tasks']}/"
            f"{summary['tasks_requested']} pipeline_errors={summary['pipeline_errors']} "
            f"report={run_dir / 'SUMMARY.md'}",
            flush=True,
        )
        return 0

    audit_module = load_module("qf_auditcoder_audit_runner", AUDIT_RUNNER_PATH)
    subtree_module = audit_module._load_subtree_replan()

    results = []
    for task_id in selected_ids:
        result_path = run_dir / "tasks" / task_id / "result.json"
        if result_path.exists() and not args.force:
            result = json.loads(result_path.read_text())
            print(f"{task_id}: resume existing result")
            results.append(result)
            continue
        print(f"{task_id}: AuditCoder pipeline start", flush=True)
        try:
            result = run_one_task(
                task_map[task_id],
                run_dir,
                audit_module=audit_module,
                subtree_module=subtree_module,
            )
            evaluation = result["external_evaluation"]
            print(
                f"{task_id}: resolved={evaluation['task_resolved']} "
                f"upstream={evaluation['upstream_tests']['passed']}/"
                f"{evaluation['upstream_tests']['collected']} "
                f"additions={evaluation['contract_addition_tests']['passed']}/"
                f"{evaluation['contract_addition_tests']['collected']}",
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
        write_json(result_path, result)
        results.append(result)
        build_summary(run_dir, config, results)
        write_checksums(run_dir)

    summary = build_summary(run_dir, config, results)
    write_checksums(run_dir)
    print(
        f"summary: resolved={summary['resolved_tasks']}/"
        f"{summary['tasks_requested']} pipeline_errors={summary['pipeline_errors']} "
        f"report={run_dir / 'SUMMARY.md'}",
        flush=True,
    )
    return 0 if summary["pipeline_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
