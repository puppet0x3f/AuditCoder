"""Run the optimized AuditCoder pipeline on the 200 sampled APPS problems.

The script uses the implementation in the local ``auditcoder`` package:

    plan/CATG -> algorithm agents -> assembler -> bounded final verifier

The APPS ``input_output`` cases are supplied to the final verifier as trusted
benchmark checks and are then executed again independently against the final
code (including a code version accepted by bounded repair).  CATG's
model-generated development checks remain advisory diagnostics.  Results are
written one record at a time to a JSONL checkpoint so a long run can be
resumed after an interruption.

Examples::

    # Validate the dataset and selection without making model calls.
    python tests/run_apps_200_auditcoder.py --dry-run

    # Run all 200 problems, at most ten APPS cases per problem.
    python tests/run_apps_200_auditcoder.py --max-cases 10

    # Run every available APPS case and resume an interrupted run.
    python tests/run_apps_200_auditcoder.py \
        --max-cases 0 --resume

Environment variables are read from the repository-root ``.env`` when that
file exists.  The historical ``QWEN_API`` and ``DEEPSEEK_API`` names are
accepted and mapped to ``QWEN_API_KEY`` and ``DEEPSEEK_API_KEY``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve()
IMPLEMENTATION_ROOT = HERE.parents[1]
WORKSPACE_ROOT = IMPLEMENTATION_ROOT
DEFAULT_INPUT = WORKSPACE_ROOT / "data" / "apps_sampled_200.jsonl"
DEFAULT_OUTPUT = IMPLEMENTATION_ROOT / "results" / "apps_200_auditcoder.jsonl"

# Import the optimized package, never the sibling baseline checkout.
sys.path.insert(0, str(IMPLEMENTATION_ROOT))

from dotenv import load_dotenv  # noqa: E402

from auditcoder.algo_agents import dispatch_algo_agents  # noqa: E402
from auditcoder.assembler_agent import build_assembler_agent_graph  # noqa: E402
from auditcoder.common.callable_execution import (  # noqa: E402
    build_callable_driver,
    callable_arguments,
    callable_stdin,
)
from auditcoder.common.output_judging import judge_output  # noqa: E402
from auditcoder.final_verifier_agent import build_final_verifier_graph  # noqa: E402
from auditcoder.plan_agent.graph import build_plan_agent_graph  # noqa: E402
from auditcoder.schemas.contract_graph import catg_summary  # noqa: E402


def parse_io_field(raw: Any) -> dict[str, Any]:
    """Parse APPS' JSON-encoded ``input_output`` field."""

    if isinstance(raw, dict):
        return raw
    if not raw:
        return {"inputs": [], "outputs": []}
    try:
        parsed = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError):
        return {"inputs": [], "outputs": []}
    return parsed if isinstance(parsed, dict) else {"inputs": [], "outputs": []}


def format_case_text(value: Any, *, callable_case: bool = False) -> str:
    """Render legacy APPS argument-list cases as ordinary stdin text.

    A subset of the sampled APPS records stores cases as nested lists even
    though no ``fn_name`` is supplied, for example
    ``["501?1?", "6 5"]``.  ``str(value)`` produces Python list syntax and
    feeds invalid input to a normal stdin solution.  Flattening the case
    values with spaces preserves the intended two-line/token protocol while
    leaving true string cases unchanged.
    """
    if callable_case:
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return " ".join(format_case_text(part).strip() for part in value).strip()
    if value is None:
        return ""
    return str(value)


def build_external_test_cases(
    item: dict[str, Any], *, max_cases: int
) -> list[dict[str, Any]]:
    """Return benchmark-backed cases for the final verifier.

    These pairs come directly from APPS ``input_output`` and therefore have a
    dataset oracle.  They are passed to the verifier separately from CATG's
    model-generated development checks, whose expected outputs are useful for
    diagnostics but must not overturn a trusted benchmark result.
    """

    io_data = parse_io_field(item.get("input_output"))
    pairs = list(zip(io_data.get("inputs") or [], io_data.get("outputs") or []))
    if max_cases > 0:
        pairs = pairs[:max_cases]
    return [
        {
            "input": format_case_text(input_data, callable_case=bool(io_data.get("fn_name"))),
            "expected_output": format_case_text(expected_data, callable_case=bool(io_data.get("fn_name"))),
            "check_id": f"apps:{index}",
            "test_source": "dataset.input_output",
            "trusted": True,
            **(
                {
                    "callable_name": str(io_data.get("fn_name")),
                    "callable_args": callable_arguments(input_data),
                }
                if io_data.get("fn_name")
                else {}
            ),
        }
        for index, (input_data, expected_data) in enumerate(pairs)
    ]


def format_apps_problem(item: dict[str, Any], sample_count: int = 5) -> str:
    """Make a faithful stdin/stdout problem prompt for the planner."""

    io_data = parse_io_field(item.get("input_output"))
    inputs = list(io_data.get("inputs") or [])
    outputs = list(io_data.get("outputs") or [])
    lines = [
        f"【题目编号】{item.get('id', '')}",
        f"【原题链接】{item.get('url', '')}",
        f"【题目难度】{item.get('difficulty', 'unknown')}",
        "",
        "【题目描述】",
        str(item.get("question", "")).strip(),
        "",
    ]

    starter = str(item.get("starter_code", "") or "").strip()
    if starter:
        lines.extend(
            [
                "【起始代码模板】",
                "保留模板要求的公开签名，并补齐完整可运行实现：",
                "```python",
                starter,
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "【输入输出协议】",
            (
                f"这是函数调用题：实现 starter_code 中 Solution.{io_data.get('fn_name')}；"
                "每个 input 是按位置传入的方法参数数组，返回值必须符合 output 结构。"
                if io_data.get("fn_name")
                else "程序从 stdin 读取输入，并向 stdout 输出答案；不要使用文件接口。"
            ),
            "",
        ]
    )
    pairs = list(zip(inputs[:sample_count], outputs[:sample_count]))
    if pairs:
        lines.append("【公共样例】")
        for index, (input_data, output_data) in enumerate(pairs, 1):
            lines.extend(
                [
                    f"样例 {index} 输入：",
                    format_case_text(input_data, callable_case=bool(io_data.get("fn_name"))).rstrip("\n"),
                    f"样例 {index} 输出：",
                    format_case_text(output_data, callable_case=bool(io_data.get("fn_name"))).rstrip("\n"),
                    "",
                ]
            )
    lines.extend(
        [
            "【实现要求】",
            "代码必须是可直接执行的 Python 程序，满足题目约束，不能输出调试信息。",
        ]
    )
    return "\n".join(lines)


def load_selected(
    path: Path,
    *,
    offset: int = 0,
    count: int = 200,
    ids: Iterable[str] | None = None,
    difficulty: str | None = None,
) -> list[dict[str, Any]]:
    """Stream the JSONL dataset and return a deterministic selected subset."""

    requested = [str(value) for value in ids] if ids is not None else None
    requested_set = set(requested or [])
    found: dict[str, dict[str, Any]] = {}
    selected: list[dict[str, Any]] = []
    skipped = 0

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            item = json.loads(raw_line)
            problem_id = str(item.get("id", ""))
            if difficulty and item.get("difficulty") != difficulty:
                continue
            if requested is not None:
                if problem_id in requested_set:
                    found[problem_id] = item
                continue
            if skipped < offset:
                skipped += 1
                continue
            selected.append(item)
            if len(selected) >= count:
                break

    if requested is not None:
        missing = [problem_id for problem_id in requested if problem_id not in found]
        if missing:
            print(f"[warning] missing problem ids: {', '.join(missing)}", file=sys.stderr)
        return [found[problem_id] for problem_id in requested if problem_id in found]
    return selected


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    """Load valid prior JSONL rows keyed by problem ID."""

    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError:
                print(f"[warning] ignoring malformed checkpoint line {line_number}", file=sys.stderr)
                continue
            if isinstance(item, dict) and item.get("problem_id") is not None:
                records[str(item["problem_id"])] = item
    return records


def record_needs_retry(record: dict[str, Any]) -> bool:
    """Return whether a checkpoint row represents an unsuccessful run."""

    if record.get("status") == "error":
        return True
    apps_test = record.get("apps_test") or {}
    if not apps_test or not apps_test.get("all_passed", False):
        return True
    return record.get("pipeline_verdict") != "accept"


def record_attempt_count(record: dict[str, Any] | None) -> int:
    """Return the cumulative number of attempts represented by a row."""

    if not record:
        return 0
    try:
        return max(1, int(record.get("attempts", 1) or 1))
    except (TypeError, ValueError):
        return 1


def _numeric_problem_id(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def run_apps_tests(
    final_code: str,
    item: dict[str, Any],
    *,
    max_cases: int,
    case_timeout: float,
) -> dict[str, Any]:
    """Run APPS stdin/stdout cases with the repository's deterministic judge."""

    io_data = parse_io_field(item.get("input_output"))
    inputs = list(io_data.get("inputs") or [])
    outputs = list(io_data.get("outputs") or [])
    pairs = list(zip(inputs, outputs))
    if max_cases > 0:
        pairs = pairs[:max_cases]
    if not pairs:
        return {
            "all_passed": False,
            "cases_run": 0,
            "cases_passed": 0,
            "compile_error": "APPS input_output has no paired test cases",
            "cases": [],
        }

    code = str(final_code or "")
    try:
        compile(code, f"apps_{item.get('id', 'unknown')}.py", "exec")
    except SyntaxError as exc:
        return {
            "all_passed": False,
            "cases_run": 0,
            "cases_passed": 0,
            "compile_error": f"{type(exc).__name__}: {exc}",
            "cases": [],
        }

    problem_id = _numeric_problem_id(item.get("id"))
    cases: list[dict[str, Any]] = []
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", encoding="utf-8", delete=False
    ) as source_file:
        source_file.write(code)
        source_path = source_file.name

    try:
        for index, (input_data, expected_data) in enumerate(pairs):
            callable_name = str(io_data.get("fn_name") or "")
            stdin_data = format_case_text(input_data, callable_case=bool(callable_name))
            expected = format_case_text(expected_data, callable_case=bool(callable_name))
            callable_args = callable_arguments(input_data) if callable_name else []
            source_for_case = code
            if callable_name and "class Solution" in code:
                source_for_case = build_callable_driver(
                    code, callable_name, callable_args
                )
                # The driver is executed through its own temporary source;
                # switch the source file before each callable case below.
            elif callable_name:
                # Backward compatibility for a generated stdin adapter that
                # predates function-style support: unwrap the argument list.
                stdin_data = callable_stdin(callable_args)
            started = time.monotonic()
            error = ""
            try:
                case_path = source_path
                if source_for_case is not code:
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".py", encoding="utf-8", delete=False
                    ) as case_file:
                        case_file.write(source_for_case)
                        case_path = case_file.name
                completed = subprocess.run(
                    [sys.executable, case_path],
                    input=stdin_data,
                    text=True,
                    capture_output=True,
                    timeout=case_timeout,
                )
                actual = completed.stdout
                stderr = completed.stderr
                result_code = completed.returncode
                timed_out = False
            except subprocess.TimeoutExpired as exc:
                actual = str(exc.stdout or "")
                stderr = str(exc.stderr or "")
                result_code = -1
                timed_out = True
            except OSError as exc:
                actual = ""
                stderr = ""
                result_code = -1
                timed_out = False
                error = f"{type(exc).__name__}: {exc}"
            finally:
                if source_for_case is not code and case_path != source_path:
                    try:
                        os.unlink(case_path)
                    except OSError:
                        pass

            elapsed_ms = (time.monotonic() - started) * 1000.0
            judge = judge_output(
                problem_id=problem_id,
                input_data=stdin_data,
                expected=expected,
                actual=actual,
            )
            passed = bool(
                judge.get("passed")
                and not timed_out
                and result_code == 0
            )
            if timed_out:
                error = f"timeout after {case_timeout:g}s"
            elif result_code != 0 and not error:
                error = stderr.strip() or f"process exited with code {result_code}"
            cases.append(
                {
                    "index": index,
                    "passed": passed,
                    "input": stdin_data,
                    "expected": expected,
                    "actual": actual[:12000],
                    "stderr": stderr[:4000],
                    "error": error,
                    "time_ms": round(elapsed_ms, 3),
                    "timed_out": timed_out,
                    "exit_code": result_code,
                    "judge_mode": judge.get("judge_mode", "token"),
                    "judge_detail": judge.get("judge_detail", ""),
                }
            )
    finally:
        try:
            os.unlink(source_path)
        except OSError:
            pass

    passed_count = sum(bool(case["passed"]) for case in cases)
    return {
        "all_passed": bool(cases) and passed_count == len(cases),
        "cases_run": len(cases),
        "cases_passed": passed_count,
        "compile_error": None,
        "cases": cases,
    }


def configure_environment(env_path: Path, provider: str | None) -> str:
    """Load model settings and normalize the repository's legacy key names."""

    if env_path.exists():
        load_dotenv(env_path, override=False)
    if not os.getenv("QWEN_API_KEY") and os.getenv("QWEN_API"):
        os.environ["QWEN_API_KEY"] = os.environ["QWEN_API"]
    if not os.getenv("GLM_API_KEY") and os.getenv("GLM_API"):
        os.environ["GLM_API_KEY"] = os.environ["GLM_API"]
    if not os.getenv("DEEPSEEK_API_KEY") and os.getenv("DEEPSEEK_API"):
        os.environ["DEEPSEEK_API_KEY"] = os.environ["DEEPSEEK_API"]

    chosen = (provider or os.getenv("AUDITCODER_LLM_PROVIDER") or "").strip().lower()
    if not chosen:
        # DeepSeek is the historical provider for the paper's AuditCoder
        # experiments.  Fall back to Qwen/GLM for installations without it.
        if os.getenv("DEEPSEEK_API_KEY"):
            chosen = "deepseek"
        elif os.getenv("QWEN_API_KEY"):
            chosen = "qwen"
        else:
            chosen = "glm"
    if chosen not in {"qwen", "deepseek", "glm"}:
        raise ValueError("optimized AuditCoder supports provider=qwen, deepseek, or glm")
    os.environ["AUDITCODER_LLM_PROVIDER"] = chosen
    key_names = {
        "qwen": "QWEN_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "glm": "GLM_API_KEY",
    }
    key_name = key_names[chosen]
    if not os.getenv(key_name):
        raise ValueError(f"missing {key_name}; configure it in {env_path} or the environment")
    return chosen


def run_pipeline(
    item: dict[str, Any],
    *,
    plan_graph: Any,
    assembler_graph: Any,
    verifier_graph: Any,
    max_repair_attempts: int,
    external_test_cases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one item through the optimized planning-to-repair pipeline."""

    problem_text = format_apps_problem(item)
    plan_state = plan_graph.invoke({"problem_raw": problem_text, "revision_count": 0})
    ir = plan_state.get("task_plan_ir") or {}
    if not ir:
        raise RuntimeError("plan_agent returned no task_plan_ir")

    code_snippets, verification_reports = dispatch_algo_agents(ir)
    assembler_state = assembler_graph.invoke(
        {"task_plan_ir": ir, "code_snippets": code_snippets}
    )
    assembled_ir = assembler_state.get("task_plan_ir") or ir
    initial_code = str(assembler_state.get("final_code") or "")
    final_state = verifier_graph.invoke(
        {
            "problem_id": _numeric_problem_id(item.get("id")),
            "final_code": initial_code,
            "task_plan_ir": assembled_ir,
            "code_snippets": code_snippets,
            "verification_reports": verification_reports,
            "max_repair_attempts": max_repair_attempts,
            "external_test_cases": external_test_cases or [],
        }
    )
    final_code = str(final_state.get("final_code") or initial_code)
    final_ir = final_state.get("task_plan_ir") or assembled_ir
    return {
        "ir": final_ir,
        "code_snippets": code_snippets,
        "verification_reports": verification_reports,
        "final_code": final_code,
        "final_report": final_state.get("final_report", {}),
        "execution_result": final_state.get("execution_result", {}),
        "catg": catg_summary(final_ir),
        "repair_attempts": int(final_state.get("repair_attempts", 0) or 0),
    }


def _compact_record(
    item: dict[str, Any],
    pipeline: dict[str, Any],
    apps_tests: dict[str, Any],
    elapsed_s: float,
) -> dict[str, Any]:
    report = pipeline.get("final_report") or {}
    return {
        "problem_id": str(item.get("id", "")),
        "difficulty": item.get("difficulty", "unknown"),
        "url": item.get("url", ""),
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(elapsed_s, 3),
        "apps_test": apps_tests,
        "pipeline_verdict": report.get("verdict", "unknown"),
        "pipeline_all_passed": report.get("all_passed"),
        "final_report": report,
        "execution_result": pipeline.get("execution_result", {}),
        "catg": pipeline.get("catg", {}),
        "repair_attempts": pipeline.get("repair_attempts", 0),
        "task_count": len((pipeline.get("ir") or {}).get("task_nodes") or []),
        "code_snippet_count": len(pipeline.get("code_snippets") or {}),
    }


def write_summary(
    path: Path,
    selected: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    *,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Write aggregate problem/case/difficulty metrics for the checkpoint."""

    selected_ids = [str(item.get("id", "")) for item in selected]
    selected_records = [records[problem_id] for problem_id in selected_ids if problem_id in records]
    difficulty_stats: dict[str, dict[str, Any]] = {}
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in selected_records:
        grouped[str(record.get("difficulty", "unknown"))].append(record)
    for difficulty, group in sorted(grouped.items()):
        difficulty_stats[difficulty] = _aggregate_records(group)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "settings": settings,
        "selected_count": len(selected),
        "processed_count": len(selected_records),
        "remaining_count": max(0, len(selected) - len(selected_records)),
        "overall": _aggregate_records(selected_records),
        "by_difficulty": difficulty_stats,
        "status_counts": dict(Counter(str(item.get("status", "unknown")) for item in selected_records)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    apps = [record.get("apps_test") or {} for record in records if record.get("status") == "ok"]
    solved = sum(bool(result.get("all_passed")) for result in apps)
    cases_run = sum(int(result.get("cases_run", 0) or 0) for result in apps)
    cases_passed = sum(int(result.get("cases_passed", 0) or 0) for result in apps)
    pipeline_accept = sum(record.get("pipeline_verdict") == "accept" for record in records)
    return {
        "problems": len(records),
        "problems_with_apps_results": len(apps),
        "problems_passed_all_selected_cases": solved,
        "problem_pass_rate": (solved / len(records)) if records else 0.0,
        "problem_pass_rate_on_completed": (solved / len(apps)) if apps else 0.0,
        "pipeline_accept_count": pipeline_accept,
        "pipeline_accept_rate": (pipeline_accept / len(records)) if records else 0.0,
        "cases_run": cases_run,
        "cases_passed": cases_passed,
        "case_pass_rate": (cases_passed / cases_run) if cases_run else 0.0,
    }


def run_one(
    item: dict[str, Any],
    *,
    plan_graph: Any,
    assembler_graph: Any,
    verifier_graph: Any,
    max_repair_attempts: int,
    max_cases: int,
    case_timeout: float,
    artifact_dir: Path | None,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        external_test_cases = build_external_test_cases(item, max_cases=max_cases)
        pipeline = run_pipeline(
            item,
            plan_graph=plan_graph,
            assembler_graph=assembler_graph,
            verifier_graph=verifier_graph,
            max_repair_attempts=max_repair_attempts,
            external_test_cases=external_test_cases,
        )
        apps_tests = run_apps_tests(
            pipeline["final_code"],
            item,
            max_cases=max_cases,
            case_timeout=case_timeout,
        )
        record = _compact_record(item, pipeline, apps_tests, time.monotonic() - started)
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = artifact_dir / f"apps_{item.get('id', 'unknown')}.json"
            artifact = {
                "problem": item,
                "task_plan_ir": pipeline.get("ir"),
                "code_snippets": pipeline.get("code_snippets"),
                "verification_reports": pipeline.get("verification_reports"),
                "final_code": pipeline.get("final_code"),
                "final_report": pipeline.get("final_report"),
                "execution_result": pipeline.get("execution_result"),
                "apps_test": apps_tests,
            }
            artifact_path.write_text(
                json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            record["artifact_path"] = str(artifact_path)
        return record
    except Exception as exc:  # keep one bad item from discarding the batch
        return {
            "problem_id": str(item.get("id", "")),
            "difficulty": item.get("difficulty", "unknown"),
            "url": item.get("url", ""),
            "status": "error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(time.monotonic() - started, 3),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=20),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--ids", help="comma-separated APPS ids; overrides offset/count")
    selection.add_argument("--offset", type=int, default=0, help="number of selected rows to skip")
    parser.add_argument("--count", type=int, default=200, help="number of problems (default: 200)")
    parser.add_argument("--difficulty", choices=["introductory", "interview", "competition"])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--env", type=Path, default=WORKSPACE_ROOT / ".env")
    parser.add_argument("--provider", choices=["qwen", "deepseek", "glm"], default=None)
    parser.add_argument("--max-cases", type=int, default=10, help="APPS cases per problem; 0 means all")
    parser.add_argument("--case-timeout", type=float, default=10.0)
    parser.add_argument("--max-repair-attempts", type=int, default=3)
    parser.add_argument("--problem-retries", type=int, default=0, help="retries after a pipeline exception")
    parser.add_argument(
        "--retry-failed-once",
        action="store_true",
        help="rerun each unsuccessful problem at most once; cumulative attempts are checkpointed",
    )
    parser.add_argument("--resume", action="store_true", help="skip ids already present in output JSONL")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="with --resume, rerun prior errors and records that did not pass",
    )
    parser.add_argument("--save-artifacts", action="store_true", help="save full IR/code/report per problem")
    parser.add_argument("--dry-run", action="store_true", help="validate selection without model calls")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.count < 0 or args.offset < 0 or args.max_cases < 0:
        parser.error("count, offset, and max-cases must be nonnegative")
    if args.case_timeout <= 0 or args.max_repair_attempts < 0 or args.problem_retries < 0:
        parser.error("case-timeout must be positive; retry counts must be nonnegative")
    if args.retry_failed_once and args.problem_retries:
        parser.error("--retry-failed-once cannot be combined with --problem-retries")
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        parser.error(f"dataset does not exist: {input_path}")

    requested_ids = None
    if args.ids:
        requested_ids = [value.strip() for value in args.ids.split(",") if value.strip()]
    selected = load_selected(
        input_path,
        offset=args.offset,
        count=args.count,
        ids=requested_ids,
        difficulty=args.difficulty,
    )
    if not selected:
        parser.error("no APPS problems selected")

    print(f"selected {len(selected)} APPS problems from {input_path}")
    if args.dry_run:
        io_counts = [
            len(list(parse_io_field(item.get("input_output")).get("inputs") or []))
            for item in selected
        ]
        print(f"ids: {[str(item.get('id')) for item in selected[:10]]}{' ...' if len(selected) > 10 else ''}")
        print(f"problems with paired cases: {sum(count > 0 for count in io_counts)}/{len(io_counts)}")
        print(f"case range: min={min(io_counts)}, max={max(io_counts)}, total={sum(io_counts)}")
        return

    summary_path = (
        args.summary.expanduser().resolve()
        if args.summary
        else output_path.with_suffix(".summary.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    previous = load_checkpoint(output_path) if args.resume else {}
    if not args.resume:
        output_path.write_text("", encoding="utf-8")

    pending = []
    for item in selected:
        problem_id = str(item.get("id", ""))
        old = previous.get(problem_id)
        if old and args.resume:
            if args.retry_failed_once:
                # In this mode a prior row may consume only its one remaining
                # retry.  Successful rows and rows already attempted twice
                # are final for this run.
                if (
                    not record_needs_retry(old)
                    or record_attempt_count(old) >= 2
                ):
                    continue
            elif not (args.retry_errors and record_needs_retry(old)):
                continue
        pending.append(item)

    # A completed --resume run only needs its summary; do not require model
    # credentials or rebuild LangGraph objects in that case.
    if not pending:
        settings = {
            "input": str(input_path),
            "provider": args.provider or os.getenv("AUDITCODER_LLM_PROVIDER", "unknown"),
            "count": len(selected),
            "offset": args.offset,
            "max_cases": args.max_cases,
            "case_timeout": args.case_timeout,
            "max_repair_attempts": args.max_repair_attempts,
            "retry_failed_once": args.retry_failed_once,
            "problem_retries": args.problem_retries,
        }
        summary = write_summary(summary_path, selected, previous, settings=settings)
        print(
            f"nothing pending: {summary['processed_count']}/{summary['selected_count']} "
            f"already recorded; summary={summary_path}"
        )
        return

    provider = configure_environment(args.env.expanduser().resolve(), args.provider)

    settings = {
        "input": str(input_path),
        "provider": provider,
        "count": len(selected),
        "offset": args.offset,
        "max_cases": args.max_cases,
        "case_timeout": args.case_timeout,
        "max_repair_attempts": args.max_repair_attempts,
        "retry_failed_once": args.retry_failed_once,
        "problem_retries": args.problem_retries,
    }
    write_summary(summary_path, selected, previous, settings=settings)

    # Compile the static graph objects once; model calls still happen per item.
    plan_graph = build_plan_agent_graph()
    assembler_graph = build_assembler_agent_graph()
    verifier_graph = build_final_verifier_graph()
    artifact_dir = output_path.with_name(output_path.stem + "_artifacts") if args.save_artifacts else None

    print(
        f"provider={provider}; pending={len(pending)}/{len(selected)}; "
        f"APPS cases/problem={args.max_cases or 'all'}; repair budget={args.max_repair_attempts}"
    )
    with output_path.open("a", encoding="utf-8") as output_handle:
        for index, item in enumerate(pending, 1):
            problem_id = str(item.get("id", ""))
            record: dict[str, Any] | None = None
            prior_attempts = record_attempt_count(previous.get(problem_id))
            if args.retry_failed_once:
                # Fresh rows get one initial attempt plus one retry.  A
                # resumed row with one recorded attempt gets only its single
                # remaining retry.
                attempt_budget = 2 - prior_attempts
            else:
                attempt_budget = args.problem_retries + 1

            if attempt_budget <= 0:
                continue

            attempt = 0
            for attempt in range(attempt_budget):
                record = run_one(
                    item,
                    plan_graph=plan_graph,
                    assembler_graph=assembler_graph,
                    verifier_graph=verifier_graph,
                    max_repair_attempts=args.max_repair_attempts,
                    max_cases=args.max_cases,
                    case_timeout=args.case_timeout,
                    artifact_dir=artifact_dir,
                )
                if args.retry_failed_once:
                    if (
                        not record_needs_retry(record)
                        or attempt + 1 >= attempt_budget
                    ):
                        break
                elif record.get("status") == "ok" or attempt >= args.problem_retries:
                    break
            assert record is not None
            record["attempts"] = prior_attempts + attempt + 1
            record["retry_performed"] = record["attempts"] > 1
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_handle.flush()
            previous[problem_id] = record
            write_summary(summary_path, selected, previous, settings=settings)

            if not args.quiet:
                apps = record.get("apps_test") or {}
                print(
                    f"[{index}/{len(pending)}] id={problem_id} "
                    f"status={record.get('status')} "
                    f"cases={apps.get('cases_passed', 0)}/{apps.get('cases_run', 0)} "
                    f"verdict={record.get('pipeline_verdict', 'error')} "
                    f"attempts={record.get('attempts', 1)}"
                )

    summary = write_summary(summary_path, selected, previous, settings=settings)
    print(
        f"finished: {summary['processed_count']}/{summary['selected_count']} processed; "
        f"all-selected-case pass={summary['overall']['problems_passed_all_selected_cases']}; "
        f"case pass rate={summary['overall']['case_pass_rate']:.3f}"
    )
    print(f"records: {output_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
