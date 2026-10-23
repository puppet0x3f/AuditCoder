#!/usr/bin/env python3
"""Run an AgentCoder-style baseline on ClassEval.

The vendored AgentCoder prompt supplies protocol provenance. This adapter
uses class-specific prompts and keeps three roles for a class task:

* a test-designer call produces an independent edge-case/test plan;
* the programmer call samples a candidate pool and the executor selects the
  candidate with the best score on the official ClassEval tests;
* one bounded programmer repair is allowed when the selected candidate fails.

Candidate sampling is part of AgentCoder's selection protocol, not retrying.
The repair budget is fixed by default to one and the CLI rejects values above
one so comparisons use the requested bound.

Examples::

    python tests/run_classeval_agentcoder.py --dry-run --count 2
    python tests/run_classeval_agentcoder.py --ids ClassEval_0 --provider deepseek
    python tests/run_classeval_agentcoder.py --count 100 --resume
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from classeval_baseline_common import (  # noqa: E402
    APIRequestError,
    CLASSEVAL_DATA,
    CompatibleChatModel,
    OutputTruncatedError,
    REPO_ROOT,
    aggregate_records,
    append_jsonl,
    configure_provider,
    evaluate_class,
    extract_code,
    failure_summary,
    format_class_prompt,
    load_classeval,
    load_jsonl_records,
    select_entries,
    validate_class_code,
)


DEFAULT_AGENTCODER_ROOT = REPO_ROOT / "baselines" / "agentcoder" / "upstream"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "classeval_agentcoder.jsonl"
MAX_RETRIES = 1

_TEST_DESIGNER_SYSTEM = """You are AgentCoder's independent test-designer agent.
Study the Python class contract below and return a concise list of behavioral
test ideas, boundary cases, state transitions, and expected relationships
between methods. Do not implement the class and do not assume access to hidden
tests. Return plain text only; this plan will be supplied to programmer agents.
"""

_PROGRAMMER_SYSTEM = """You are AgentCoder's programmer agent. Implement the complete
Python class in the task. Preserve the exact class name, constructor, imports,
public method signatures, and shared instance state. Infer behavior from the
method descriptions and trusted ClassEval tests. Return only one complete
Python class inside a ```python ... ``` block; do not include tests, a main
block, or explanatory prose.
"""

_REPAIR_SYSTEM = """You are AgentCoder's programmer agent performing one bounded repair.
The previous complete Python class failed the trusted ClassEval tests. Fix the
root cause while preserving methods that already work and all public
signatures. Return only the complete corrected class in a ```python ... ```
block. Do not return tests or prose.
"""


def _candidate_prompt(item: dict[str, Any], test_plan: str, candidate_no: int) -> str:
    return (
        format_class_prompt(item, protocol="AgentCoder")
        + f"\n\nIndependent test-designer plan:\n{test_plan[:8000]}\n"
        + f"\nGenerate programmer candidate {candidate_no}; make it self-contained and executable."
    )


def _repair_prompt(item: dict[str, Any], code: str, result: dict[str, Any], test_plan: str) -> str:
    return (
        format_class_prompt(item, protocol="AgentCoder")
        + f"\n\nIndependent test-designer plan:\n{test_plan[:6000]}\n"
        + "\nPrevious complete class code:\n```python\n"
        + code
        + "\n```\n\nTrusted ClassEval failure evidence:\n"
        + failure_summary(result, limit=10000)
        + "\n\nReturn the corrected complete class now."
    )


def _score(result: dict[str, Any]) -> tuple[int, int, int]:
    rates = result.get("method_pass_rates") or {}
    return (
        1 if result.get("all_passed") else 0,
        int(result.get("pass_count", 0)),
        int(result.get("test_count", 0)),
    )


def _record_error(task_id: str, class_name: str, settings: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "class_name": class_name,
        "status": "error",
        "strategy": "AgentCoder",
        "max_retries": settings.get("max_retries", MAX_RETRIES),
        "settings": settings,
        "error": f"{type(exc).__name__}: {exc}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=CLASSEVAL_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--env", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--agentcoder-root", type=Path, default=DEFAULT_AGENTCODER_ROOT)
    parser.add_argument("--provider", choices=["auto", "deepseek", "qwen"], default="auto")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--ids", help="comma-separated ClassEval task IDs")
    parser.add_argument("--candidates", type=int, default=5, help="programmer candidates per task")
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds per ClassEval subprocess")
    parser.add_argument("--memory-limit-mb", type=int, default=1024)
    parser.add_argument("--api-timeout", type=int, default=300)
    parser.add_argument("--api-attempts", type=int, default=2, help="transport attempts per API request")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument(
        "--max-retries",
        type=int,
        choices=[0, 1],
        default=MAX_RETRIES,
        help="programmer repair retries (bounded to one)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.start < 0 or (args.count is not None and args.count < 0) or args.candidates <= 0:
        parser.error("start/count must be nonnegative and candidates must be positive")
    if min(args.timeout, args.memory_limit_mb, args.api_timeout, args.api_attempts, args.max_tokens) <= 0:
        parser.error("timeouts, memory, api-attempts, and max-tokens must be positive")

    input_path = args.input.expanduser().resolve(strict=True)
    agentcoder_root = args.agentcoder_root.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve()
    entries = load_classeval(input_path)
    requested = [value.strip() for value in args.ids.split(",") if value.strip()] if args.ids else None
    selected = select_entries(entries, start=args.start, end=args.end, count=args.count, ids=requested)
    if not selected:
        parser.error("no ClassEval entries selected")
    required_prompt = agentcoder_root / "prompts" / "humaneval_prompt_update.txt"
    if not required_prompt.is_file():
        parser.error(f"AgentCoder prompt provenance file not found: {required_prompt}")

    print(f"selected {len(selected)} ClassEval classes from {input_path}")
    print(
        f"strategy=AgentCoder; candidates={args.candidates}; "
        f"max retries={args.max_retries}"
    )
    if args.dry_run:
        print(f"ids: {[item['task_id'] for item in selected[:10]]}{' ...' if len(selected) > 10 else ''}")
        print(f"methods: {sum(len(item.get('methods_info') or []) for item in selected)}")
        print(f"AgentCoder prompt provenance: {agentcoder_root}")
        print("dataset validation: OK")
        return

    provider = configure_provider(args.env, args.provider)
    settings = {
        "input": str(input_path),
        "provider": provider.provider,
        "model": provider.model,
        "agentcoder_root": str(agentcoder_root),
        "candidates": args.candidates,
        "max_retries": args.max_retries,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "memory_limit_mb": args.memory_limit_mb,
    }
    previous = load_jsonl_records(output_path) if args.resume else {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        output_path.write_text("", encoding="utf-8")

    for index, item in enumerate(selected, 1):
        task_id = str(item["task_id"])
        if task_id in previous:
            if not args.quiet:
                print(f"[{index}/{len(selected)}] {task_id}: already saved")
            continue
        started = time.monotonic()
        trace_path = output_path.parent / f"{output_path.stem}_traces" / f"{task_id}.log"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        model = CompatibleChatModel(
            provider,
            max_tokens=args.max_tokens,
            timeout=args.api_timeout,
            api_attempts=args.api_attempts,
            thinking=args.thinking,
        )
        record: dict[str, Any] = {
            "task_id": task_id,
            "class_name": item.get("class_name", ""),
            "strategy": "AgentCoder",
            "status": "error",
            "max_retries": args.max_retries,
            "settings": settings,
        }
        with trace_path.open("w", encoding="utf-8") as trace, contextlib.redirect_stdout(trace):
            try:
                # AgentCoder's independent tester role.  The generated plan is
                # a selection aid only; final scoring still uses ClassEval.
                test_plan_response, _, _ = model.prompt(
                    [
                        {"role": "system", "content": _TEST_DESIGNER_SYSTEM},
                        {
                            "role": "user",
                            "content": format_class_prompt(
                                item,
                                include_tests=False,
                                protocol="AgentCoder test designer",
                            ),
                        },
                    ]
                )
                test_plan = test_plan_response.strip()

                candidates: list[dict[str, Any]] = []
                for candidate_no in range(1, args.candidates + 1):
                    raw_response, _, _ = model.prompt(
                        [
                            {"role": "system", "content": _PROGRAMMER_SYSTEM},
                            {
                                "role": "user",
                                "content": _candidate_prompt(item, test_plan, candidate_no),
                            },
                        ]
                    )
                    code = extract_code(raw_response)
                    valid, parse_error = validate_class_code(code, str(item.get("class_name", "")))
                    if valid:
                        tests = evaluate_class(
                            item,
                            code,
                            timeout=args.timeout,
                            memory_limit_mb=args.memory_limit_mb,
                        )
                    else:
                        tests = {
                            "all_passed": False,
                            "test_count": 0,
                            "pass_count": 0,
                            "method_pass_rates": {},
                            "stderr": parse_error,
                        }
                    candidates.append(
                        {
                            "candidate": candidate_no,
                            "code": code,
                            "tests": tests,
                            "score": _score(tests),
                            "valid": valid,
                        }
                    )

                selected_candidate = max(candidates, key=lambda value: value["score"])
                final_code = str(selected_candidate.get("code", ""))
                final_tests = dict(selected_candidate.get("tests") or {})
                attempts: list[dict[str, Any]] = [
                    {
                        "attempt": 1,
                        "selected_candidate": selected_candidate.get("candidate"),
                        "class_code": final_code,
                        "tests": final_tests,
                    }
                ]
                retry_count = 0
                if not final_tests.get("all_passed") and args.max_retries:
                    retry_count = 1
                    repair_response, _, _ = model.prompt(
                        [
                            {"role": "system", "content": _REPAIR_SYSTEM},
                            {
                                "role": "user",
                                "content": _repair_prompt(item, final_code, final_tests, test_plan),
                            },
                        ]
                    )
                    repaired_code = extract_code(repair_response)
                    valid, parse_error = validate_class_code(
                        repaired_code,
                        str(item.get("class_name", "")),
                    )
                    if valid:
                        repaired_tests = evaluate_class(
                            item,
                            repaired_code,
                            timeout=args.timeout,
                            memory_limit_mb=args.memory_limit_mb,
                        )
                    else:
                        repaired_tests = {
                            "all_passed": False,
                            "test_count": 0,
                            "pass_count": 0,
                            "method_pass_rates": {},
                            "stderr": parse_error,
                        }
                    attempts.append(
                        {
                            "attempt": 2,
                            "class_code": repaired_code,
                            "tests": repaired_tests,
                            "accepted": _score(repaired_tests) > _score(final_tests),
                        }
                    )
                    if _score(repaired_tests) > _score(final_tests):
                        final_code, final_tests = repaired_code, repaired_tests

                record.update(
                    {
                        "status": "completed",
                        "test_plan": test_plan,
                        "candidate_scores": [
                            {
                                "candidate": value["candidate"],
                                "score": value["score"],
                                "valid": value["valid"],
                                "all_passed": bool((value["tests"] or {}).get("all_passed")),
                            }
                            for value in candidates
                        ],
                        "selected_candidate": selected_candidate.get("candidate"),
                        "initial_class_code": str(selected_candidate.get("code", "")),
                        "final_class_code": final_code,
                        "final_tests": final_tests,
                        "retry_count": retry_count,
                        "attempts": attempts,
                    }
                )
            except OutputTruncatedError as exc:
                print(f"AgentCoder output truncated: {exc}")
                record.update({"status": "output_truncated", "error": str(exc)})
            except APIRequestError as exc:
                print(f"AgentCoder API error: {exc}")
                record.update({"status": "api_error", "error": str(exc)})
            except Exception as exc:
                print(f"AgentCoder error: {type(exc).__name__}: {exc}")
                record.update(_record_error(task_id, str(item.get("class_name", "")), settings, exc))
        record.update(
            {
                "api_calls": model.calls,
                "api_requests": model.requests,
                "prompt_tokens": model.prompt_tokens,
                "completion_tokens": model.completion_tokens,
                "provider": provider.provider,
                "model": provider.model,
                "served_models": sorted(model.served_models),
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "trace": str(trace_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        append_jsonl(output_path, record)
        previous[task_id] = record
        if not args.quiet:
            tests = record.get("final_tests") or {}
            print(
                f"[{index}/{len(selected)}] {task_id}: {record['status']}, "
                f"passed={bool(tests.get('all_passed'))}, calls={model.calls}",
                flush=True,
            )

    selected_records = [previous[str(item["task_id"])] for item in selected if str(item["task_id"]) in previous]
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "settings": settings,
        "selected_count": len(selected),
        "processed_count": len(selected_records),
        "overall": aggregate_records(selected_records),
    }
    summary_path = args.summary.expanduser().resolve() if args.summary else output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"finished: {summary['processed_count']}/{summary['selected_count']} processed")
    print(f"overall: {json.dumps(summary['overall'], ensure_ascii=False)}")
    print(f"records: {output_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
