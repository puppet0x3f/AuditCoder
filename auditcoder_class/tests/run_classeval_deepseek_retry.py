"""Direct Qwen baseline for ClassEval with test-feedback retry.

This runner skips the auditcoder_class pipeline. It sends the whole ClassEval task
directly to Qwen, extracts the generated class code, runs the official
unit tests, and feeds failures back to the model for retry.

Usage:
  python auditcoder_class/tests/run_classeval_deepseek_retry.py --start 0 --count 5
  python auditcoder_class/tests/run_classeval_deepseek_retry.py --ids ClassEval_0,ClassEval_5
  python auditcoder_class/tests/run_classeval_deepseek_retry.py --start 0 --count 10 --max-retries 3
  python auditcoder_class/tests/run_classeval_deepseek_retry.py --start 0 --count 3 --output data/classeval_qwen_retry
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

from auditcoder_class.tests.run_classeval_cot import (
    CLASSEVAL_DATA_PATH,
    _build_error_summary,
    _format_entry,
    extract_class_code,
    load_classeval,
    run_classeval_evaluation,
    select_problems,
)

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "classeval_qwen_retry"
)
MAX_RETRIES_DEFAULT = 2

_API_RETRY_ATTEMPTS = 3
_API_RETRY_BASE_DELAY = 5.0

_SYSTEM_PROMPT = """\
You are a senior Python engineer. Implement the requested Python class.

Requirements:
1. Output exactly one complete Python code block wrapped in ```python ... ```.
2. The code must define the requested class and all required methods.
3. If an __init__ method is provided in the prompt, copy its behavior exactly.
4. Do not add top-level execution code, examples, print calls, or explanations.
5. Use only the Python standard library and the imports listed in the prompt.
"""

_RETRY_SYSTEM_PROMPT = """\
You are a senior Python engineer. Your previous class implementation failed
some unit tests. Use the failure details to produce a corrected complete class.

Requirements:
1. Output exactly one complete Python code block wrapped in ```python ... ```.
2. Fix the root cause of the failing methods while preserving working behavior.
3. Include the full class definition, not only changed methods.
4. Do not add top-level execution code, examples, print calls, or explanations.
"""


def _build_llm() -> Any:
    """Build a Qwen ChatOpenAI client."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("QWEN_MODEL", "qwen-plus"),
        base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/"),
        api_key=os.getenv("QWEN_API_KEY"),
        temperature=0.0,
        extra_body={"enable_thinking": False},
    )


def _invoke_with_retry(messages: list[Any], label: str = "") -> str:
    """Invoke Qwen with a small retry loop for transient API errors."""
    llm = _build_llm()
    last_exc: Exception | None = None
    for attempt in range(1, _API_RETRY_ATTEMPTS + 1):
        try:
            return llm.invoke(messages).content
        except Exception as exc:
            err_text = f"{type(exc).__name__}: {exc}"
            is_transient = any(
                key in err_text
                for key in (
                    "ConnectionError",
                    "APIConnectionError",
                    "RemoteProtocolError",
                    "ConnectError",
                    "ReadTimeout",
                    "TimeoutError",
                )
            )
            if not is_transient or attempt == _API_RETRY_ATTEMPTS:
                raise
            delay = _API_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            suffix = f" ({label})" if label else ""
            print(
                f"  [API transient error{suffix}; retry {attempt}/"
                f"{_API_RETRY_ATTEMPTS - 1} after {delay:.0f}s] {exc}"
            )
            time.sleep(delay)
            last_exc = exc
    raise last_exc  # type: ignore[misc]


def call_qwen(problem_text: str) -> str:
    """Ask Qwen for the initial complete class implementation."""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"{problem_text}\n\n"
                "Please implement the complete Python class now."
            )
        ),
    ]
    return _invoke_with_retry(messages, label="initial")


def call_qwen_retry(
    problem_text: str,
    prev_code: str,
    error_summary: str,
    attempt: int,
) -> str:
    """Ask Qwen to repair the previous class using unit-test failures."""
    from langchain_core.messages import HumanMessage, SystemMessage

    user_content = (
        f"{problem_text}\n\n"
        "---\n"
        f"Retry attempt: {attempt}\n\n"
        "Previous class code:\n"
        "```python\n"
        f"{prev_code}\n"
        "```\n\n"
        "Unit-test failure summary:\n"
        f"{error_summary}\n"
        "---\n\n"
        "Return the corrected complete Python class code."
    )
    messages = [
        SystemMessage(content=_RETRY_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    return _invoke_with_retry(messages, label=f"retry-{attempt}")


def run_single_problem(
    entry: dict[str, Any],
    args: argparse.Namespace,
    max_retries: int = MAX_RETRIES_DEFAULT,
) -> dict[str, Any]:
    task_id = entry["task_id"]
    if not args.quiet:
        print(f"\n{'=' * 60}")
        print(f"[{task_id}] {entry.get('class_name', '')}")
        print(f"{'=' * 60}")

    result: dict[str, Any] = {
        "task_id": task_id,
        "class_name": entry.get("class_name", ""),
    }

    try:
        problem_text = _format_entry(entry)

        if not args.quiet:
            print("  Calling Qwen for initial class code...")
        raw_response = call_qwen(problem_text)
        final_class_code = extract_class_code(raw_response)
        eval_result = run_classeval_evaluation(final_class_code, entry)

        attempts: list[dict[str, Any]] = [
            {
                "attempt": 1,
                "raw_response": raw_response,
                "final_class_code": final_class_code,
                "classeval_test_results": eval_result,
            }
        ]

        retry_count = 0
        while not eval_result.get("all_passed", False) and retry_count < max_retries:
            retry_count += 1
            n_pass = eval_result.get("pass_count", 0)
            n_methods = len(eval_result.get("method_pass_rates", {}))
            if not args.quiet:
                print(
                    f"  [Retry {retry_count}/{max_retries}] "
                    f"{n_pass}/{n_methods} methods passed; sending failures..."
                )

            error_summary = _build_error_summary(eval_result)
            raw_response = call_qwen_retry(
                problem_text,
                final_class_code,
                error_summary,
                retry_count,
            )
            final_class_code = extract_class_code(raw_response)
            eval_result = run_classeval_evaluation(final_class_code, entry)

            attempts.append(
                {
                    "attempt": retry_count + 1,
                    "raw_response": raw_response,
                    "final_class_code": final_class_code,
                    "classeval_test_results": eval_result,
                }
            )

        result["classeval_test_results"] = eval_result
        result["final_class_code"] = final_class_code
        result["raw_response"] = raw_response
        result["retry_count"] = retry_count
        result["total_attempts"] = len(attempts)
        result["attempts"] = attempts
        result["timestamp"] = datetime.now().isoformat()

        passed = eval_result.get("all_passed", False)
        n_methods = len(eval_result.get("method_pass_rates", {}))
        n_pass = eval_result.get("pass_count", 0)
        status = "PASS" if passed else "FAIL"
        if not args.quiet:
            print(f"Result: {status} ({n_pass}/{n_methods} methods passed)")

    except Exception as exc:
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        result["classeval_test_results"] = {"all_passed": False}
        result["timestamp"] = datetime.now().isoformat()
        if not args.quiet:
            print(f"ERROR: {exc}")

    return result


def save_result(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = result["task_id"].replace("/", "_")
    out_path = output_dir / f"{safe_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run direct Qwen baseline with retry on ClassEval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--start", type=int, default=None, help="Start index, 0-based")
    parser.add_argument("--end", type=int, default=None, help="End index, exclusive")
    parser.add_argument("--count", type=int, default=None, help="Max number of tasks")
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated task IDs")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES_DEFAULT,
        help=f"Max model repair retries per task, default {MAX_RETRIES_DEFAULT}",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce terminal output")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR
    problems = load_classeval(CLASSEVAL_DATA_PATH)
    selected = select_problems(problems, args)

    if not selected:
        print("No problems selected.")
        return

    retry_label = f"max retries {args.max_retries}" if args.max_retries > 0 else "no retry"
    print("[Direct Qwen ClassEval]")
    print(f"Model: {os.getenv('QWEN_MODEL', 'qwen-plus')} | {retry_label}")
    print(f"Running {len(selected)} task(s) -> {output_dir}")

    passed_count = 0
    for entry in selected:
        result = run_single_problem(entry, args, max_retries=args.max_retries)
        save_result(result, output_dir)
        if result.get("classeval_test_results", {}).get("all_passed", False):
            passed_count += 1

    print(f"\nDone. {passed_count}/{len(selected)} tasks passed all tests.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
