"""Run the vendored codesim strategy on the fixed 200-task APPS sample.

From the repository root::

    python baselines/codesim/run_apps_200_qwen.py --check
    python baselines/codesim/run_apps_200_qwen.py --limit 1
    python baselines/codesim/run_apps_200_qwen.py --max-cases 10

The first two APPS pairs serve as public examples. The final judge checks ten
pairs by default; --max-cases 0 checks all stored pairs. Reusing the same results
path resumes saved task IDs. See docs/TESTING.md for protocol and output details.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT
DEFAULT_INPUT = REPO_ROOT / "data" / "apps_sampled_200.jsonl"
DEFAULT_RESULTS = REPO_ROOT / "results" / "codesim_apps_200_qwen.jsonl"
CODE_SIM_ROOT = REPO_ROOT / "baselines" / "codesim" / "upstream"
EXPECTED_COUNTS = {"introductory": 50, "interview": 100, "competition": 50}


# Reuse the tested local APPS loader, judge, and Qwen HTTP client.  The
# import is deliberately done by path so this script works from any cwd.
MAPCODER_RUNNER_DIR = REPO_ROOT / "baselines" / "mapcoder"
if str(MAPCODER_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(MAPCODER_RUNNER_DIR))

from run_apps_200_qwen import (  # noqa: E402
    APIRequestError,
    QwenModel,
    LocalAPPSJudge,
    OutputTruncatedError,
    append_result,
    code_from_response,
    load_previous,
    load_sample,
    normalize_qwen_model,
    read_env_file,
)


def make_dataset_class(upstream_apps_class: type):
    """Build an APPSDataset-compatible view over the fixed JSONL sample."""

    class SampledAPPSDataset(upstream_apps_class):
        def __init__(self, rows: list[dict[str, Any]], judge: LocalAPPSJudge):
            self.id_key = "id"
            self.judge = judge
            self.data: list[dict[str, Any]] = []
            for row in rows:
                io = row["_io"]
                fn_name = io.get("fn_name")
                cases = []
                for raw_input, expected in zip(io["inputs"], io["outputs"]):
                    cases.append({
                        "input": json.dumps(raw_input, ensure_ascii=False) if fn_name else raw_input,
                        "args": raw_input if fn_name else None,
                        "expected": expected,
                        "output": [expected],
                    })
                self.data.append({
                    "id": row["id"],
                    "description": row["question"],
                    "difficulty": row["difficulty"],
                    "fn_name": fn_name,
                    "starter_code": row.get("starter_code", ""),
                    "sample_io": cases[:2],
                    "test_list": cases,
                })

        @staticmethod
        def get_prompt(item: dict[str, Any]) -> str:
            sample_io_format = ""
            if item["sample_io"]:
                sample = item["sample_io"][0]
                sample_io_format = (
                    f"Sample Input Format:\n{sample['input']}\n"
                    f"Sample Output Format:\n{sample['output'][0]}\n\n-------\n"
                )

            if item["fn_name"]:
                pre_marker_instruction = (
                    f"Implement the Python function {item['fn_name']} (either at top level "
                    "or as a method of class Solution). The evaluator calls it with "
                    "positional arguments and compares its return value. Do not read "
                    "stdin or print the answer."
                )
                post_marker_instruction = pre_marker_instruction
            else:
                pre_marker_instruction = ""
                post_marker_instruction = (
                    "You must follow the input output format. Input "
                    "should be taken from standard input and output should be given to "
                    "standard output."
                )

            # CodeSIM removes everything from this marker onward for its
            # planning/simulation prompt, so retain the upstream marker exactly.
            return (
                f"{item['description']}\n\n{pre_marker_instruction}\n\n"
                f"{sample_io_format}\n-------\nImportant Note: "
                f"{post_marker_instruction}"
            )

        def evaluate_sample_io(self, item: dict[str, Any], cur_imp: str, language: str):
            passed, feedback = self.judge.check(
                cur_imp, item["sample_io"], item["fn_name"], sample=True
            )
            if passed:
                return True, "## Tests failed:\n"
            return False, f"## Tests failed:\n{feedback}"

        def evaluate_additional_io(
            self,
            id: int,
            tests: list[str],
            cur_imp: str,
            language: str,
        ):
            # CodeSIM's additional-IO generation is disabled for competitive
            # datasets in the upstream implementation.
            return True, ""

        def evaluate(self, item: dict[str, Any], cur_imp: str, language: str):
            passed, _ = self.judge.check(
                cur_imp, item["test_list"], item["fn_name"], sample=False
            )
            return passed

    return SampledAPPSDataset


class CodeSimQwenModel:
    """Adapt the local Qwen client to CodeSim's model protocol."""

    def __init__(self, **kwargs: Any):
        self.raw = QwenModel(**kwargs)

    @property
    def calls(self) -> int:
        return self.raw.calls

    @property
    def prompt_tokens(self) -> int:
        return self.raw.prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self.raw.completion_tokens

    @property
    def served_models(self) -> set[str]:
        return self.raw.served_models

    def prompt(
        self,
        processed_input: list[dict[str, str]],
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
    ) -> tuple[str, dict[str, Any]]:
        del frequency_penalty, presence_penalty
        before = self.raw.calls
        started = time.monotonic()
        content, prompt_tokens, completion_tokens = self.raw.prompt(processed_input)
        details = {
            "api_calls": self.raw.calls - before,
            "taken_time": time.monotonic() - started,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            # Full prompts/responses are already available in the per-task
            # trace when CodeSIM verbose mode is requested.  Keeping this list
            # empty prevents result files from becoming enormous by default.
            "details": [],
        }
        return content, details


def summarize_run_details(details: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in details.items()
        if key != "details"
    }


def print_summary(records: dict[int, dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    print(f"Processed: {len(records)}/{len(rows)}")
    for difficulty, target in EXPECTED_COUNTS.items():
        subset = [r for r in records.values() if r["difficulty"] == difficulty]
        solved = sum(bool(r.get("is_solved")) for r in subset)
        status_counts = Counter(r.get("status") for r in subset)
        score = f"{100 * solved / target:.1f}%" if len(subset) == target else "incomplete"
        print(
            f"{difficulty}: {solved}/{target} solved, "
            f"{len(subset)}/{target} processed, score={score}, "
            f"statuses={dict(status_counts)}"
        )
    solved = sum(bool(r.get("is_solved")) for r in records.values())
    score = f"{100 * solved / len(rows):.1f}%" if len(records) == len(rows) else "incomplete"
    print(f"Total: {solved}/{len(rows)} solved, {len(records)}/{len(rows)} processed, score={score}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--env", type=Path, default=WORKSPACE_ROOT / ".env")
    parser.add_argument("--codesim-root", type=Path, default=CODE_SIM_ROOT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--limit", type=int, default=0, help="First N problems; 0 means all")
    parser.add_argument("--max-cases", type=int, default=10, help="Final cases per problem; 0 means all")
    parser.add_argument("--case-timeout", type=int, default=10)
    parser.add_argument("--api-timeout", type=int, default=120, help="Timeout for one Qwen HTTP request in seconds")
    parser.add_argument("--api-attempts", type=int, default=2, help="Attempts per Qwen request")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-plan-try", type=int, default=5)
    parser.add_argument("--max-debug-try", type=int, default=5)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--check", action="store_true", help="Validate files without sending API calls")
    args = parser.parse_args()

    if min(args.limit, args.max_cases) < 0:
        parser.error("limit and max-cases must be nonnegative")
    if min(
        args.case_timeout,
        args.api_timeout,
        args.api_attempts,
        args.max_tokens,
        args.max_plan_try,
        args.max_debug_try,
    ) <= 0:
        parser.error("timeouts, api-attempts, max-tokens, max-plan-try, and max-debug-try must be positive")

    input_path = args.input.resolve(strict=True)
    codesim_root = args.codesim_root.resolve(strict=True)
    results_path = args.results.resolve()
    rows = load_sample(input_path)
    args.env = args.env.expanduser().resolve()
    sys.path.insert(0, str(codesim_root / "src"))
    os.chdir(codesim_root)
    from datasets.APPSDataset import APPSDataset
    import promptings.CodeSIM as codesim_module
    from promptings.CodeSIM import CodeSIM

    if args.check:
        dataset = make_dataset_class(APPSDataset)(
            rows, LocalAPPSJudge(timeout=args.case_timeout, max_cases=args.max_cases)
        )
        print(f"CodeSim imports OK; dataset: {len(dataset)} tasks; no API calls.")
        return

    read_env_file(args.env)
    requested_model = os.getenv("QWEN_MODEL", "").strip()
    base_url = os.getenv("QWEN_BASE_URL", "").strip()
    api_key = (os.getenv("QWEN_API_KEY") or os.getenv("QWEN_API") or "").strip()
    model_name = normalize_qwen_model(requested_model, base_url)
    if not model_name or not base_url or not api_key:
        raise ValueError("Configure QWEN_API_KEY, QWEN_MODEL, and QWEN_BASE_URL")
    print(f"Dataset: {len(rows)} tasks; model ID: {model_name}; final cases: {args.max_cases or 'all'}")

    settings = {
        "input": str(input_path),
        "model": model_name,
        "thinking": args.thinking,
        "max_tokens": args.max_tokens,
        "max_cases": args.max_cases,
        "case_timeout": args.case_timeout,
        "api_timeout": args.api_timeout,
        "api_attempts": args.api_attempts,
        "codesim_root": str(codesim_root),
        "max_plan_try": args.max_plan_try,
        "max_debug_try": args.max_debug_try,
    }
    previous = load_previous(results_path, settings)

    # The upstream parser has brittle language-fence matching.  Use the same
    # tested parser as the MapCoder runner for code and debug responses.
    codesim_module.parse_response = code_from_response

    judge = LocalAPPSJudge(timeout=args.case_timeout, max_cases=args.max_cases)
    dataset = make_dataset_class(APPSDataset)(rows, judge)
    target = dataset.data[:args.limit] if args.limit else dataset.data

    for index, item in enumerate(target, 1):
        pid = item["id"]
        if pid in previous:
            print(f"[{index}/{len(target)}] {pid}: already saved")
            continue

        model = CodeSimQwenModel(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            thinking=args.thinking,
            max_tokens=args.max_tokens,
            timeout=args.api_timeout,
            attempts=args.api_attempts,
        )
        # CodeSIM checks exact type equality in its constructor.  The sampled
        # dataset is a deliberate subclass, so set the competitive flag after
        # construction to retain the upstream competitive prompts/flow.
        strategy = CodeSIM(
            model=model,
            data=dataset,
            language="Python3",
            pass_at_k=1,
            results=None,
            verbose=args.verbose,
            max_plan_try=args.max_plan_try,
            max_debug_try=args.max_debug_try,
        )
        strategy.is_competitive = True
        strategy.run_details = {}

        trace_path = results_path.parent / (results_path.stem + "_traces") / f"{pid:04d}.log"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        record: dict[str, Any] = {
            "id": pid,
            "difficulty": item["difficulty"],
            "settings": settings,
        }
        code = ""
        with trace_path.open("w", encoding="utf-8") as trace, contextlib.redirect_stdout(trace):
            try:
                code = strategy.run_single_pass(item)
                code = code_from_response(code)
                solved = dataset.evaluate(item, code, "Python3")
                record.update({
                    "status": "completed",
                    "is_solved": solved,
                    "code": code,
                    "evaluation": judge.last_result,
                })
            except OutputTruncatedError as exc:
                print(f"Output truncated: {exc}")
                record.update({
                    "status": "output_truncated",
                    "is_solved": False,
                    "error": str(exc),
                })
            except APIRequestError:
                # A connectivity/authentication/server failure should stop the
                # run rather than silently turning the remaining tasks into 0s.
                raise
            except Exception as exc:
                print(f"Method error: {type(exc).__name__}: {exc}")
                record.update({
                    "status": "method_error",
                    "is_solved": False,
                    "error": f"{type(exc).__name__}: {exc}",
                })

        record.update({
            "api_calls": model.calls,
            "prompt_tokens": model.prompt_tokens,
            "completion_tokens": model.completion_tokens,
            "served_models": sorted(model.served_models),
            "api_requests": model.raw.requests,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "run_details": summarize_run_details(strategy.run_details),
            "trace": str(trace_path),
        })
        append_result(results_path, record)
        previous[pid] = record
        print(
            f"[{index}/{len(target)}] {pid} {item['difficulty']}: {record['status']}, "
            f"solved={record['is_solved']}, calls={model.calls}",
            flush=True,
        )

    print_summary(previous, rows)
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
