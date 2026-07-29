"""Node: counterexample_gen — 反例生成与最小化。

程序化：收集失败用例 + delta-debugging 缩小。
"""

from __future__ import annotations

import json
from typing import Any

from auditcoder.verifier_agent.sandbox import SandboxExecutor
from auditcoder.verifier_agent.state import VerifierState


def _classify_error(stderr: str, timed_out: bool) -> str:
    if timed_out:
        return "time_limit_exceeded"
    if "MemoryError" in stderr:
        return "memory_limit_exceeded"
    if stderr.strip():
        return "runtime_error"
    return "wrong_answer"


def _build_run_code(
    source_code: str,
    function_name: str,
    test_input: Any,
    params: list[dict] | None = None,
) -> str:
    """Build executable code for a single test, supporting both calling styles."""
    if params and isinstance(test_input, dict):
        return (
            f"import json\n\n"
            f"{source_code}\n\n"
            f"_args = json.loads({json.dumps(json.dumps(test_input))})\n"
            f"_r = {function_name}(**_args)\n"
            f"if _r is not None: print(_r)\n"
        )

    stdin_lines = test_input if isinstance(test_input, str) else ""
    lines = stdin_lines.strip().splitlines()
    return (
        f"{source_code}\n\n"
        f"import sys\n"
        f"_lines = {lines!r}\n"
        f"_idx = 0\n"
        f"def input():\n"
        f"    global _idx\n"
        f"    if _idx < len(_lines):\n"
        f"        line = _lines[_idx]\n"
        f"        _idx += 1\n"
        f"        return line\n"
        f"    return ''\n"
        f"_r = {function_name}()\n"
        f"if _r is not None: print(_r)\n"
    )


def _try_minimize_input(
    source_code: str,
    function_name: str,
    failing_input: Any,
    expected_output: str,
    executor: SandboxExecutor,
    params: list[dict] | None = None,
) -> Any:
    """Attempt to shrink a failing input via delta debugging.

    For dict inputs (argument mode): try shortening list-valued params.
    For string inputs (stdin mode): try removing lines from the end.
    """
    if params and isinstance(failing_input, dict):
        return _minimize_dict_input(
            source_code, function_name, failing_input,
            expected_output, executor, params,
        )

    if not isinstance(failing_input, str):
        return failing_input

    lines = failing_input.strip().splitlines()
    if len(lines) <= 1:
        return failing_input

    for i in range(len(lines) - 1, 0, -1):
        code = _build_run_code(source_code, function_name, "\n".join(lines[:i]))
        result = executor.run(code, timeout_s=5.0)
        actual = result.stdout.strip()
        still_fails = (
            actual != expected_output
            or result.exit_code != 0
            or result.timed_out
        )
        if still_fails:
            lines = lines[:i]
        else:
            break

    return "\n".join(lines)


def _minimize_dict_input(
    source_code: str,
    function_name: str,
    failing_input: dict,
    expected_output: str,
    executor: SandboxExecutor,
    params: list[dict],
) -> dict:
    """Shrink list-valued arguments in a kwargs dict."""
    current = dict(failing_input)

    for p in params:
        key = p["name"]
        val = current.get(key)
        if not isinstance(val, list) or len(val) <= 1:
            continue

        items = list(val)
        for i in range(len(items) - 1, 0, -1):
            candidate = dict(current, **{key: items[:i]})
            code = _build_run_code(source_code, function_name, candidate, params)
            result = executor.run(code, timeout_s=5.0)
            actual = result.stdout.strip()
            still_fails = (
                actual != expected_output
                or result.exit_code != 0
                or result.timed_out
            )
            if still_fails:
                items = items[:i]
            else:
                break

        current[key] = items

    return current


def counterexample_gen(state: VerifierState) -> dict:
    sample_report = state.get("sample_test_report", {})
    bruteforce_report = state.get("bruteforce_report", {})
    compile_report = state.get("compile_report", {})

    snippet = state.get("code_snippet", {})
    source_code = snippet.get("source_code", "")
    contract = state.get("task_contract", {})
    interface = contract.get("interface", {})
    function_name = interface.get("function_name", "")
    params = interface.get("params", [])

    failing_cases: list[dict] = []

    if not compile_report.get("compile_ok", False):
        errors = compile_report.get("errors", [])
        return {
            "counterexample": {
                "minimal_input": "",
                "expected_output": "",
                "actual_output": "; ".join(errors)[:500],
                "error_category": "runtime_error",
                "suspected_reason": "代码编译/语法错误",
            }
        }

    for r in sample_report.get("results", []):
        if not r.get("passed", True):
            failing_cases.append({
                "input": r.get("input", ""),
                "expected": r.get("expected", ""),
                "actual": r.get("actual", ""),
                "stderr": r.get("stderr", ""),
                "timed_out": r.get("timed_out", False),
            })

    for m in bruteforce_report.get("mismatches", []):
        failing_cases.append({
            "input": str(m.get("input", "")),
            "expected": str(m.get("bruteforce_output", "")),
            "actual": str(m.get("candidate_output", m.get("candidate_error", ""))),
            "stderr": "",
            "timed_out": False,
        })

    if not failing_cases:
        return {"counterexample": None}

    best = failing_cases[0]
    error_cat = _classify_error(best.get("stderr", ""), best.get("timed_out", False))

    minimal_input = best.get("input", "")
    if function_name and error_cat == "wrong_answer" and minimal_input:
        executor = SandboxExecutor(default_timeout_s=5.0)
        minimal_input = _try_minimize_input(
            source_code, function_name, minimal_input,
            best.get("expected", ""), executor, params=params,
        )

    suspected_reasons = {
        "wrong_answer": "算法逻辑错误或边界处理不当",
        "runtime_error": "运行时异常（越界/除零/类型错误等）",
        "time_limit_exceeded": "时间复杂度过高或存在死循环",
        "memory_limit_exceeded": "空间复杂度过高或内存泄漏",
    }

    return {
        "counterexample": {
            "minimal_input": minimal_input,
            "expected_output": best.get("expected", ""),
            "actual_output": best.get("actual", ""),
            "error_category": error_cat,
            "suspected_reason": suspected_reasons.get(error_cat, "未知"),
        }
    }
