"""Node: sample_test_runner — 执行公共测试用例。

程序化：在沙箱中逐个运行测试用例并比对输出。
"""

from __future__ import annotations

import json
from typing import Any

from auditcoder.verifier_agent.sandbox import SandboxExecutor
from auditcoder.verifier_agent.state import VerifierState


def _outputs_match(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    try:
        return json.loads(actual) == json.loads(expected)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False


def _normalize_input(test_input: Any) -> dict | None:
    """Turn *test_input* into a kwargs dict, or ``None`` if not parseable."""
    if isinstance(test_input, dict):
        return test_input
    if isinstance(test_input, str) and test_input.strip():
        try:
            parsed = json.loads(test_input)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _build_test_code(
    source_code: str,
    function_name: str,
    test_input: Any,
    params: list[dict] | None = None,
) -> str | None:
    """Wrap source code with a driver that invokes the target function.

    Three modes depending on available metadata:

    1. **Argument mode** – *function_name* and *params* are provided and
       *test_input* is a dict (or JSON-encoded dict) of keyword arguments.
       The function is called with ``function_name(**kwargs)``.
    2. **Stdin mode** – *function_name* is provided but *params* is empty.
       The function is assumed to read from ``input()`` (competition style).
    3. **Script mode** – *function_name* is empty.  The source code is
       executed directly with *test_input* piped to stdin.
    """
    if not function_name:
        return source_code

    if params:
        kwargs = _normalize_input(test_input)
        if kwargs is not None:
            param_names = [param.get("name", "") for param in params]
            if not all(name in kwargs for name in param_names):
                return None
            kwargs = {name: kwargs[name] for name in param_names}
            return (
                f"import json\n\n"
                f"{source_code}\n\n"
                f"_args = json.loads({json.dumps(json.dumps(kwargs))})\n"
                f"_result = {function_name}(**_args)\n"
                f"if _result is not None:\n"
                f"    print(json.dumps(_result, sort_keys=True, default=lambda value: value.tolist() if hasattr(value, 'tolist') else float(value)))\n"
            )
        return None

    # Fallback: stdin / competition style
    return (
        f"{source_code}\n\n"
        f"import sys\n"
        f"_input_data = sys.stdin.read()\n"
        f"_lines = _input_data.strip().splitlines()\n"
        f"_line_idx = 0\n"
        f"def input():\n"
        f"    global _line_idx\n"
        f"    if _line_idx < len(_lines):\n"
        f"        line = _lines[_line_idx]\n"
        f"        _line_idx += 1\n"
        f"        return line\n"
        f"    return ''\n"
        f"\n"
        f"_result = {function_name}()\n"
        f"if _result is not None:\n"
        f"    print(_result)\n"
    )


def sample_test_runner(state: VerifierState) -> dict:
    compile_report = state.get("compile_report", {})
    if not compile_report.get("compile_ok", False):
        return {
            "sample_test_report": {
                "passed": False,
                "results": [],
                "resource_ok": True,
                "max_time_ms": 0.0,
                "max_memory_kb": 0.0,
                "skipped": True,
            }
        }

    snippet = state["code_snippet"]
    source_code = snippet["source_code"]
    contract = state.get("task_contract", {})
    interface = contract.get("interface", {})
    function_name = interface.get("function_name", "")
    params = interface.get("params", [])

    validation = state.get("validation_info", {})
    sample_cases = validation.get("sample_test_cases", [])

    constraints = state.get("problem_info", {}).get("constraints", {})
    time_limit_str = constraints.get("time_limit", "2s")
    try:
        time_limit = float(time_limit_str.replace("s", "").strip())
    except (ValueError, AttributeError):
        time_limit = 2.0

    executor = SandboxExecutor(default_timeout_s=time_limit + 2.0)

    results: list[dict] = []
    all_passed = True
    max_time = 0.0
    resource_ok = True

    for i, case in enumerate(sample_cases):
        test_input = case.get("input", "")
        expected = case.get("expected_output", case.get("output", "")).strip()

        code = _build_test_code(source_code, function_name, test_input, params=params)
        if code is None:
            continue
        stdin_data = test_input if isinstance(test_input, str) else ""
        exec_result = executor.run(code, stdin=stdin_data)

        actual = exec_result.stdout.strip()
        passed = _outputs_match(actual, expected)
        if not passed:
            all_passed = False

        if exec_result.timed_out:
            resource_ok = False

        max_time = max(max_time, exec_result.time_ms)

        results.append({
            "case_index": i,
            "passed": passed,
            "expected": expected,
            "actual": actual,
            "time_ms": exec_result.time_ms,
            "stderr": exec_result.stderr[:500] if exec_result.stderr else "",
            "timed_out": exec_result.timed_out,
        })

    return {
        "sample_test_report": {
            "passed": all_passed,
            "results": results,
            "resource_ok": resource_ok,
            "max_time_ms": max_time,
            "max_memory_kb": 0.0,
            "skipped": not results,
            "validation_scope": validation.get("scope", "unspecified"),
        }
    }
