"""Node: full_code_runner — run the assembled final_code against all validation test cases.

Uses SandboxExecutor to execute the complete solution and records the outcome
in execution_result, capturing timing, stdout/stderr, and timeout status.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from auditcoder.common.execution_contract import (
    normalize_execution_contract,
    resolve_required_input_paths,
    resolve_required_output_paths,
)
from auditcoder.common.output_judging import judge_output
from auditcoder.verifier_agent.sandbox import SandboxExecutor
from auditcoder.final_verifier_agent.state import FinalVerifierState


def _parse_time_limit(constraints: dict[str, Any]) -> float:
    raw = constraints.get("time_limit", "2s")
    try:
        return float(str(raw).replace("s", "").strip())
    except (ValueError, AttributeError):
        return 2.0


def _extract_function_name_from_code(final_code: str) -> str:
    """Best-effort: extract the outermost solve() function name, otherwise first def."""
    for name in ("solve", "main", "solution"):
        if re.search(rf"^\s*def\s+{name}\s*\(", final_code, re.MULTILINE):
            return name
    m = re.search(r"^\s*def\s+(\w+)\s*\(", final_code, re.MULTILINE)
    return m.group(1) if m else ""


def _file_contract_error(detail: str) -> dict[str, Any]:
    return {
        "ran_tests": False,
        "contract_checks": True,
        "all_passed": False,
        "timed_out": False,
        "exit_code": -1,
        "stdout": "",
        "stderr": detail,
        "time_ms": 0.0,
        "results": [],
        "error_type": "contract_error",
        "test_source": "execution_contract",
    }


def _run_file_contract(
    final_code: str,
    task_plan_ir: dict[str, Any],
    executor: SandboxExecutor,
) -> dict[str, Any]:
    """Execute a file task once and require fresh declared output files."""
    try:
        contract = normalize_execution_contract(
            task_plan_ir.get("execution_contract")
        )
        input_paths = resolve_required_input_paths(contract)
        output_paths = resolve_required_output_paths(contract)
    except ValueError as exc:
        return _file_contract_error(str(exc))

    app_dir = Path(contract["app_dir"])
    if not app_dir.is_dir():
        return _file_contract_error(
            f"execution contract app_dir does not exist: {app_dir}"
        )

    missing_inputs = [
        relative
        for relative, path in input_paths
        if not path.is_file() or path.is_symlink()
    ]
    if missing_inputs:
        return _file_contract_error(
            "required input files are missing or invalid: "
            + ", ".join(missing_inputs)
        )

    cleanup_errors: list[str] = []
    for relative, path in output_paths:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            cleanup_errors.append(
                f"required output path is not a file: {relative}"
            )
    if cleanup_errors:
        return _file_contract_error("; ".join(cleanup_errors))

    result = executor.run(
        final_code,
        stdin="",
        workspace=app_dir,
        writable_workspace=True,
    )
    output_checks: list[dict[str, Any]] = []
    for relative, path in output_paths:
        passed = (
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size > 0
        )
        output_checks.append(
            {
                "path": relative,
                "passed": passed,
                "expected": "fresh non-empty regular file",
                "actual": (
                    f"file size={path.stat().st_size}"
                    if passed
                    else "missing, empty, non-regular, or symlink"
                ),
                "judge_mode": "file_contract",
                "judge_detail": "required output materialization",
            }
        )

    execution_ok = not result.timed_out and result.exit_code == 0
    outputs_ok = all(check["passed"] for check in output_checks)
    if not execution_ok:
        error_type = _classify_error(result)
    elif not outputs_ok:
        error_type = "contract_error"
    else:
        error_type = "none"
    return {
        "ran_tests": False,
        "contract_checks": True,
        "all_passed": execution_ok and outputs_ok,
        "timed_out": result.timed_out,
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "time_ms": result.time_ms,
        "results": output_checks,
        "error_type": error_type,
        "test_source": "execution_contract",
        "execution_contract": contract,
    }


def full_code_runner(state: FinalVerifierState) -> dict[str, Any]:
    final_code = state.get("final_code", "")
    task_plan_ir = state.get("task_plan_ir", {})
    problem_id = state.get("problem_id")

    validation = task_plan_ir.get("validation") or {}
    sample_cases = validation.get("sample_test_cases", [])

    constraints = task_plan_ir.get("problem", {}).get("constraints", {}) or {}
    time_limit = _parse_time_limit(constraints)
    executor = SandboxExecutor(default_timeout_s=time_limit + 3.0)
    execution_contract = normalize_execution_contract(
        task_plan_ir.get("execution_contract")
    )

    if execution_contract["io_mode"] == "file":
        return {
            "execution_result": _run_file_contract(
                final_code, task_plan_ir, executor
            )
        }

    if not sample_cases:
        result = executor.run(final_code)
        execution_result: dict[str, Any] = {
            "ran_tests": False,
            "all_passed": not (result.timed_out or result.exit_code != 0),
            "timed_out": result.timed_out,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "time_ms": result.time_ms,
            "results": [],
            "error_type": _classify_error(result),
        }
        return {"execution_result": execution_result}

    results: list[dict] = []
    all_passed = True
    timed_out_global = False
    max_time_ms = 0.0
    first_stderr = ""

    for i, case in enumerate(sample_cases):
        test_input = case.get("input", "")
        expected = str(case.get("expected_output", case.get("output", ""))).strip()
        stdin_data = test_input if isinstance(test_input, str) else ""

        exec_result = executor.run(final_code, stdin=stdin_data)
        actual = exec_result.stdout.strip()
        judge = judge_output(
            problem_id=problem_id,
            input_data=stdin_data,
            expected=expected,
            actual=actual,
        )
        passed = (
            judge["passed"]
            and not exec_result.timed_out
            and exec_result.exit_code == 0
        )

        if not passed:
            all_passed = False
        if exec_result.timed_out:
            timed_out_global = True
        if exec_result.time_ms > max_time_ms:
            max_time_ms = exec_result.time_ms
        if exec_result.stderr and not first_stderr:
            first_stderr = exec_result.stderr[:1000]

        results.append({
            "case_index": i,
            "input": stdin_data,
            "passed": passed,
            "expected": expected,
            "actual": actual,
            "time_ms": exec_result.time_ms,
            "stderr": exec_result.stderr[:500] if exec_result.stderr else "",
            "timed_out": exec_result.timed_out,
            "exit_code": exec_result.exit_code,
            "judge_mode": judge["judge_mode"],
            "judge_detail": judge["judge_detail"],
        })

    execution_result = {
        "ran_tests": True,
        "all_passed": all_passed,
        "timed_out": timed_out_global,
        "exit_code": 0 if all_passed else -1,
        "stdout": "",
        "stderr": first_stderr,
        "time_ms": max_time_ms,
        "results": results,
        "error_type": _classify_results(results, timed_out_global),
        "test_source": (
            "dataset.input_output"
            if sample_cases
            and all(
                case.get("test_source") == "dataset.input_output"
                for case in sample_cases
            )
            else "task_plan_ir.validation.sample_test_cases"
        ),
    }
    return {"execution_result": execution_result}


def post_repair_code_runner(state: FinalVerifierState) -> dict[str, Any]:
    """Re-run the repaired program while retaining the pre-repair outcome."""
    initial_execution_result = state.get("execution_result", {})
    repaired_result = full_code_runner(state).get("execution_result", {})
    return {
        "initial_execution_result": initial_execution_result,
        "post_repair_execution_result": repaired_result,
        "execution_result": repaired_result,
    }


def _classify_error(result: Any) -> str:
    if result.timed_out:
        return "timeout"
    if result.exit_code != 0:
        return "runtime_error"
    return "none"


def _classify_results(results: list[dict], timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    for r in results:
        if r.get("exit_code", 0) != 0:
            return "runtime_error"
    for r in results:
        if not r.get("passed", True):
            return "wrong_answer"
    return "none"
