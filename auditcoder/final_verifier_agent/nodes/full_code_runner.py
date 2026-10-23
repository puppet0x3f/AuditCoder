"""Node: full_code_runner — run the assembled final_code against all validation test cases.

Uses SandboxExecutor to execute the complete solution and records the outcome
in execution_result, capturing timing, stdout/stderr, and timeout status.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from auditcoder.common.execution_contract import (
    normalize_execution_contract,
    resolve_required_input_paths,
    resolve_required_output_paths,
)
from auditcoder.common.callable_execution import (
    build_callable_driver,
    callable_arguments,
    callable_stdin,
)
from auditcoder.common.output_judging import judge_output
from auditcoder.final_verifier_agent.audit import (
    build_owner_map,
    frozen_complement_ok,
    make_snapshot,
    rank_execution,
    repair_event,
)
from auditcoder.verifier_agent.sandbox import SandboxExecutor
from auditcoder.final_verifier_agent.state import FinalVerifierState
from auditcoder.schemas.contract_graph import (
    ensure_catg,
    register_code_owner,
    register_assembly_units,
)


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
    task_plan_ir = ensure_catg(
        state.get("task_plan_ir", {}),
        ownership_map=state.get("registered_units") or state.get("ownership_map"),
    )
    for task_id, source in (state.get("code_snippets") or {}).items():
        task_plan_ir = register_code_owner(task_plan_ir, str(task_id), str(source))
    task_plan_ir = register_assembly_units(task_plan_ir, final_code)
    problem_id = state.get("problem_id")

    validation = task_plan_ir.get("validation") or {}
    external_cases = [
        dict(case)
        for case in (state.get("external_test_cases") or [])
        if isinstance(case, dict)
    ]
    # Q is materialized once by CATG: examples first, then planner-derived
    # boundary checks. Dataset-backed cases are authoritative; planner-derived
    # boundary checks are advisory because their expected outputs come from an
    # LLM and may be inconsistent with the original specification.
    development_checks = task_plan_ir.get("development_checks") or []
    raw_planner_cases = development_checks or validation.get("sample_test_cases", [])
    planner_cases = [
        {
            "input": check.get("input", ""),
            "expected_output": check.get(
                "expected_output", check.get("output", "")
            ),
            "check_id": check.get("id") or check.get("check_id"),
            "test_source": check.get(
                "source", "task_plan_ir.development_checks"
            ),
            "trusted": False,
        }
        for check in raw_planner_cases
        if isinstance(check, dict)
    ]
    if external_cases:
        sample_cases = [
            {
                **case,
                "trusted": bool(case.get("trusted", True)),
                "test_source": case.get("test_source", "dataset.input_output"),
            }
            for case in external_cases
        ] + planner_cases
    else:
        # Direct/offline callers have no benchmark oracle, so preserve the
        # historical behavior and treat their available development checks as
        # authoritative.
        sample_cases = [dict(case, trusted=True) for case in planner_cases]

    constraints = task_plan_ir.get("problem", {}).get("constraints", {}) or {}
    time_limit = _parse_time_limit(constraints)
    executor = SandboxExecutor(default_timeout_s=time_limit + 3.0)
    execution_contract = normalize_execution_contract(
        task_plan_ir.get("execution_contract")
    )

    if execution_contract["io_mode"] == "file":
        execution_result = _run_file_contract(final_code, task_plan_ir, executor)
        return _audit_state_update(state, task_plan_ir, execution_result)

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
        return _audit_state_update(state, task_plan_ir, execution_result)

    results: list[dict] = []
    all_passed = True
    timed_out_global = False
    max_time_ms = 0.0
    first_stderr = ""
    trusted_failures: list[dict[str, Any]] = []
    advisory_failures: list[dict[str, Any]] = []
    trusted_case_count = sum(bool(case.get("trusted", True)) for case in sample_cases)

    for i, case in enumerate(sample_cases):
        test_input = case.get("input", "")
        expected = str(case.get("expected_output", case.get("output", ""))).strip()
        stdin_data = test_input if isinstance(test_input, str) else ""

        callable_name = str(case.get("callable_name") or "")
        callable_args = case.get("callable_args")
        if callable_name and not isinstance(callable_args, list):
            callable_args = callable_arguments(test_input)
        source_for_case = final_code
        if callable_name and "class Solution" in final_code:
            source_for_case = build_callable_driver(
                final_code, callable_name, callable_args or []
            )
            stdin_data = ""
        elif callable_name:
            stdin_data = callable_stdin(callable_args or callable_arguments(test_input))

        exec_result = executor.run(source_for_case, stdin=stdin_data)
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

        trusted = bool(case.get("trusted", True))
        if not passed:
            if trusted:
                all_passed = False
            else:
                advisory_failures.append(
                    {
                        "case_index": i,
                        "check_id": case.get("check_id"),
                        "input": stdin_data,
                        "expected": expected,
                        "actual": actual,
                    }
                )
            if trusted:
                trusted_failures.append(
                    {
                        "case_index": i,
                        "check_id": case.get("check_id"),
                        "input": stdin_data,
                        "expected": expected,
                        "actual": actual,
                    }
                )
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
            "trusted": trusted,
            "advisory": not trusted,
            "symptom": _status_for_case(
                passed=passed,
                timed_out=exec_result.timed_out,
                exit_code=exec_result.exit_code,
                actual=actual,
                expected=expected,
            ),
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
        "error_type": (
            _classify_results(results, timed_out_global)
            if trusted_failures
            else ("advisory_mismatch" if advisory_failures else "none")
        ),
        "trusted_test_count": trusted_case_count,
        "trusted_failures": trusted_failures,
        "advisory_failures": advisory_failures,
        "test_source": (
            "dataset.input_output+task_plan_ir.development_checks"
            if external_cases
            else "task_plan_ir.validation.sample_test_cases"
        ),
    }
    return _audit_state_update(state, task_plan_ir, execution_result)


def _audit_state_update(
    state: FinalVerifierState,
    task_plan_ir: dict[str, Any],
    execution_result: dict[str, Any],
) -> dict[str, Any]:
    """Return runner output plus the CATG state required by later nodes."""

    enriched = _enrich_execution_result(execution_result)
    snippets = state.get("code_snippets") or {}
    owner_map = build_owner_map(
        task_plan_ir,
        snippets,
        str(state.get("final_code", "") or ""),
        task_plan_ir.get("ownership_map"),
    )
    # Dispatcher-local evidence lives on the serialized IR, while callers of
    # the final verifier may also pass an already accumulated state list.
    # Carry both forward and de-duplicate by record ID so Local localization
    # can actually consume verifier evidence produced before assembly.
    records: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    for source in (task_plan_ir.get("evidence_records") or [], state.get("evidence_records") or []):
        for item in source:
            if not isinstance(item, dict):
                continue
            identifier = str(item.get("id") or "")
            key = identifier or repr(sorted(item.items()))
            if key in seen_record_ids:
                continue
            seen_record_ids.add(key)
            records.append(dict(item))
    records.append(
        {
            "id": f"global:{len(records)}",
            "owners": [],
            "version": dict(state.get("audit_versions") or {}),
            "verdict": "Pass" if enriched.get("all_passed") else "Reject",
            "obligation": None,
            "witness": {
                "checks": enriched.get("ordered_checks", []),
                "error_type": enriched.get("error_type", "none"),
            },
            "kind": "exec",
            "source": "global_execution",
            "current": True,
        }
    )
    task_plan_ir["evidence_records"] = copy.deepcopy(records)
    graph = task_plan_ir.get("contract_annotated_task_graph") or task_plan_ir.get("contract_graph") or {}
    return {
        "execution_result": enriched,
        "task_plan_ir": task_plan_ir,
        "contract_graph": graph,
        "contract_annotated_task_graph": graph,
        "responsibility_records": task_plan_ir.get("responsibility_records", []),
        "development_checks": task_plan_ir.get("development_checks", []),
        "registered_units": task_plan_ir.get("ownership_map", {"units": []}),
        "ownership_map": owner_map,
        "evidence_records": records,
        "audit_trace": {
            "G": graph,
            "S": task_plan_ir.get("responsibility_records", []),
            "E": records,
            "H": list(state.get("repair_history") or []),
        },
    }


def _append_repair_to_records(
    task_plan_ir: dict[str, Any],
    event: dict[str, Any],
    scope: dict[str, Any] | None,
    location: dict[str, Any] | None,
) -> dict[str, Any]:
    """Append one intervention to the affected responsibility records."""

    ir = ensure_catg(task_plan_ir)
    scope = scope or {}
    affected = {
        str(identifier)
        for identifier in (scope.get("region") or [])
        if str(identifier)
    }
    if not affected:
        affected.update(
            str(identifier)
            for identifier in (scope.get("nominal_region") or [])
            if str(identifier)
        )
    if not affected and isinstance(location, dict):
        owners = location.get("owners", location.get("implicated_nodes", []))
        if isinstance(owners, str):
            owners = [owners]
        affected.update(str(identifier) for identifier in (owners or []) if str(identifier))
    if not affected:
        return ir
    records = []
    for raw in ir.get("responsibility_records", []) or []:
        record = dict(raw) if isinstance(raw, dict) else {}
        if str(record.get("id", "")) in affected:
            history = list(record.get("repair_history") or [])
            history.append(copy.deepcopy(event))
            record["repair_history"] = history
        records.append(record)
    ir["responsibility_records"] = records
    # Rebuild the graph so serialized S and G expose the same intervention
    # record while preserving the active ownership map and evidence history.
    return ensure_catg(ir, ownership_map=ir.get("ownership_map"))


def post_repair_code_runner(state: FinalVerifierState) -> dict[str, Any]:
    """Execute, rank, and commit/rollback one bounded repair transaction."""

    snapshot = state.get("repair_snapshot") or make_snapshot(state)
    initial_execution_result = snapshot.get(
        "execution_result", state.get("execution_result", {})
    )
    runner_output = full_code_runner(state)
    candidate_result = runner_output.get("execution_result", {})
    candidate_ir = runner_output.get("task_plan_ir", state.get("task_plan_ir", {}))
    scope = state.get("repair_scope") or {}
    region = set(scope.get("region") or [])
    before_snippets = snapshot.get("code_snippets") or {}
    candidate_snippets = state.get("code_snippets") or {}
    boundary_ok, frozen_violations = frozen_complement_ok(
        before_snippets, candidate_snippets, region
    )
    assembly_boundary = state.get("repair_boundary_check") or {}
    declared_boundary_violations = list(
        assembly_boundary.get("frozen_violations") or []
    )
    if declared_boundary_violations:
        boundary_ok = False
        frozen_violations = sorted(
            set(frozen_violations) | set(map(str, declared_boundary_violations))
        )
    before_rank = rank_execution(
        initial_execution_result, snapshot.get("verification_reports") or {}
    )
    candidate_rank = rank_execution(
        candidate_result, state.get("verification_reports") or {}
    )
    rank_improved = candidate_rank > before_rank
    accepted = boundary_ok and rank_improved
    attempt = int(state.get("repair_attempts", 0) or 0) + 1

    candidate_state = {
        "final_code": state.get("final_code", ""),
        "code_snippets": candidate_snippets,
        "task_plan_ir": candidate_ir,
        "verification_reports": state.get("verification_reports", {}),
        "execution_result": candidate_result,
    }
    event = repair_event(
        location=state.get("error_location"),
        scope=scope,
        before_state=snapshot,
        candidate_state=candidate_state,
        candidate_execution=candidate_result,
        decision="Accept" if accepted else "Reject",
        rollback_status="NotApplicable" if accepted else "Restored",
        attempt=attempt,
        reason=(
            "strict_rank_improvement"
            if accepted
            else (
                "frozen_complement_violation"
                if not boundary_ok
                else "candidate_rank_not_improved"
            )
        ),
    )
    history = list(state.get("repair_history") or [])
    history.append(event)

    if accepted:
        candidate_ir = _append_repair_to_records(candidate_ir, event, scope, state.get("error_location"))
        versions = dict(state.get("audit_versions") or {})
        changed_ids = set()
        for identifier in set(before_snippets) | set(candidate_snippets):
            if str(before_snippets.get(identifier, "")) != str(
                candidate_snippets.get(identifier, "")
            ):
                changed_ids.add(str(identifier))
        for identifier in changed_ids:
            versions[identifier] = int(versions.get(identifier, 0) or 0) + 1
        return {
            "initial_execution_result": initial_execution_result,
            "post_repair_execution_result": candidate_result,
            "execution_result": candidate_result,
            "final_code": state.get("final_code", ""),
            "code_snippets": dict(candidate_snippets),
            "task_plan_ir": candidate_ir,
            "audit_versions": versions,
            "ownership_map": build_owner_map(
                candidate_ir, candidate_snippets, state.get("final_code", "")
            ),
            "repair_attempts": attempt,
            "repair_history": history,
            "repair_event": event,
            "repair_decision": "Accept",
            "rollback_status": "NotApplicable",
            "contract_graph": candidate_ir.get("contract_annotated_task_graph", candidate_ir.get("contract_graph", {})),
            "contract_annotated_task_graph": candidate_ir.get("contract_annotated_task_graph", candidate_ir.get("contract_graph", {})),
            "responsibility_records": candidate_ir.get("responsibility_records", []),
            "development_checks": candidate_ir.get("development_checks", []),
            "registered_units": candidate_ir.get("ownership_map", {"units": []}),
            "evidence_records": runner_output.get("evidence_records", state.get("evidence_records", [])),
            "audit_trace": {
                "G": candidate_ir.get("contract_annotated_task_graph", candidate_ir.get("contract_graph", {})),
                "S": candidate_ir.get("responsibility_records", []),
                "E": runner_output.get("evidence_records", state.get("evidence_records", [])),
                "H": history,
            },
        }

    # Restore every active snapshot field.  Candidate evidence/results remain
    # in the append-only event but cannot influence the next Locate call.
    restored_ir = _append_repair_to_records(
        snapshot.get("task_plan_ir", state.get("task_plan_ir", {})),
        event,
        scope,
        state.get("error_location"),
    )
    return {
        "initial_execution_result": initial_execution_result,
        "post_repair_execution_result": candidate_result,
        "execution_result": initial_execution_result,
        "final_code": snapshot.get("final_code", state.get("final_code", "")),
        "code_snippets": dict(before_snippets),
        "task_plan_ir": restored_ir,
        "verification_reports": snapshot.get(
            "verification_reports", state.get("verification_reports", {})
        ),
        "audit_versions": snapshot.get("audit_versions", state.get("audit_versions", {})),
        "ownership_map": restored_ir.get("ownership_map", state.get("ownership_map", {})),
        "repair_attempts": attempt,
        "repair_history": history,
        "repair_event": event,
        "repair_decision": "Reject",
        "rollback_status": "Restored",
        "contract_graph": restored_ir.get("contract_annotated_task_graph", restored_ir.get("contract_graph", {})),
        "contract_annotated_task_graph": restored_ir.get("contract_annotated_task_graph", restored_ir.get("contract_graph", {})),
        "responsibility_records": restored_ir.get("responsibility_records", []),
        "development_checks": restored_ir.get(
            "development_checks", snapshot.get("development_checks", state.get("development_checks", []))
        ),
        "registered_units": snapshot.get(
            "registered_units", state.get("registered_units", {"units": []})
        ),
        "evidence_records": runner_output.get(
            "evidence_records", state.get("evidence_records", [])
        ),
        "audit_trace": {
            "G": restored_ir.get("contract_annotated_task_graph", restored_ir.get("contract_graph", {})),
            "S": restored_ir.get("responsibility_records", []),
            "E": runner_output.get("evidence_records", state.get("evidence_records", [])),
            "H": history,
        },
    }


def _status_for_case(
    *,
    passed: bool,
    timed_out: bool,
    exit_code: int | None,
    actual: str = "",
    expected: str = "",
) -> str:
    """Map one ordered Q check to the paper's execution symptom alphabet."""

    if timed_out:
        return "TLE"
    if exit_code not in (None, 0):
        return "RE"
    if not passed:
        return "WA"
    return "Pass"


def _enrich_execution_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy runner output into an auditable global record."""

    enriched = dict(result)
    checks = enriched.get("results") or []
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(checks):
        check = dict(raw) if isinstance(raw, dict) else {"value": raw}
        check.setdefault("case_index", index)
        check.setdefault(
            "symptom",
            _status_for_case(
                passed=bool(check.get("passed", False)),
                timed_out=bool(check.get("timed_out", False)),
                exit_code=check.get("exit_code", 0),
                actual=str(check.get("actual", "")),
                expected=str(check.get("expected", "")),
            ),
        )
        normalized.append(check)
    enriched["results"] = normalized
    enriched["ordered_checks"] = normalized
    enriched["anchor_case_index"] = next(
        (check["case_index"] for check in normalized if not check.get("passed", False)),
        None,
    )
    # A global record is useful even when the runner has no per-owner probe;
    # the Resource rule intentionally ignores it unless owner_measurements is
    # supplied by an instrumented executor.
    enriched.setdefault("evidence_kind", "exec")
    enriched.setdefault("version", {})
    return enriched


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
