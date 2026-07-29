"""Node: error_locator — pinpoint which task/agent caused the final code failure.

Strategy:
1. Parse traceback in stderr to find the offending function name, then
   cross-reference with task_plan_ir.task_nodes[].interface.function_name.
2. For timeout errors, attribute only a snippet whose isolated import also
   times out; importing function definitions is not a valid performance rank.
3. Fall back to a unique rejected per-task verification report.
4. Abstain when evidence is absent or ambiguous.
"""

from __future__ import annotations

import re
from typing import Any

from auditcoder.verifier_agent.sandbox import SandboxExecutor
from auditcoder.final_verifier_agent.state import FinalVerifierState


def _build_interface_map(
    task_plan_ir: dict[str, Any],
) -> dict[str, list[dict[str, str]]]:
    """Return every leaf owner for each function name.

    Keeping all owners is essential for auditability: silently overwriting a
    duplicate function name would turn an ambiguous mapping into false
    high-confidence localization.
    """
    mapping: dict[str, list[dict[str, str]]] = {}
    for node in task_plan_ir.get("task_nodes", []):
        if not node.get("is_leaf"):
            continue
        iface = node.get("interface") or {}
        fn = iface.get("function_name", "")
        if fn:
            mapping.setdefault(fn, []).append({
                "task_id": node.get("id", ""),
                "agent_type": node.get("agent_type", ""),
            })
    return mapping


def _locate_by_traceback(
    stderr: str,
    interface_map: dict[str, list[dict[str, str]]],
) -> dict[str, str] | None:
    """Extract a function from a traceback only when it has one owner."""
    # Collect all function names mentioned in traceback frames
    frame_funcs = re.findall(r'in\s+(\w+)', stderr)
    # Search in reverse (innermost frame first)
    for fn in reversed(frame_funcs):
        owners = interface_map.get(fn, [])
        if len(owners) == 1:
            info = owners[0]
            return {
                "task_id": info["task_id"],
                "agent_type": info["agent_type"],
                "function_name": fn,
                "method": "traceback",
                "confidence": "high",
            }
        if owners:
            # The innermost registered frame is already ambiguous; selecting an
            # outer frame would hide that ambiguity rather than resolve it.
            return None
    return None


def _locate_by_timing(
    code_snippets: dict[str, str],
    task_plan_ir: dict[str, Any],
    time_limit_s: float = 2.0,
) -> dict[str, str] | None:
    """Return a snippet only if its isolated import itself times out.

    The snippets normally contain function definitions but no representative
    invocation.  Small differences in import time therefore cannot support a
    responsibility claim for an end-to-end timeout.
    """
    executor = SandboxExecutor(default_timeout_s=time_limit_s + 1.0)
    node_map = {n["id"]: n for n in task_plan_ir.get("task_nodes", [])}

    for task_id, snippet in code_snippets.items():
        # Wrap snippet in a minimal driver that just imports it
        driver = f"{snippet}\nprint('__ok__')\n"
        result = executor.run(driver)
        if result.timed_out:
            node = node_map.get(task_id, {})
            return {
                "task_id": task_id,
                "agent_type": node.get("agent_type", ""),
                "method": "isolated_import_timeout",
                "confidence": "medium",
            }
    return None


def _locate_by_verification_reports(
    verification_reports: dict[str, dict[str, Any]],
) -> dict[str, str] | None:
    """Use a local rejection only when it identifies a unique candidate."""
    rejected = [
        (task_id, report)
        for task_id, report in verification_reports.items()
        if report.get("verdict") == "reject"
    ]
    if len(rejected) != 1:
        return None
    task_id, report = rejected[0]
    return {
        "task_id": task_id,
        "agent_type": report.get("agent_type", ""),
        "method": "verification_report",
        "confidence": "low",
    }


def error_locator(state: FinalVerifierState) -> dict[str, Any]:
    execution_result = state.get("execution_result", {})
    error_type = execution_result.get("error_type", "none")

    if error_type == "none":
        return {
            "error_location": None,
            "localization_record": {
                "status": "not_needed",
                "method": None,
                "error_type": "none",
            },
        }

    task_plan_ir = state.get("task_plan_ir", {})
    code_snippets = state.get("code_snippets", {})
    verification_reports = state.get("verification_reports", {})

    interface_map = _build_interface_map(task_plan_ir)

    constraints = task_plan_ir.get("problem", {}).get("constraints", {}) or {}
    try:
        time_limit_s = float(str(constraints.get("time_limit", "2s")).replace("s", "").strip())
    except (ValueError, AttributeError):
        time_limit_s = 2.0

    location: dict[str, Any] | None = None

    if error_type == "timeout":
        location = _locate_by_timing(code_snippets, task_plan_ir, time_limit_s)
    elif error_type in ("runtime_error", "wrong_answer", "contract_error"):
        stderr = execution_result.get("stderr", "")
        location = _locate_by_traceback(stderr, interface_map)

    # Supplement with failing test case details
    if location is None:
        location = _locate_by_verification_reports(verification_reports)

    if location is None:
        rejected_ids = [
            task_id
            for task_id, report in verification_reports.items()
            if report.get("verdict") == "reject"
        ]
        duplicate_functions = sorted(
            fn for fn, owners in interface_map.items() if len(owners) > 1
        )
        return {
            "error_location": None,
            "localization_record": {
                "status": "abstained",
                "method": "abstention",
                "error_type": error_type,
                "reason": "no_unique_evidence_supported_owner",
                "rejected_task_ids": rejected_ids,
                "ambiguous_function_names": duplicate_functions,
            },
        }

    # Enrich with error_type and detail from execution result
    failing_case: dict[str, Any] = {}
    for r in execution_result.get("results", []):
        if not r.get("passed", True):
            failing_case = r
            break

    location["error_type"] = error_type
    if not location.get("agent_type"):
        node = next(
            (
                candidate
                for candidate in task_plan_ir.get("task_nodes", [])
                if candidate.get("id") == location.get("task_id")
            ),
            {},
        )
        location["agent_type"] = node.get("agent_type", "")
    location["detail"] = (
        execution_result.get("stderr", "")[:500]
        or str(failing_case.get("actual", ""))[:200]
    )
    if failing_case:
        location["failing_case"] = {
            "case_index": failing_case.get("case_index"),
            "input": failing_case.get("input", ""),
            "expected": failing_case.get("expected", ""),
            "actual": failing_case.get("actual", ""),
            "stderr": failing_case.get("stderr", ""),
            "timed_out": failing_case.get("timed_out", False),
            "exit_code": failing_case.get("exit_code"),
            "time_ms": failing_case.get("time_ms"),
        }

    return {
        "error_location": location,
        "localization_record": {
            "status": "localized",
            **location,
        },
    }
