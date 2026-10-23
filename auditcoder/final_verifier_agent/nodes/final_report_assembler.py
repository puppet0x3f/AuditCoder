"""Assemble the final verdict and the append-only repair trace."""

from __future__ import annotations

from typing import Any

from auditcoder.final_verifier_agent.audit import rank_execution
from auditcoder.final_verifier_agent.state import FinalVerifierState
from auditcoder.schemas.contract_graph import catg_summary


def _legacy_history(state: FinalVerifierState, localization: Any) -> list[dict[str, Any]]:
    """Keep the compact pre-paper shape for callers that only provide lists."""

    retried_tasks = state.get("retried_tasks") or []
    if not retried_tasks:
        return []
    initial_execution = state.get("initial_execution_result", {}) or {}
    post_repair_execution = state.get("post_repair_execution_result", {}) or {}
    return [
        {
            "repair_attempt": 1,
            "localization": localization,
            "repair_region": {"region_type": "node", "nodes": retried_tasks},
            "initial_validation_passed": initial_execution.get("all_passed", False),
            "post_repair_validation_passed": post_repair_execution.get(
                "all_passed", False
            ),
        }
    ]


def _accepted_repair_nodes(history: list[dict[str, Any]]) -> set[str]:
    """Return responsibility IDs superseded by an accepted transaction.

    Verification reports describe the pre-repair candidate.  Once a bounded
    transaction is accepted by the global rank and boundary checks, its old
    rejection must not keep the final report at ``reject``.  The report keeps
    those IDs separately for auditability; only the active rejection set is
    used for the final verdict.
    """

    repaired: set[str] = set()
    for event in history:
        if str(event.get("decision", "")).lower() != "accept":
            continue
        region = event.get("repair_region") or {}
        nodes = region.get("nodes") if isinstance(region, dict) else None
        if nodes is None:
            nodes = event.get("nominal_region") or []
        if isinstance(nodes, str):
            nodes = [nodes]
        repaired.update(str(identifier) for identifier in (nodes or []) if str(identifier))
    return repaired


def final_report_assembler(state: FinalVerifierState) -> dict[str, Any]:
    execution_result = state.get("execution_result", {}) or {}
    error_location = state.get("error_location")
    localization_record = state.get("localization_record")
    verification_reports = state.get("verification_reports", {}) or {}

    raw_rejected_task_ids = [
        str(task_id)
        for task_id, report in verification_reports.items()
        if isinstance(report, dict)
        and str(report.get("verdict", "")).lower() in {"reject", "rejected", "fail", "failed"}
    ]
    history = list(state.get("repair_history") or [])
    superseded_rejected_task_ids = sorted(
        set(raw_rejected_task_ids) & _accepted_repair_nodes(history)
    )
    rejected_task_ids = [
        task_id
        for task_id in raw_rejected_task_ids
        if task_id not in superseded_rejected_task_ids
    ]
    execution_passed = bool(execution_result.get("all_passed", False))
    # Dataset-backed cases are the authoritative end-to-end oracle.  CATG
    # development checks and leaf reports are advisory in that mode: a helper
    # may reject a planner-generated edge case while the assembled program
    # passes every trusted benchmark case.  Do not turn that evidence
    # mismatch into a false whole-program rejection.  Offline callers without
    # trusted cases retain the stricter historical behavior.
    benchmark_backed = execution_result.get("trusted_test_count") is not None
    verification_only_rejections = (
        rejected_task_ids if benchmark_backed and execution_passed else []
    )
    all_passed = execution_passed and (
        benchmark_backed or not rejected_task_ids
    )
    verdict = "accept" if all_passed else "reject"

    if not history:
        history = _legacy_history(state, localization_record)

    final_report: dict[str, Any] = {
        "verdict": verdict,
        "all_passed": all_passed,
        "error_type": (
            "advisory_mismatch"
            if execution_passed and (rejected_task_ids or execution_result.get("advisory_failures"))
            else "rejected_leaf_code"
            if execution_passed and rejected_task_ids
            else execution_result.get("error_type", "none")
        ),
        "time_ms": execution_result.get("time_ms", 0.0),
        "error_location": error_location,
        "localization_record": localization_record,
        "scope_record": state.get("scope_record"),
        "selected_scope": state.get("repair_scope"),
        "test_results": execution_result.get("results", []),
        "trusted_test_count": execution_result.get("trusted_test_count"),
        "trusted_failures": execution_result.get("trusted_failures", []),
        "advisory_failures": execution_result.get("advisory_failures", []),
        "rejected_task_ids": rejected_task_ids,
        "verification_only_rejections": verification_only_rejections,
        "raw_rejected_task_ids": raw_rejected_task_ids,
        "superseded_rejected_task_ids": superseded_rejected_task_ids,
        "repair_history": history,
        "audit_versions": state.get("audit_versions", {}) or {},
        "active_rank": list(rank_execution(execution_result, verification_reports)),
        "per_task_reports": {
            task_id: {
                "verdict": r.get("verdict", "unknown"),
                "summary": r.get("summary", ""),
            }
            for task_id, r in verification_reports.items()
            if isinstance(r, dict)
        },
        "catg": catg_summary(state.get("task_plan_ir", {})),
        "responsibility_records": state.get("responsibility_records", []),
        "evidence_record_count": len(state.get("evidence_records", []) or []),
        "audit_trace": state.get("audit_trace", {}),
    }

    if error_location:
        final_report["responsible_agent"] = {
            "task_id": error_location.get("task_id"),
            "owners": error_location.get("owners", []),
            "agent_type": error_location.get("agent_type", "unknown"),
            "error_type": error_location.get("error_type", "unknown"),
            "detail": error_location.get("detail", ""),
            "location_method": error_location.get(
                "method", error_location.get("rule", "unknown")
            ),
            "evidence_grade": error_location.get("confidence", "unknown"),
        }

    retried_tasks = state.get("retried_tasks") or []
    if retried_tasks or history:
        initial_execution = state.get("initial_execution_result", {}) or {}
        post_repair_execution = state.get("post_repair_execution_result", {}) or {}
        final_report["retry_info"] = {
            "retried_task_ids": retried_tasks,
            "repair_attempt_count": int(state.get("repair_attempts", len(history)) or 0),
            "max_repair_attempts": int(state.get("max_repair_attempts", 3) or 0),
            "retry_attempted": bool(history),
            "global_validation_after_repair": post_repair_execution.get(
                "all_passed", execution_passed
            ),
            "last_decision": state.get("repair_decision"),
            "rollback_status": state.get("rollback_status"),
        }
        # The old fields remain useful to existing result consumers.
        if history:
            final_report["initial_validation_passed"] = initial_execution.get(
                "all_passed", False
            )
            final_report["post_repair_validation_passed"] = post_repair_execution.get(
                "all_passed", False
            )

    return {"final_report": final_report}


__all__ = ["final_report_assembler"]
