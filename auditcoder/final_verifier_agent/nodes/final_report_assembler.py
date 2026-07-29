"""Node: final_report_assembler — assemble the final pipeline verdict report."""

from __future__ import annotations

from typing import Any

from auditcoder.final_verifier_agent.state import FinalVerifierState


def final_report_assembler(state: FinalVerifierState) -> dict[str, Any]:
    execution_result = state.get("execution_result", {})
    error_location = state.get("error_location")
    localization_record = state.get("localization_record")
    verification_reports = state.get("verification_reports", {})

    rejected_task_ids = [
        task_id
        for task_id, report in verification_reports.items()
        if report.get("verdict") == "reject"
    ]
    execution_passed = execution_result.get("all_passed", False)
    all_passed = execution_passed and not rejected_task_ids
    verdict = "accept" if all_passed else "reject"

    final_report: dict[str, Any] = {
        "verdict": verdict,
        "all_passed": all_passed,
        "error_type": (
            "rejected_leaf_code"
            if execution_passed and rejected_task_ids
            else execution_result.get("error_type", "none")
        ),
        "time_ms": execution_result.get("time_ms", 0.0),
        "error_location": error_location,
        "localization_record": localization_record,
        "per_task_reports": {
            task_id: {
                "verdict": r.get("verdict", "unknown"),
                "summary": r.get("summary", ""),
            }
            for task_id, r in verification_reports.items()
        },
        "test_results": execution_result.get("results", []),
        "rejected_task_ids": rejected_task_ids,
    }

    if error_location:
        final_report["responsible_agent"] = {
            "task_id": error_location.get("task_id", "unknown"),
            "agent_type": error_location.get("agent_type", "unknown"),
            "error_type": error_location.get("error_type", "unknown"),
            "detail": error_location.get("detail", ""),
            "location_method": error_location.get("method", "unknown"),
        }

    retried_tasks = state.get("retried_tasks") or []
    if retried_tasks:
        initial_execution = state.get("initial_execution_result", {})
        post_repair_execution = state.get("post_repair_execution_result", {})
        final_report["retry_info"] = {
            "retried_task_ids": retried_tasks,
            "retry_attempted": True,
            "global_validation_after_repair": post_repair_execution.get(
                "all_passed", False
            ),
        }
        final_report["repair_history"] = [
            {
                "repair_attempt": 1,
                "localization": localization_record,
                "repair_region": {
                    "region_type": "node",
                    "nodes": retried_tasks,
                },
                "initial_validation_passed": initial_execution.get(
                    "all_passed", False
                ),
                "post_repair_validation_passed": post_repair_execution.get(
                    "all_passed", False
                ),
            }
        ]

    return {"final_report": final_report}
