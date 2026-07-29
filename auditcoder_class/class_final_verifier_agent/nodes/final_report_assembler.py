"""Node: produce the final report."""

from __future__ import annotations

from typing import Any

from auditcoder_class.class_final_verifier_agent.state import ClassFinalVerifierState


def final_report_assembler(state: ClassFinalVerifierState) -> dict[str, Any]:
    exec_result = state.get("execution_result", {})
    error_location = state.get("error_location")
    retried = state.get("retried_methods", [])

    verdict = "accept" if exec_result.get("all_passed", False) else "reject"

    return {
        "final_report": {
            "verdict": verdict,
            "all_passed": exec_result.get("all_passed", False),
            "per_method_reports": state.get("per_method_reports", {}),
            "execution_result": exec_result,
            "error_location": error_location,
            "retry_info": {
                "retry_attempted": bool(retried),
                "retried_methods": retried,
            },
        }
    }
