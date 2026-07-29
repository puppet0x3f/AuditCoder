"""Node: locate the failing method from the execution_result traceback/output."""

from __future__ import annotations

import re
from typing import Any

from auditcoder_class.class_final_verifier_agent.state import ClassFinalVerifierState


def error_locator(state: ClassFinalVerifierState) -> dict[str, Any]:
    exec_result = state.get("execution_result", {})
    ir = state["class_plan_ir"]
    method_names = [m["method_name"] for m in ir.get("method_nodes", [])]
    test_classes = ir.get("test_classes", [])

    stderr = exec_result.get("stderr", "")
    stdout = exec_result.get("stdout", "")
    combined_output = stderr + "\n" + stdout

    # Strategy 0: structural violations detected by the assembler smoke check.
    # Missing @staticmethod/@classmethod causes every test for that method to ERROR
    # with a wrong-argument-count TypeError — catch it before traceback parsing.
    structural_violations: list[dict[str, Any]] = state.get("structural_violations", [])
    if structural_violations:
        v = structural_violations[0]
        method_name = v.get("method_name", "")
        if method_name in method_names:
            return {
                "error_location": {
                    "method_name": method_name,
                    "error_type": "structural_violation",
                    "detail": v.get("issue", ""),
                }
            }

    # Strategy 1: traceback — find "in method_name" references
    traceback_methods = re.findall(r'\bin\s+(\w+)', combined_output)
    for name in reversed(traceback_methods):  # innermost frame last
        if name in method_names:
            return {
                "error_location": {
                    "method_name": name,
                    "error_type": "traceback",
                    "detail": stderr[:1000],
                }
            }

    # Strategy 2: parse failing TestCase class name from unittest output, map to method
    # Build test_class → method_name mapping from ir
    tc_to_method = {}
    for m in ir.get("method_nodes", []):
        tc = m.get("test_class_name", "")
        if tc:
            tc_to_method[tc] = m["method_name"]

    for tc in test_classes:
        failed_tc = re.search(
            r'(?:FAIL|ERROR):\s+.*\b' + re.escape(tc) + r'\b',
            combined_output,
        )
        if failed_tc:
            method = tc_to_method.get(tc)
            if method:
                return {
                    "error_location": {
                        "method_name": method,
                        "error_type": "test_class",
                        "detail": combined_output[:1000],
                    }
                }

    # Fallback: return first method that had a rejection in per_method_reports
    per_method = state.get("per_method_reports", {})
    for name in ir.get("execution_order", []):
        r = per_method.get(name, {})
        if r.get("verdict") == "reject":
            return {
                "error_location": {
                    "method_name": name,
                    "error_type": "per_method_report",
                    "detail": r.get("summary", ""),
                }
            }

    return {"error_location": None}
