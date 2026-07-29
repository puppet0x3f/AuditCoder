"""Node: combine compile and unittest reports into a final verification_report."""

from __future__ import annotations

from typing import Any

from auditcoder_class.class_verifier_agent.state import ClassVerifierState


def verdict_assembler(state: ClassVerifierState) -> dict[str, Any]:
    method_name = state.get("method_name", "?")
    compile_report = state.get("compile_report", {})
    unittest_report = state.get("unittest_report", {})

    if not compile_report.get("compile_ok", True):
        verdict = "reject"
        summary = f"Compile error: {compile_report.get('error', '')}"
    elif not unittest_report.get("passed", True):
        f = unittest_report.get("failures", 0)
        e = unittest_report.get("errors", 0)
        verdict = "reject"
        summary = f"Unittest failed: {f} failure(s), {e} error(s)"
    else:
        verdict = "accept"
        summary = f"All {unittest_report.get('test_count', 0)} tests passed"

    return {
        "verification_report": {
            "method_name": method_name,
            "verdict": verdict,
            "summary": summary,
            "compile_ok": compile_report.get("compile_ok", True),
            "unittest_report": unittest_report,
        }
    }
