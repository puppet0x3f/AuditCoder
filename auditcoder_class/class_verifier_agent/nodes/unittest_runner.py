"""Node: run the per-method unittest TestCase via SandboxExecutor."""

from __future__ import annotations

import re
from typing import Any

from auditcoder.verifier_agent.sandbox.executor import SandboxExecutor
from auditcoder_class.class_verifier_agent.state import ClassVerifierState

_executor = SandboxExecutor(default_timeout_s=15.0)


def _parse_unittest_output(stdout: str) -> dict[str, Any]:
    """Parse TextTestRunner(verbosity=2) output to extract counts."""
    test_count = 0
    failures = 0
    errors = 0

    # Count individual test lines: "test_xxx ... ok" or "test_xxx ... FAIL"
    ok_matches = re.findall(r'\.\.\. ok', stdout)
    fail_matches = re.findall(r'\.\.\. FAIL', stdout)
    error_matches = re.findall(r'\.\.\. ERROR', stdout)
    test_count = len(ok_matches) + len(fail_matches) + len(error_matches)
    failures = len(fail_matches)
    errors = len(error_matches)

    # Also try to parse the summary line "Ran N test(s)"
    ran_match = re.search(r'Ran (\d+) test', stdout)
    if ran_match:
        test_count = int(ran_match.group(1))

    # "FAILED (failures=N, errors=M)" or "FAILED (failures=N)" etc.
    failed_summary = re.search(r'FAILED\s*\((.+?)\)', stdout)
    if failed_summary:
        f_match = re.search(r'failures=(\d+)', failed_summary.group(1))
        e_match = re.search(r'errors=(\d+)', failed_summary.group(1))
        if f_match:
            failures = int(f_match.group(1))
        if e_match:
            errors = int(e_match.group(1))

    passed = failures == 0 and errors == 0
    return {
        "passed": passed,
        "test_count": test_count,
        "failures": failures,
        "errors": errors,
        "output": stdout,
    }


def unittest_runner(state: ClassVerifierState) -> dict[str, Any]:
    # Skip if compile failed
    compile_report = state.get("compile_report", {})
    if not compile_report.get("compile_ok", True):
        return {
            "unittest_report": {
                "passed": False,
                "test_count": 0,
                "failures": 0,
                "errors": 1,
                "output": "",
                "stderr": compile_report.get("error", ""),
            }
        }

    imports = state.get("import_statements", [])
    partial_class_code = state.get("partial_class_code", "")
    test_code = state.get("test_code", "")
    test_class_name = _extract_test_class_name(test_code)

    if not test_code or not test_class_name:
        return {
            "unittest_report": {
                "passed": True,
                "test_count": 0,
                "failures": 0,
                "errors": 0,
                "output": "no test code",
                "stderr": "",
            }
        }

    script_parts = []
    if imports:
        script_parts.append("\n".join(imports))
    script_parts.append(partial_class_code)
    script_parts.append("import unittest")
    script_parts.append(test_code)
    script_parts.append(f"""
if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase({test_class_name})
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    import sys
    sys.exit(0 if result.wasSuccessful() else 1)
""")

    script = "\n\n".join(script_parts)
    exec_result = _executor.run(script)

    # TextTestRunner writes progress to stderr by default; parse both streams
    combined = exec_result.stdout + "\n" + exec_result.stderr
    report = _parse_unittest_output(combined)
    report["output"] = combined
    report["stderr"] = exec_result.stderr
    if exec_result.timed_out:
        report["passed"] = False
        report["errors"] += 1
        report["stderr"] = "Timed out"

    return {"unittest_report": report}


def _extract_test_class_name(test_code: str) -> str:
    """Extract the first class name from the test_code source."""
    m = re.search(r'class\s+(\w+)\s*\(', test_code)
    return m.group(1) if m else ""
