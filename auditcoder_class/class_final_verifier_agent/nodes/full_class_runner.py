"""Node: run the full combined test suite against the assembled class."""

from __future__ import annotations

import re
from typing import Any

from auditcoder.verifier_agent.sandbox.executor import SandboxExecutor
from auditcoder_class.class_final_verifier_agent.state import ClassFinalVerifierState

_executor = SandboxExecutor(default_timeout_s=30.0)


def full_class_runner(state: ClassFinalVerifierState) -> dict[str, Any]:
    ir = state["class_plan_ir"]
    final_class_code = state["final_class_code"]

    imports = ir.get("import_statements", [])
    test_code = ir.get("test", "")
    test_classes = ir.get("test_classes", [])

    if not test_code:
        return {
            "execution_result": {
                "all_passed": True,
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
                "time_ms": 0.0,
                "timed_out": False,
            }
        }

    # Build test class list string for the runner
    test_class_list = ", ".join(test_classes)

    script_parts = []
    if imports:
        script_parts.append("\n".join(imports))
    script_parts.append(final_class_code)
    script_parts.append("import unittest")
    script_parts.append(test_code)
    script_parts.append(f"""
if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls_name in [{', '.join(repr(c) for c in test_classes)}]:
        try:
            cls = globals()[cls_name]
            suite.addTests(loader.loadTestsFromTestCase(cls))
        except KeyError:
            pass
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    import sys
    sys.exit(0 if result.wasSuccessful() else 1)
""")

    script = "\n\n".join(script_parts)
    exec_result = _executor.run(script)

    all_passed = exec_result.exit_code == 0 and not exec_result.timed_out

    return {
        "execution_result": {
            "all_passed": all_passed,
            "stdout": exec_result.stdout,
            "stderr": exec_result.stderr,
            "exit_code": exec_result.exit_code,
            "time_ms": exec_result.time_ms,
            "timed_out": exec_result.timed_out,
        }
    }
