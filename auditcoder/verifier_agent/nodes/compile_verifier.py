"""Node: compile_verifier — 语法检查与 import 测试。

纯程序化：py_compile + 沙箱 import。
"""

from __future__ import annotations

import py_compile
import tempfile
import os

from auditcoder.verifier_agent.sandbox import SandboxExecutor
from auditcoder.verifier_agent.state import VerifierState


def compile_verifier(state: VerifierState) -> dict:
    source_code = state["code_snippet"]["source_code"]

    syntax_ok = False
    compile_ok = False
    errors: list[str] = []

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(source_code)
            tmp_path = f.name

        py_compile.compile(tmp_path, doraise=True)
        syntax_ok = True
    except py_compile.PyCompileError as exc:
        errors.append(f"语法错误: {exc}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if syntax_ok:
        import_code = f"{source_code}\nprint('__import_ok__')\n"
        executor = SandboxExecutor(default_timeout_s=10.0)
        result = executor.run(import_code)
        if "__import_ok__" in result.stdout:
            compile_ok = True
        else:
            errors.append(f"Import 失败: {result.stderr.strip()}")

    return {
        "compile_report": {
            "syntax_ok": syntax_ok,
            "compile_ok": compile_ok,
            "errors": errors,
        }
    }
