"""Node: compile-check the partial class code."""

from __future__ import annotations

from typing import Any

from auditcoder_class.class_verifier_agent.state import ClassVerifierState


def compile_verifier(state: ClassVerifierState) -> dict[str, Any]:
    code = state.get("partial_class_code", "")
    imports = state.get("import_statements", [])
    full_code = "\n".join(imports) + "\n\n" + code if imports else code

    try:
        compile(full_code, "<string>", "exec")
        return {"compile_report": {"compile_ok": True, "error": ""}}
    except SyntaxError as e:
        return {"compile_report": {"compile_ok": False, "error": str(e)}}
