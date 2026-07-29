"""Node: code_reassembler — re-assemble final code after a retry."""

from __future__ import annotations

from typing import Any

from auditcoder.assembler_agent.agent import build_assembler_agent_graph
from auditcoder.final_verifier_agent.state import FinalVerifierState


def code_reassembler(state: FinalVerifierState) -> dict[str, Any]:
    task_plan_ir = state.get("task_plan_ir", {})
    code_snippets = state.get("code_snippets", {})

    assembler = build_assembler_agent_graph()
    result = assembler.invoke(
        {"task_plan_ir": task_plan_ir, "code_snippets": code_snippets}
    )

    new_final_code: str = result.get("final_code", "")
    return {"final_code": new_final_code}
