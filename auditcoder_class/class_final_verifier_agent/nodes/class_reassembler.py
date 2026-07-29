"""Node: re-assemble the class after a method has been retried."""

from __future__ import annotations

from typing import Any

from auditcoder_class.class_assembler_agent.agent import build_class_assembler_graph
from auditcoder_class.class_final_verifier_agent.state import ClassFinalVerifierState


def class_reassembler(state: ClassFinalVerifierState) -> dict[str, Any]:
    assembler = build_class_assembler_graph()
    result = assembler.invoke({
        "class_plan_ir": state["class_plan_ir"],
        "method_bodies": state["method_bodies"],
    })
    return {
        "final_class_code": result["final_class_code"],
        "structural_violations": result.get("structural_violations", []),
    }
