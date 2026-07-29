"""Class final verifier agent graph: full test run + error locator + retry."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, StateGraph

from auditcoder_class.class_final_verifier_agent.state import ClassFinalVerifierState
from auditcoder_class.class_final_verifier_agent.nodes.full_class_runner import full_class_runner
from auditcoder_class.class_final_verifier_agent.nodes.error_locator import error_locator
from auditcoder_class.class_final_verifier_agent.nodes.method_retrier import method_retrier
from auditcoder_class.class_final_verifier_agent.nodes.class_reassembler import class_reassembler
from auditcoder_class.class_final_verifier_agent.nodes.final_report_assembler import final_report_assembler


def _route_after_runner(state: ClassFinalVerifierState) -> Literal["error_locator", "final_report_assembler"]:
    exec_result = state.get("execution_result", {})
    if exec_result.get("all_passed", False):
        return "final_report_assembler"
    return "error_locator"


def _route_after_locator(state: ClassFinalVerifierState) -> Literal["method_retrier", "final_report_assembler"]:
    error_location = state.get("error_location")
    retried = state.get("retried_methods", [])
    if not error_location:
        return "final_report_assembler"
    # Structural violations (missing decorator) are assembler bugs — the method body
    # agent cannot fix them by regenerating the body, so skip retry entirely.
    if error_location.get("error_type") == "structural_violation":
        return "final_report_assembler"
    method_name = error_location.get("method_name", "")
    if method_name in retried:
        return "final_report_assembler"
    return "method_retrier"


def build_class_final_verifier_graph() -> StateGraph:
    graph = StateGraph(ClassFinalVerifierState)

    graph.add_node("full_class_runner", full_class_runner)
    graph.add_node("error_locator", error_locator)
    graph.add_node("method_retrier", method_retrier)
    graph.add_node("class_reassembler", class_reassembler)
    graph.add_node("final_report_assembler", final_report_assembler)

    graph.set_entry_point("full_class_runner")
    graph.add_conditional_edges("full_class_runner", _route_after_runner)
    graph.add_conditional_edges("error_locator", _route_after_locator)
    graph.add_edge("method_retrier", "class_reassembler")
    graph.add_edge("class_reassembler", "full_class_runner")
    graph.add_edge("final_report_assembler", END)

    return graph.compile()
