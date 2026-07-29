"""LangGraph StateGraph definition for the Final Verifier Agent.

Runs the assembled final code against all test cases, then (on failure)
locates which task/agent is responsible before producing the final report.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from auditcoder.final_verifier_agent.nodes import (
    algo_retrier,
    code_reassembler,
    error_locator,
    final_report_assembler,
    full_code_runner,
    task_replanner,
)
from auditcoder.final_verifier_agent.nodes.full_code_runner import post_repair_code_runner
from auditcoder.final_verifier_agent.state import FinalVerifierState


def _run_router(state: FinalVerifierState) -> str:
    """Route to error_locator only when the run failed."""
    execution_result = state.get("execution_result", {})
    if execution_result.get("all_passed", False):
        return "final_report_assembler"
    return "error_locator"


def _locator_router(state: FinalVerifierState) -> str:
    """Route to task_replanner on the first located failure; otherwise skip to report."""
    location = state.get("error_location")
    if not location:
        return "final_report_assembler"
    task_id = location.get("task_id", "")
    if task_id in (state.get("retried_tasks") or []):
        return "final_report_assembler"
    return "task_replanner"


def build_final_verifier_graph() -> StateGraph:
    """Construct and compile the Final Verifier Agent graph."""
    graph = StateGraph(FinalVerifierState)

    graph.add_node("full_code_runner", full_code_runner)
    graph.add_node("error_locator", error_locator)
    graph.add_node("task_replanner", task_replanner)
    graph.add_node("algo_retrier", algo_retrier)
    graph.add_node("code_reassembler", code_reassembler)
    graph.add_node("post_repair_code_runner", post_repair_code_runner)
    graph.add_node("final_report_assembler", final_report_assembler)

    graph.set_entry_point("full_code_runner")

    graph.add_conditional_edges(
        "full_code_runner",
        _run_router,
        {
            "error_locator": "error_locator",
            "final_report_assembler": "final_report_assembler",
        },
    )

    graph.add_conditional_edges(
        "error_locator",
        _locator_router,
        {
            "task_replanner": "task_replanner",
            "final_report_assembler": "final_report_assembler",
        },
    )

    graph.add_edge("task_replanner", "algo_retrier")
    graph.add_edge("algo_retrier", "code_reassembler")
    graph.add_edge("code_reassembler", "post_repair_code_runner")
    graph.add_edge("post_repair_code_runner", "final_report_assembler")
    graph.add_edge("final_report_assembler", END)

    return graph.compile()
