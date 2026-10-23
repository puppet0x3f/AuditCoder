"""LangGraph orchestration for evidence-guided bounded repair."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from auditcoder.final_verifier_agent.nodes import (
    algo_retrier,
    code_reassembler,
    error_locator,
    final_report_assembler,
    full_code_runner,
    scope_selector,
    task_replanner,
)
from auditcoder.final_verifier_agent.nodes.full_code_runner import (
    post_repair_code_runner,
)
from auditcoder.final_verifier_agent.state import FinalVerifierState


def _max_attempts(state: FinalVerifierState) -> int:
    raw = state.get("max_repair_attempts", 3)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 3


def _run_router(state: FinalVerifierState) -> str:
    execution_result = state.get("execution_result", {}) or {}
    if execution_result.get("all_passed", False):
        return "final_report_assembler"
    if _max_attempts(state) <= 0:
        return "final_report_assembler"
    return "error_locator"


def _locator_router(state: FinalVerifierState) -> str:
    """Compatibility router retained for callers of the original graph."""

    location = state.get("error_location")
    if not location:
        return "final_report_assembler"
    if int(state.get("repair_attempts", 0) or 0) >= _max_attempts(state):
        return "final_report_assembler"
    task_id = str(location.get("task_id") or "")
    if task_id and task_id in (state.get("retried_tasks") or []):
        return "final_report_assembler"
    # The original public API routed directly to task_replanner.  Keep that
    # return value stable; the compiled graph uses _graph_locator_router below
    # to insert the deterministic scope-selection node.
    return "task_replanner"


def _graph_locator_router(state: FinalVerifierState) -> str:
    """Route the compiled graph through paper Algorithm SelectScope."""

    location = state.get("error_location")
    if not location:
        return "final_report_assembler"
    if int(state.get("repair_attempts", 0) or 0) >= _max_attempts(state):
        return "final_report_assembler"
    return "scope_selector"


def _scope_router(state: FinalVerifierState) -> str:
    if not state.get("repair_scope"):
        return "final_report_assembler"
    return "task_replanner"


def _post_repair_router(state: FinalVerifierState) -> str:
    execution_result = state.get("execution_result", {}) or {}
    if execution_result.get("all_passed", False):
        return "final_report_assembler"
    if int(state.get("repair_attempts", 0) or 0) >= _max_attempts(state):
        return "final_report_assembler"
    # Each subsequent attempt starts from the accepted active state (or the
    # exact restored snapshot), and Locate therefore recomputes the anchor and
    # current evidence instead of reusing a stale location.
    return "error_locator"


def build_final_verifier_graph() -> StateGraph:
    graph = StateGraph(FinalVerifierState)

    graph.add_node("full_code_runner", full_code_runner)
    graph.add_node("error_locator", error_locator)
    graph.add_node("scope_selector", scope_selector)
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
        _graph_locator_router,
        {
            "scope_selector": "scope_selector",
            "final_report_assembler": "final_report_assembler",
        },
    )
    graph.add_conditional_edges(
        "scope_selector",
        _scope_router,
        {
            "task_replanner": "task_replanner",
            "final_report_assembler": "final_report_assembler",
        },
    )
    graph.add_edge("task_replanner", "algo_retrier")
    graph.add_edge("algo_retrier", "code_reassembler")
    graph.add_edge("code_reassembler", "post_repair_code_runner")
    graph.add_conditional_edges(
        "post_repair_code_runner",
        _post_repair_router,
        {
            "error_locator": "error_locator",
            "final_report_assembler": "final_report_assembler",
        },
    )
    graph.add_edge("final_report_assembler", END)

    return graph.compile()


__all__ = [
    "build_final_verifier_graph",
    "_locator_router",
    "_run_router",
]
