"""LangGraph StateGraph definition for the Plan Agent.

This module wires all nodes together into a directed graph that produces a
TaskPlanIR in a single pass (no revision loop).
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from auditcoder.plan_agent.nodes import (
    algorithm_selector,
    constraint_analyzer,
    deep_thinker,
    interface_designer,
    ir_assembler,
    problem_parser,
    task_decomposer,
    test_case_analyzer,
)
from auditcoder.plan_agent.state import PlanState


def build_plan_agent_graph() -> StateGraph:
    """Construct and compile the Plan Agent graph."""

    graph = StateGraph(PlanState)

    graph.add_node("problem_parser", problem_parser)
    graph.add_node("deep_thinker", deep_thinker)
    graph.add_node("constraint_analyzer", constraint_analyzer)
    graph.add_node("algorithm_selector", algorithm_selector)
    graph.add_node("task_decomposer", task_decomposer)
    graph.add_node("interface_designer", interface_designer)
    graph.add_node("test_case_analyzer", test_case_analyzer)
    graph.add_node("ir_assembler", ir_assembler)

    graph.set_entry_point("problem_parser")

    graph.add_edge("problem_parser", "deep_thinker")
    graph.add_edge("deep_thinker", "constraint_analyzer")
    graph.add_edge("constraint_analyzer", "algorithm_selector")
    graph.add_edge("algorithm_selector", "task_decomposer")
    graph.add_edge("task_decomposer", "interface_designer")
    graph.add_edge("interface_designer", "test_case_analyzer")
    graph.add_edge("test_case_analyzer", "ir_assembler")
    graph.add_edge("ir_assembler", END)

    return graph.compile()


def build_partial_replan_graph() -> StateGraph:
    """Partial replan graph: starts from deep_thinker, skipping only problem_parser.

    problem_structured is reused (the problem text hasn't changed), but deep_thinker
    is re-run with failure_context injected so it can revise its structural insights,
    remove failed solution families, and guide algorithm_selector toward a new strategy.

    The caller must pre-populate problem_structured, failure_context, and
    failed_algorithms in the initial state.
    """
    graph = StateGraph(PlanState)

    graph.add_node("deep_thinker", deep_thinker)
    graph.add_node("constraint_analyzer", constraint_analyzer)
    graph.add_node("algorithm_selector", algorithm_selector)
    graph.add_node("task_decomposer", task_decomposer)
    graph.add_node("interface_designer", interface_designer)
    graph.add_node("test_case_analyzer", test_case_analyzer)
    graph.add_node("ir_assembler", ir_assembler)

    graph.set_entry_point("deep_thinker")

    graph.add_edge("deep_thinker", "constraint_analyzer")
    graph.add_edge("constraint_analyzer", "algorithm_selector")
    graph.add_edge("algorithm_selector", "task_decomposer")
    graph.add_edge("task_decomposer", "interface_designer")
    graph.add_edge("interface_designer", "test_case_analyzer")
    graph.add_edge("test_case_analyzer", "ir_assembler")
    graph.add_edge("ir_assembler", END)

    return graph.compile()
