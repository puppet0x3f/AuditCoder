"""LangGraph StateGraph definition for the Verifier Agent.

Wires all verifier nodes into a directed graph with a conditional edge
after compile_verifier to skip runtime checks when compilation fails.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from auditcoder.verifier_agent.nodes import (
    bruteforce_verifier,
    compile_verifier,
    complexity_verifier,
    contract_verifier,
    counterexample_gen,
    sample_test_runner,
    verdict_assembler,
    weak_signal_eval,
)
from auditcoder.verifier_agent.state import VerifierState


def _compile_router(state: VerifierState) -> str:
    """Skip runtime verifiers when compilation failed."""
    report = state.get("compile_report", {})
    if report.get("compile_ok", False):
        return "sample_test_runner"
    return "counterexample_gen"


def build_verifier_agent_graph() -> StateGraph:
    """Construct and compile the Verifier Agent graph."""

    graph = StateGraph(VerifierState)

    graph.add_node("contract_verifier", contract_verifier)
    graph.add_node("compile_verifier", compile_verifier)
    graph.add_node("sample_test_runner", sample_test_runner)
    graph.add_node("bruteforce_verifier", bruteforce_verifier)
    graph.add_node("complexity_verifier", complexity_verifier)
    graph.add_node("counterexample_gen", counterexample_gen)
    graph.add_node("weak_signal_eval", weak_signal_eval)
    graph.add_node("verdict_assembler", verdict_assembler)

    graph.set_entry_point("contract_verifier")

    graph.add_edge("contract_verifier", "compile_verifier")

    graph.add_conditional_edges(
        "compile_verifier",
        _compile_router,
        {
            "sample_test_runner": "sample_test_runner",
            "counterexample_gen": "counterexample_gen",
        },
    )

    graph.add_edge("sample_test_runner", "bruteforce_verifier")
    graph.add_edge("bruteforce_verifier", "complexity_verifier")
    graph.add_edge("complexity_verifier", "counterexample_gen")

    graph.add_edge("counterexample_gen", "weak_signal_eval")
    graph.add_edge("weak_signal_eval", "verdict_assembler")
    graph.add_edge("verdict_assembler", END)

    return graph.compile()
