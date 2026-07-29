"""Class verifier agent graph: compile → unittest → verdict."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from auditcoder_class.class_verifier_agent.state import ClassVerifierState
from auditcoder_class.class_verifier_agent.nodes.compile_verifier import compile_verifier
from auditcoder_class.class_verifier_agent.nodes.unittest_runner import unittest_runner
from auditcoder_class.class_verifier_agent.nodes.verdict_assembler import verdict_assembler


def build_class_verifier_agent_graph() -> StateGraph:
    graph = StateGraph(ClassVerifierState)
    graph.add_node("compile_verifier", compile_verifier)
    graph.add_node("unittest_runner", unittest_runner)
    graph.add_node("verdict_assembler", verdict_assembler)

    graph.set_entry_point("compile_verifier")
    graph.add_edge("compile_verifier", "unittest_runner")
    graph.add_edge("unittest_runner", "verdict_assembler")
    graph.add_edge("verdict_assembler", END)

    return graph.compile()
