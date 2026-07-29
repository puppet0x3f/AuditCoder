"""Class plan agent graph: parse ClassEval entry → ClassPlanIR."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from auditcoder_class.class_plan_agent.state import ClassPlanState
from auditcoder_class.class_plan_agent.nodes.class_context_builder import class_context_builder
from auditcoder_class.class_plan_agent.nodes.method_thinker import method_thinker
from auditcoder_class.class_plan_agent.nodes.method_algorithm_selector import method_algorithm_selector
from auditcoder_class.class_plan_agent.nodes.dependency_sorter import dependency_sorter
from auditcoder_class.class_plan_agent.nodes.ir_assembler import ir_assembler


def build_class_plan_agent_graph() -> StateGraph:
    graph = StateGraph(ClassPlanState)
    graph.add_node("class_context_builder", class_context_builder)
    graph.add_node("method_thinker", method_thinker)
    graph.add_node("method_algorithm_selector", method_algorithm_selector)
    graph.add_node("dependency_sorter", dependency_sorter)
    graph.add_node("ir_assembler", ir_assembler)

    graph.set_entry_point("class_context_builder")
    graph.add_edge("class_context_builder", "method_thinker")
    graph.add_edge("method_thinker", "method_algorithm_selector")
    graph.add_edge("method_algorithm_selector", "dependency_sorter")
    graph.add_edge("dependency_sorter", "ir_assembler")
    graph.add_edge("ir_assembler", END)

    return graph.compile()
