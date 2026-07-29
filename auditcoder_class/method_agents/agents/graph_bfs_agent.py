"""Graph BFS method agent."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from auditcoder.common.llm import get_llm
from auditcoder.common.utils import extract_json
from auditcoder_class.method_agents.state import MethodAgentState
from auditcoder_class.method_agents._utils import format_class_context, format_method_feedback, format_method_interface
from auditcoder_class.method_agents.prompts.graph_bfs import GRAPH_BFS_METHOD_SYSTEM, GRAPH_BFS_METHOD_USER


def _graph_bfs_code_generator(state: MethodAgentState) -> dict:
    method_node = state["method_node"]
    ir = state["class_plan_ir"]
    generated = state.get("generated_methods", {})
    hints_text = "\n".join(f"- {h}" for h in method_node.get("hints", [])) or "(none)"
    user_content = GRAPH_BFS_METHOD_USER.format(
        class_context=format_class_context(ir, method_node, generated),
        method_interface=format_method_interface(method_node.get("interface")),
        method_description=method_node.get("method_description", ""),
        hints=hints_text,
    )
    feedback_text = format_method_feedback(state.get("feedback"))
    if feedback_text:
        user_content += "\n" + feedback_text
    llm = get_llm(temperature=0.2)
    messages = [SystemMessage(content=GRAPH_BFS_METHOD_SYSTEM), HumanMessage(content=user_content)]
    response = llm.invoke(messages)
    parsed = extract_json(response)
    method_body = parsed.get("method_body", "") if isinstance(parsed, dict) else ""
    return {"method_body": method_body, "messages": messages + [response]}


def build_graph_bfs_agent_graph() -> StateGraph:
    graph = StateGraph(MethodAgentState)
    graph.add_node("code_generator", _graph_bfs_code_generator)
    graph.set_entry_point("code_generator")
    graph.add_edge("code_generator", END)
    return graph.compile()
