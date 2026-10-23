"""Greedy algorithm agent — LangGraph graph that generates code for a single greedy task node."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from auditcoder.common.llm import get_llm, invoke_llm
from auditcoder.common.utils import extract_json
from auditcoder.algo_agents.prompts.greedy import GREEDY_AGENT_SYSTEM, GREEDY_AGENT_USER
from auditcoder.algo_agents.state import AlgoAgentState
from auditcoder.algo_agents._utils import (
    format_contract_guardrails,
    format_feedback,
    format_interface,
    format_slots,
)


def _greedy_code_generator(state: AlgoAgentState) -> dict:
    task_node = state["task_node"]
    ir = state["task_plan_ir"]

    problem_description = ir.get("problem", {}).get("description", "")
    global_strategy = ir.get("global_strategy", "")

    hints = task_node.get("hints", [])
    hints_text = "\n".join(f"- {h}" for h in hints) if hints else "（无）"

    user_content = GREEDY_AGENT_USER.format(
        task_id=task_node.get("id", ""),
        goal=task_node.get("goal", ""),
        complexity_budget=task_node.get("complexity_budget", ""),
        context=task_node.get("context", ""),
        interface=format_interface(task_node.get("interface")),
        hints=hints_text,
        inputs=format_slots(task_node.get("inputs", [])),
        outputs=format_slots(task_node.get("outputs", [])),
        problem_description=problem_description,
        global_strategy=global_strategy,
    )
    user_content += "\n\n" + format_contract_guardrails(task_node, ir)

    feedback_text = format_feedback(state.get("feedback"))
    if feedback_text:
        user_content = user_content + "\n" + feedback_text

    llm = get_llm(temperature=0.2, max_tokens=3500)
    messages = [
        SystemMessage(content=GREEDY_AGENT_SYSTEM),
        HumanMessage(content=user_content),
    ]
    response = invoke_llm(llm, messages)
    parsed = extract_json(response)

    code_snippet = parsed.get("code_snippet", "") if isinstance(parsed, dict) else ""
    return {"code_snippet": code_snippet, "messages": messages + [response]}


def build_greedy_agent_graph() -> StateGraph:
    """Build and compile the Greedy algorithm agent graph."""
    graph = StateGraph(AlgoAgentState)
    graph.add_node("code_generator", _greedy_code_generator)
    graph.set_entry_point("code_generator")
    graph.add_edge("code_generator", END)
    return graph.compile()
