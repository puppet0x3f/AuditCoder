"""Two Pointer / Sliding Window agent — LangGraph graph that generates code for a single two-pointer task node."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from auditcoder.common.llm import get_llm
from auditcoder.common.utils import extract_json
from auditcoder.algo_agents.prompts.two_pointer import TWO_POINTER_AGENT_SYSTEM, TWO_POINTER_AGENT_USER
from auditcoder.algo_agents.state import AlgoAgentState
from auditcoder.algo_agents._utils import format_feedback, format_interface, format_slots


def _two_pointer_code_generator(state: AlgoAgentState) -> dict:
    task_node = state["task_node"]
    ir = state["task_plan_ir"]

    problem_description = ir.get("problem", {}).get("description", "")
    global_strategy = ir.get("global_strategy", "")

    hints = task_node.get("hints", [])
    hints_text = "\n".join(f"- {h}" for h in hints) if hints else "（无）"

    user_content = TWO_POINTER_AGENT_USER.format(
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

    feedback_text = format_feedback(state.get("feedback"))
    if feedback_text:
        user_content = user_content + "\n" + feedback_text

    llm = get_llm(temperature=0.2, max_tokens=4096)
    messages = [
        SystemMessage(content=TWO_POINTER_AGENT_SYSTEM),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(messages)
    parsed = extract_json(response)

    code_snippet = parsed.get("code_snippet", "") if isinstance(parsed, dict) else ""
    return {"code_snippet": code_snippet, "messages": messages + [response]}


def build_two_pointer_agent_graph() -> StateGraph:
    """Build and compile the Two Pointer / Sliding Window agent graph."""
    graph = StateGraph(AlgoAgentState)
    graph.add_node("code_generator", _two_pointer_code_generator)
    graph.set_entry_point("code_generator")
    graph.add_edge("code_generator", END)
    return graph.compile()
