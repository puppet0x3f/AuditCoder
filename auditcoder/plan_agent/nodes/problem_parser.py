"""Node: problem_parser — 解析原始题目文本为结构化信息。"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from auditcoder.plan_agent.llm import get_llm, invoke_llm
from auditcoder.plan_agent.nodes._utils import extract_json
from auditcoder.plan_agent.prompts.parser import (
    PROBLEM_PARSER_SYSTEM,
    PROBLEM_PARSER_USER,
)
from auditcoder.plan_agent.state import PlanState


def problem_parser(state: PlanState) -> dict:
    llm = get_llm()
    messages = [
        SystemMessage(content=PROBLEM_PARSER_SYSTEM),
        HumanMessage(
            content=PROBLEM_PARSER_USER.format(problem_raw=state["problem_raw"])
        ),
    ]
    response = invoke_llm(llm, messages)
    parsed = extract_json(response)

    return {"problem_structured": parsed, "messages": messages + [response]}
