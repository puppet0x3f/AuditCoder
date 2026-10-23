"""Node: constraint_analyzer — 从数据范围推导复杂度目标。"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from auditcoder.plan_agent.llm import get_llm, invoke_llm
from auditcoder.plan_agent.nodes._utils import extract_json
from auditcoder.plan_agent.prompts.analyzer import (
    CONSTRAINT_ANALYZER_SYSTEM,
    CONSTRAINT_ANALYZER_USER,
)
from auditcoder.plan_agent.state import PlanState


def constraint_analyzer(state: PlanState) -> dict:
    llm = get_llm()
    problem_json = json.dumps(state["problem_structured"], ensure_ascii=False, indent=2)
    thinking_json = json.dumps(state.get("deep_thinking", {}), ensure_ascii=False, indent=2)
    messages = [
        SystemMessage(content=CONSTRAINT_ANALYZER_SYSTEM),
        HumanMessage(
            content=CONSTRAINT_ANALYZER_USER.format(
                problem_structured=problem_json,
                deep_thinking=thinking_json,
            )
        ),
    ]
    response = invoke_llm(llm, messages)
    parsed = extract_json(response)

    return {"constraint_analysis": parsed, "messages": messages + [response]}
