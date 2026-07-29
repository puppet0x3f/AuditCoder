"""Node: ir_reviewer — 审查 TaskPlanIR 的完整性与一致性。"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from auditcoder.plan_agent.llm import get_llm
from auditcoder.plan_agent.nodes._utils import extract_json
from auditcoder.plan_agent.prompts.reviewer import (
    IR_REVIEWER_SYSTEM,
    IR_REVIEWER_USER,
)
from auditcoder.plan_agent.state import PlanState


def ir_reviewer(state: PlanState) -> dict:
    llm = get_llm()
    ir_json = json.dumps(state["task_plan_ir"], ensure_ascii=False, indent=2)

    messages = [
        SystemMessage(content=IR_REVIEWER_SYSTEM),
        HumanMessage(
            content=IR_REVIEWER_USER.format(task_plan_ir=ir_json)
        ),
    ]
    response = llm.invoke(messages)
    parsed = extract_json(response)

    revision_count = state.get("revision_count", 0)
    if parsed.get("status") != "pass":
        revision_count += 1

    return {
        "review_result": parsed,
        "revision_count": revision_count,
        "messages": messages + [response],
    }
