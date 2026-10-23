"""Node: ir_reviewer — 审查 TaskPlanIR 的完整性与一致性。"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from auditcoder.plan_agent.llm import get_llm, invoke_llm
from auditcoder.plan_agent.nodes._utils import extract_json
from auditcoder.plan_agent.prompts.reviewer import (
    IR_REVIEWER_SYSTEM,
    IR_REVIEWER_USER,
)
from auditcoder.plan_agent.state import PlanState
from auditcoder.schemas.contract_graph import ensure_catg, validate_contract_graph


def ir_reviewer(state: PlanState) -> dict:
    # The CATG integrity checks are deterministic and run even when an LLM is
    # unavailable.  This prevents a missing reviewer response from silently
    # releasing an unowned or cyclic graph to code generation.
    ir = ensure_catg(state.get("task_plan_ir", {}))
    structural_issues = validate_contract_graph(ir)
    ir_json = json.dumps(ir, ensure_ascii=False, indent=2)

    messages = [
        SystemMessage(content=IR_REVIEWER_SYSTEM),
        HumanMessage(
            content=IR_REVIEWER_USER.format(task_plan_ir=ir_json)
        ),
    ]
    try:
        llm = get_llm()
        response = invoke_llm(llm, messages)
        parsed = extract_json(response)
    except Exception as exc:
        parsed = {
            "status": "pass" if not structural_issues else "fail",
            "issues": structural_issues,
            "retry_node": "",
            "reason": "deterministic CATG review used because reviewer call failed: "
            + f"{type(exc).__name__}: {str(exc)[:180]}",
        }
        return {
            "task_plan_ir": ir,
            "review_result": parsed,
            "revision_count": state.get("revision_count", 0),
            "messages": messages,
        }

    if not isinstance(parsed, dict):
        parsed = {"status": "fail", "issues": ["IR reviewer returned non-object JSON"], "retry_node": "", "reason": "invalid reviewer output"}
    if structural_issues:
        parsed = dict(parsed)
        parsed["status"] = "fail"
        parsed["issues"] = list(dict.fromkeys(list(parsed.get("issues") or []) + structural_issues))
        parsed.setdefault("reason", "deterministic CATG integrity checks failed")

    revision_count = state.get("revision_count", 0)
    if parsed.get("status") != "pass":
        revision_count += 1

    return {
        "task_plan_ir": ir,
        "review_result": parsed,
        "revision_count": revision_count,
        "messages": messages + [response],
    }
