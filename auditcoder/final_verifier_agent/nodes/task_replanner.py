"""Node: task_replanner — re-plan a failed task node using error context."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from auditcoder.common.llm import get_llm
from auditcoder.common.utils import extract_json
from auditcoder.final_verifier_agent.prompts.task_replanner import (
    TASK_REPLANNER_SYSTEM,
    TASK_REPLANNER_USER,
)
from auditcoder.final_verifier_agent.state import FinalVerifierState


def task_replanner(state: FinalVerifierState) -> dict[str, Any]:
    error_location = state.get("error_location") or {}
    task_id = error_location.get("task_id", "")
    task_plan_ir = state.get("task_plan_ir", {})

    node_map: dict[str, dict] = {
        n["id"]: n for n in task_plan_ir.get("task_nodes", [])
    }
    task_node = node_map.get(task_id, {})

    problem = task_plan_ir.get("problem", {})
    problem_context = (
        f"题目：{problem.get('title', '')}\n"
        f"描述：{problem.get('description', '')}\n"
        f"输入格式：{problem.get('input_format', '')}\n"
        f"输出格式：{problem.get('output_format', '')}\n"
        f"约束：{json.dumps(problem.get('constraints', {}), ensure_ascii=False)}\n"
        f"验证不变量：{json.dumps((task_plan_ir.get('validation') or {}).get('invariants', []), ensure_ascii=False)}"
    )

    failing_case = error_location.get("failing_case", {})
    failing_case_str = (
        json.dumps(failing_case, ensure_ascii=False) if failing_case else "（无）"
    )

    user_content = TASK_REPLANNER_USER.format(
        problem_context=problem_context,
        task_node_json=json.dumps(task_node, ensure_ascii=False, indent=2),
        error_type=error_location.get("error_type", "unknown"),
        error_detail=error_location.get("detail", ""),
        failing_case=failing_case_str,
        localization_method=error_location.get("method", "unknown"),
        localization_confidence=error_location.get("confidence", "unknown"),
    )

    llm = get_llm(temperature=0.3)
    messages = [
        SystemMessage(content=TASK_REPLANNER_SYSTEM),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(messages)
    parsed = extract_json(response)

    proposed_node: dict[str, Any] = parsed if isinstance(parsed, dict) else {}
    updated_node = {**task_node, **proposed_node}

    # Node-level retry must not mutate the responsibility boundary.  Structural
    # changes require branch-level replanning because callers and provenance
    # outside this node are otherwise not updated.
    immutable_fields = {
        "id",
        "goal",
        "type",
        "is_leaf",
        "agent_type",
        "parent",
        "dependencies",
        "inputs",
        "outputs",
        "interface",
    }
    for field in immutable_fields:
        if field in task_node:
            updated_node[field] = task_node[field]
    updated_node["id"] = task_id

    # Replace the task node in task_plan_ir
    updated_task_nodes = [
        updated_node if n["id"] == task_id else n
        for n in task_plan_ir.get("task_nodes", [])
    ]
    updated_ir = {**task_plan_ir, "task_nodes": updated_task_nodes}

    retried_tasks = list(state.get("retried_tasks") or [])
    if task_id not in retried_tasks:
        retried_tasks.append(task_id)

    return {
        "updated_task_node": updated_node,
        "task_plan_ir": updated_ir,
        "retried_tasks": retried_tasks,
    }
