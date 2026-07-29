"""Node: interface_designer — 为任务树中每个叶节点定义严格的函数接口。"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from auditcoder.plan_agent.llm import get_llm
from auditcoder.plan_agent.nodes._utils import extract_json
from auditcoder.plan_agent.prompts.designer import (
    INTERFACE_DESIGNER_SYSTEM,
    INTERFACE_DESIGNER_USER,
)
from auditcoder.plan_agent.state import PlanState


def interface_designer(state: PlanState) -> dict:
    llm = get_llm()
    task_tree = state["task_tree"]
    constraint_json = json.dumps(state["constraint_analysis"], ensure_ascii=False, indent=2)

    leaf_nodes = [n for n in task_tree if n.get("is_leaf", False)]
    leaf_json = json.dumps(leaf_nodes, ensure_ascii=False, indent=2)
    tree_json = json.dumps(task_tree, ensure_ascii=False, indent=2)

    messages = [
        SystemMessage(content=INTERFACE_DESIGNER_SYSTEM),
        HumanMessage(
            content=INTERFACE_DESIGNER_USER.format(
                leaf_nodes=leaf_json,
                task_tree=tree_json,
                constraint_analysis=constraint_json,
            )
        ),
    ]
    response = llm.invoke(messages)
    parsed = extract_json(response)

    return {"interface_definitions": parsed, "messages": messages + [response]}
