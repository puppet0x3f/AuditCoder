"""Node: task_decomposer — 将算法问题递归分解为层级任务树。"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from auditcoder.plan_agent.llm import get_llm
from auditcoder.plan_agent.nodes._utils import extract_json
from auditcoder.plan_agent.prompts.decomposer import (
    TASK_DECOMPOSER_SYSTEM,
    TASK_DECOMPOSER_USER,
)
from auditcoder.plan_agent.state import PlanState


def task_decomposer(state: PlanState) -> dict:
    llm = get_llm()
    problem_json = json.dumps(state["problem_structured"], ensure_ascii=False, indent=2)
    constraint_json = json.dumps(state["constraint_analysis"], ensure_ascii=False, indent=2)
    algo_json = json.dumps(state["algorithm_selection"], ensure_ascii=False, indent=2)

    messages = [
        SystemMessage(content=TASK_DECOMPOSER_SYSTEM),
        HumanMessage(
            content=TASK_DECOMPOSER_USER.format(
                problem_structured=problem_json,
                constraint_analysis=constraint_json,
                algorithm_selection=algo_json,
            )
        ),
    ]
    response = llm.invoke(messages)
    parsed = extract_json(response)

    if isinstance(parsed, list):
        task_nodes = parsed
        meta: dict = {}
    else:
        task_nodes = parsed.get("task_nodes", [])
        meta = parsed

    return {
        "task_tree": task_nodes,
        "solution_strategy": {
            "global_strategy": meta.get("global_strategy", ""),
            "root_id": meta.get("root_id", "S0"),
            "execution_order": meta.get("execution_order", []),
        },
        "messages": messages + [response],
    }
