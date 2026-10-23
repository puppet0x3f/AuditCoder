"""Node: algorithm_selector — 基于约束分析选择算法并输出推理链。"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from auditcoder.plan_agent.llm import get_llm, invoke_llm
from auditcoder.plan_agent.nodes._utils import extract_json
from auditcoder.common.problem_guidance import strategy_card
from auditcoder.plan_agent.prompts.selector import (
    ALGORITHM_SELECTOR_SYSTEM,
    ALGORITHM_SELECTOR_USER,
)
from auditcoder.plan_agent.state import PlanState


def _fallback_algorithm_selection(state: PlanState, reason: str) -> dict:
    constraint = state.get("constraint_analysis", {}) or {}
    time_budget = constraint.get("time_complexity_upper_bound", "")
    features = constraint.get("special_constraint_features", [])

    return {
        "problem_category": "Algorithmic construction / data structures",
        "key_observations": [
            "The solution must satisfy the problem's feasibility invariant before optimizing the requested output.",
            f"The selected algorithm must fit the time budget: {time_budget or 'derived from constraints'}.",
            "A linear or near-linear constructive algorithm is preferred for large input sizes.",
        ],
        "algorithm_selection": [
            {
                "algorithm_type": "greedy",
                "agent_id": "agent_greedy",
                "subtask": "Construct a feasible answer while preserving the required ordering or lexicographic invariant.",
                "reason": "The problem asks for a valid construction, and greedy choices with a maintained invariant are often sufficient.",
            },
            {
                "algorithm_type": "sorting",
                "agent_id": "agent_sorting",
                "subtask": "Maintain or select remaining values in the required order.",
                "reason": "Ordering constraints and lexicographic optimality often require sorted sets or ordered traversal.",
            },
        ],
        "reasoning_chain": [
            "Use constraints and feasibility rules to identify the invariant.",
            "Maintain available values in an order that supports the requested optimality criterion.",
            "Validate each constructed step against the invariant.",
            f"Fallback used because algorithm_selector failed: {reason}",
        ],
        "fallback_features": features,
    }


def algorithm_selector(state: PlanState) -> dict:
    llm = get_llm(max_tokens=4096)
    problem_json = json.dumps(state["problem_structured"], ensure_ascii=False, indent=2)
    constraint_json = json.dumps(state["constraint_analysis"], ensure_ascii=False, indent=2)
    thinking_json = json.dumps(state.get("deep_thinking", {}), ensure_ascii=False, indent=2)

    user_content = ALGORITHM_SELECTOR_USER.format(
        problem_structured=problem_json,
        constraint_analysis=constraint_json,
        deep_thinking=thinking_json,
    )
    card = strategy_card(state["problem_structured"])
    if card:
        user_content += (
            "\n\n## Deterministic audit card (must be proved and sample-checked)\n"
            + card
            + "\nDo not select a contradictory family; state the invariant and complexity explicitly."
        )

    messages = [
        SystemMessage(content=ALGORITHM_SELECTOR_SYSTEM),
        HumanMessage(
            content=(
                user_content
                + "\n\nOutput must be concise: at most 3 key_observations, "
                "2 algorithm_selection items, 4 reasoning_chain steps, and each string under 35 words."
            )
        ),
    ]
    try:
        response = invoke_llm(llm, messages)
        parsed = extract_json(response)
    except Exception as exc:
        fallback = _fallback_algorithm_selection(
            state,
            f"{type(exc).__name__}: {str(exc)[:300]}",
        )
        return {"algorithm_selection": fallback, "messages": messages}

    if not isinstance(parsed, dict):
        fallback = _fallback_algorithm_selection(
            state,
            "algorithm_selector returned non-object JSON",
        )
        return {"algorithm_selection": fallback, "messages": messages + [response]}

    return {"algorithm_selection": parsed, "messages": messages + [response]}
