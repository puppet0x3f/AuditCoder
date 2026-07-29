"""Node: deep_thinker — 在约束分析前对问题做深度开放式思考。

Sits between problem_parser and constraint_analyzer. Produces deep_thinking,
a structured record of insights, traps, and candidate solution families that
all downstream nodes can reference.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from auditcoder.plan_agent.llm import get_llm
from auditcoder.plan_agent.nodes._utils import extract_json
from auditcoder.plan_agent.prompts.thinker import DEEP_THINKER_REPLAN_USER, DEEP_THINKER_SYSTEM, DEEP_THINKER_USER
from auditcoder.plan_agent.state import PlanState


def _fallback_deep_thinking(problem_structured: dict, reason: str) -> dict:
    constraints = problem_structured.get("constraints", {}) or {}
    data_ranges = constraints.get("data_ranges", []) if isinstance(constraints, dict) else []
    scale_hint = ""
    if data_ranges:
        scale_hint = "; ".join(
            f"{r.get('var')} <= {r.get('max')}"
            for r in data_ranges
            if isinstance(r, dict) and r.get("var") and r.get("max") is not None
        )

    return {
        "problem_nature": "Competitive-programming task requiring a correct efficient algorithm under the stated constraints.",
        "surface_traps": [
            {
                "trap": "Using a straightforward simulation or quadratic enumeration without checking input scale.",
                "why_it_fails": f"It may exceed time limits when constraints are large. {scale_hint}".strip(),
            },
            {
                "trap": "Ignoring edge cases implied by the input/output format.",
                "why_it_fails": "Boundary cases can change feasibility or ordering requirements.",
            },
        ],
        "structural_insights": [
            "Respect the problem's ordering and feasibility constraints before optimizing output.",
            "Use the stated constraints to rule out superlinear or quadratic approaches when n is large.",
        ],
        "solution_space": [
            {
                "family": "Greedy/data-structure based construction",
                "pros": "Often fits ordering or lexicographic constraints efficiently.",
                "cons": "Requires a proof that each local choice preserves feasibility.",
            },
            {
                "family": "Graph/order validation plus constructive algorithm",
                "pros": "Useful when dependencies or stack/order constraints define feasible states.",
                "cons": "Implementation must carefully maintain invariants.",
            },
        ],
        "critical_questions": [
            "What invariant characterizes feasibility?",
            "Which local choice preserves feasibility while optimizing the requested output?",
        ],
        "thinking_summary": (
            "Use constraints to select an efficient constructive strategy, validate feasibility early, "
            "and preserve the ordering invariant required by the problem."
        ),
        "analysis_error": reason,
    }


def deep_thinker(state: PlanState) -> dict:
    llm = get_llm(max_tokens=4096)
    problem_json = json.dumps(state["problem_structured"], ensure_ascii=False, indent=2)

    failure_context = state.get("failure_context", "")
    if failure_context:
        user_content = DEEP_THINKER_REPLAN_USER.format(
            problem_structured=problem_json,
            failure_context=failure_context,
        )
    else:
        user_content = DEEP_THINKER_USER.format(problem_structured=problem_json)

    user_content += (
        "\n\nOutput must be concise: at most 2 surface_traps, 3 structural_insights, "
        "2 solution families, 2 critical_questions, and each string under 35 words."
    )

    messages = [
        SystemMessage(content=DEEP_THINKER_SYSTEM),
        HumanMessage(content=user_content),
    ]
    try:
        response = llm.invoke(messages)
        parsed = extract_json(response)
    except Exception as exc:
        fallback = _fallback_deep_thinking(
            state["problem_structured"],
            f"{type(exc).__name__}: {str(exc)[:300]}",
        )
        return {"deep_thinking": fallback, "messages": messages}

    if not isinstance(parsed, dict):
        fallback = _fallback_deep_thinking(
            state["problem_structured"],
            "deep_thinker returned non-object JSON",
        )
        return {"deep_thinking": fallback, "messages": messages + [response]}

    return {"deep_thinking": parsed, "messages": messages + [response]}
