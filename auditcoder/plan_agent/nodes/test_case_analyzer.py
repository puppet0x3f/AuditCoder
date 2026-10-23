"""Node: test_case_analyzer — 提取样例并推导边界测试用例。"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from auditcoder.plan_agent.llm import get_llm, invoke_llm
from auditcoder.plan_agent.nodes._utils import extract_json
from auditcoder.plan_agent.prompts.reviewer import (
    TEST_CASE_ANALYZER_SYSTEM,
    TEST_CASE_ANALYZER_USER,
)
from auditcoder.plan_agent.state import PlanState


def _fallback_test_cases(problem_structured: dict) -> dict:
    """Build validation info without an LLM when test-case analysis fails."""
    sample_test_cases: list[dict] = []
    for case in problem_structured.get("sample_cases", []) or []:
        if not isinstance(case, dict):
            continue
        expected = case.get("expected_output", case.get("output", ""))
        sample_test_cases.append({
            "input": str(case.get("input", "")),
            "expected_output": str(expected),
        })

    return {
        "sample_test_cases": sample_test_cases,
        "edge_cases": [],
        "invariants": [
            "Output must exactly follow the problem output format.",
            "The solution must handle all constraints stated in the problem.",
        ],
    }


def test_case_analyzer(state: PlanState) -> dict:
    llm = get_llm()
    problem_structured = state["problem_structured"]
    problem_json = json.dumps(problem_structured, ensure_ascii=False, indent=2)

    messages = [
        SystemMessage(content=TEST_CASE_ANALYZER_SYSTEM),
        HumanMessage(
            content=TEST_CASE_ANALYZER_USER.format(problem_structured=problem_json)
        ),
    ]
    try:
        response = invoke_llm(llm, messages)
        parsed = extract_json(response)
    except Exception as exc:
        fallback = _fallback_test_cases(problem_structured)
        fallback["analysis_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return {"test_cases": fallback, "messages": messages}

    if not isinstance(parsed, dict):
        fallback = _fallback_test_cases(problem_structured)
        fallback["analysis_error"] = "test_case_analyzer returned non-object JSON"
        return {"test_cases": fallback, "messages": messages + [response]}

    return {"test_cases": parsed, "messages": messages + [response]}
