"""LLM node: reason about how to implement each method before algorithm selection."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from auditcoder.common.llm import get_llm
from auditcoder.common.utils import extract_json
from auditcoder_class.class_plan_agent.prompts.method_thinker import (
    METHOD_THINKER_SYSTEM,
    METHOD_THINKER_USER,
)
from auditcoder_class.class_plan_agent.state import ClassPlanState

_MAX_DESC_LINES = 8   # keep only first N lines of each method description
_HEAD_TESTS = 3       # number of test functions to keep from the start
_TAIL_TESTS = 3       # number of test functions to keep from the end

# Assert types that signal distinct return/exception branches
_ASSERT_PATTERN = re.compile(
    r'\b(assertTrue|assertFalse|assertIsNone|assertIsNotNone|assertEqual|assertNotEqual'
    r'|assertRaises|assertIn|assertNotIn)\b'
)

# Matches the start of each test method definition inside a TestCase class
_TEST_DEF_PATTERN = re.compile(r'(?=\n    def test_\w+\s*\()')


def _compress_test_code(test_code: str) -> str:
    """Keep the first _HEAD_TESTS and last _TAIL_TESTS test functions.

    Splits on `def test_` boundaries so edge-case tests (usually last) are
    never truncated by a character-count cap. Middle tests are replaced with
    a one-line summary. If there are few enough tests to fit, returns as-is.
    """
    # Split into [class_header, func1, func2, ...] segments
    parts = _TEST_DEF_PATTERN.split(test_code)
    # parts[0] is the class definition line; parts[1:] are individual test functions
    header = parts[0]
    funcs = parts[1:]

    total = len(funcs)
    if total <= _HEAD_TESTS + _TAIL_TESTS:
        return test_code  # short enough — return unchanged

    head = funcs[:_HEAD_TESTS]
    tail = funcs[total - _TAIL_TESTS:]
    omitted = total - _HEAD_TESTS - _TAIL_TESTS
    placeholder = f"\n    # ... {omitted} test(s) omitted ...\n"
    return header + "".join(head) + placeholder + "".join(tail)


def method_thinker(state: ClassPlanState) -> dict[str, Any]:
    """Call LLM to reason about how each method should be implemented."""
    ctx = state["class_context"]

    methods_lines = []
    for i, m in enumerate(ctx["methods"], 1):
        deps = m["method_dependencies"]
        dep_str = f"  calls: {', '.join(deps)}" if deps else "  standalone"
        field_deps = m["field_dependencies"]
        field_str = f"  uses fields: {', '.join(field_deps)}" if field_deps else ""
        desc_lines = m["method_description"].strip().splitlines()[:_MAX_DESC_LINES]
        desc = "\n    ".join(desc_lines)
        block = f"{i}. {m['method_name']}\n{dep_str}"
        if field_str:
            block += f"\n{field_str}"
        block += f"\n  description:\n    {desc}"
        test_code = (m.get("reasoning_test_code") or m.get("test_code", "")).strip()
        if test_code:
            # Summarise distinct assert types so LLM immediately sees return-value diversity
            assert_types = sorted(set(_ASSERT_PATTERN.findall(test_code)))
            if assert_types:
                block += f"\n  assert_types_used: {', '.join(assert_types)}"
            compressed = _compress_test_code(test_code)
            block += f"\n  test_cases:\n    {compressed.replace(chr(10), chr(10) + '    ')}"
        methods_lines.append(block)

    methods_detail = "\n\n".join(methods_lines)

    user_content = METHOD_THINKER_USER.format(
        class_name=ctx["class_name"],
        class_description=ctx["class_description"],
        class_constructor=ctx.get("class_constructor", ""),
        fields=", ".join(ctx.get("fields", [])) or "none",
        methods_detail=methods_detail,
    )

    # Disable json_object mode: DeepSeek returns None on long prompts with that mode.
    # extract_json handles plain-text JSON extraction.
    llm = get_llm(temperature=0.0, json_mode=False)
    messages = [
        SystemMessage(content=METHOD_THINKER_SYSTEM),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(messages)
    if response is None or not response.content:
        parsed = {}
    else:
        try:
            parsed = extract_json(response)
        except ValueError:
            parsed = {}

    thoughts_list = parsed.get("thoughts", []) if isinstance(parsed, dict) else []
    thoughts_map: dict[str, str] = {
        t["method_name"]: t.get("reasoning", "")
        for t in thoughts_list
        if "method_name" in t
    }

    method_thoughts: list[dict[str, Any]] = [
        {"method_name": m["method_name"], "reasoning": thoughts_map.get(m["method_name"], "")}
        for m in ctx["methods"]
    ]

    return {
        "method_thoughts": method_thoughts,
        "messages": messages + ([response] if response is not None else []),
    }
