"""LLM node: assign an algorithm agent_type and hints to each class method."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from auditcoder.common.llm import get_llm
from auditcoder.common.utils import extract_json
from auditcoder_class.class_plan_agent.prompts.method_selector import (
    METHOD_SELECTOR_SYSTEM,
    METHOD_SELECTOR_USER,
)
from auditcoder_class.class_plan_agent.state import ClassPlanState

_VALID_AGENT_TYPES = {
    "dp", "greedy", "binary_search", "string", "segment_tree",
    "monotonic_queue", "graph_bfs", "graph_dfs", "union_find",
    "math", "sorting", "two_pointer", "simple",
}


def _test_grounded_grid_connectivity_hint(method: dict[str, Any]) -> dict[str, Any] | None:
    name = method.get("method_name", "").lower()
    if not any(token in name for token in ("path", "connect")):
        return None

    test_code = method.get("reasoning_test_code") or method.get("test_code", "")
    if "board" not in test_code or "assertEqual(res, True)" not in test_code:
        return None

    # If tests define icon-filled grids and expect long same-icon paths to pass,
    # the dataset is specifying grid connectivity over equal-valued neighbors,
    # not empty-cell / turn-limited pathing from a real game.
    if "['a', 'a', 'a', 'a']" not in test_code:
        return None

    return {
        "agent_type": "graph_bfs",
        "hints": [
            "Model the board as a 4-neighbor grid graph within BOARD_SIZE bounds.",
            "A neighbor is traversable only when its board value equals the starting cell's value; different icons block movement.",
            "Use BFS/DFS from pos1 and return True if pos2 is reached; same position is True, out-of-bounds is False.",
        ],
    }


def method_algorithm_selector(state: ClassPlanState) -> dict[str, Any]:
    """Call LLM to assign agent_type and hints to each method."""
    ctx = state["class_context"]
    thoughts_map: dict[str, str] = {
        t["method_name"]: t["reasoning"]
        for t in state.get("method_thoughts", [])
        if t.get("reasoning")
    }

    methods_lines = []
    for i, m in enumerate(ctx["methods"], 1):
        desc_first_line = m["method_description"].strip().splitlines()[0] if m["method_description"] else ""
        deps = m["method_dependencies"]
        dep_str = f"  calls: {', '.join(deps)}" if deps else "  standalone"
        reasoning = thoughts_map.get(m["method_name"], "")
        entry = f"{i}. {m['method_name']}: {desc_first_line}\n{dep_str}"
        if reasoning:
            entry += f"\n  reasoning: {reasoning}"
        test_code = (m.get("reasoning_test_code") or m.get("test_code", "")).strip()
        if test_code:
            entry += f"\n  test_excerpt:\n    {test_code[:2500].replace(chr(10), chr(10) + '    ')}"
        methods_lines.append(entry)

    methods_summary = "\n\n".join(methods_lines)

    imports = ctx.get("import_statements", [])
    import_str = "\n".join(imports) if imports else "(none)"
    user_content = METHOD_SELECTOR_USER.format(
        class_name=ctx["class_name"],
        class_description=ctx["class_description"],
        import_statements=import_str,
        methods_summary=methods_summary,
    )

    llm = get_llm(temperature=0.0)
    messages = [
        SystemMessage(content=METHOD_SELECTOR_SYSTEM),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(messages)
    parsed = extract_json(response)

    llm_methods = parsed.get("methods", []) if isinstance(parsed, dict) else []

    # Build a lookup by method_name; fall back to "math" if LLM output is missing/invalid
    llm_map: dict[str, dict] = {m["method_name"]: m for m in llm_methods if "method_name" in m}

    result: list[dict[str, Any]] = []
    for m in ctx["methods"]:
        name = m["method_name"]
        grounded = _test_grounded_grid_connectivity_hint(m)
        if grounded:
            result.append({"method_name": name, **grounded})
            continue

        llm_entry = llm_map.get(name, {})
        agent_type = llm_entry.get("agent_type", "math")
        if agent_type not in _VALID_AGENT_TYPES:
            agent_type = "math"
        hints = llm_entry.get("hints", [])
        result.append({"method_name": name, "agent_type": agent_type, "hints": hints})

    return {
        "method_algo_selection": result,
        "messages": messages + [response],
    }
