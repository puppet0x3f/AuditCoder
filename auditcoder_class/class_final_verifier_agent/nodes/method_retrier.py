"""Node: retry the failing method with error context as feedback."""

from __future__ import annotations

import re
from typing import Any

from auditcoder_class.class_final_verifier_agent.state import ClassFinalVerifierState
from auditcoder_class.method_agents._utils import extract_test_method_bodies


def method_retrier(state: ClassFinalVerifierState) -> dict[str, Any]:
    from auditcoder_class.method_agents.agents.dp_agent import build_dp_agent_graph
    from auditcoder_class.method_agents.agents.greedy_agent import build_greedy_agent_graph
    from auditcoder_class.method_agents.agents.binary_search_agent import build_binary_search_agent_graph
    from auditcoder_class.method_agents.agents.string_agent import build_string_agent_graph
    from auditcoder_class.method_agents.agents.segment_tree_agent import build_segment_tree_agent_graph
    from auditcoder_class.method_agents.agents.monotonic_queue_agent import build_monotonic_queue_agent_graph
    from auditcoder_class.method_agents.agents.graph_bfs_agent import build_graph_bfs_agent_graph
    from auditcoder_class.method_agents.agents.graph_dfs_agent import build_graph_dfs_agent_graph
    from auditcoder_class.method_agents.agents.union_find_agent import build_union_find_agent_graph
    from auditcoder_class.method_agents.agents.math_agent import build_math_agent_graph
    from auditcoder_class.method_agents.agents.sorting_agent import build_sorting_agent_graph
    from auditcoder_class.method_agents.agents.two_pointer_agent import build_two_pointer_agent_graph
    from auditcoder_class.method_agents.agents.simple_agent import build_simple_agent_graph

    _agents = {
        "dp": build_dp_agent_graph,
        "greedy": build_greedy_agent_graph,
        "binary_search": build_binary_search_agent_graph,
        "string": build_string_agent_graph,
        "segment_tree": build_segment_tree_agent_graph,
        "monotonic_queue": build_monotonic_queue_agent_graph,
        "graph_bfs": build_graph_bfs_agent_graph,
        "graph_dfs": build_graph_dfs_agent_graph,
        "union_find": build_union_find_agent_graph,
        "math": build_math_agent_graph,
        "sorting": build_sorting_agent_graph,
        "two_pointer": build_two_pointer_agent_graph,
        "simple": build_simple_agent_graph,
    }

    error_location = state.get("error_location")
    if not error_location:
        return {}

    method_name = error_location.get("method_name")
    ir = state["class_plan_ir"]
    method_map = {m["method_name"]: m for m in ir.get("method_nodes", [])}
    node = method_map.get(method_name)
    if not node:
        return {}

    agent_type = node.get("agent_type", "math")
    agent_builder = _agents.get(agent_type, _agents["math"])
    agent = agent_builder()

    exec_result = state.get("execution_result", {})
    combined_output = exec_result.get("stderr", "") + "\n" + exec_result.get("stdout", "")
    failed_assertions = []
    for expected, actual in re.findall(r"AssertionError:\s+(.+?)\s+!=\s+(.+)", combined_output):
        failed_assertions.append({
            "expected": expected.strip(),
            "actual": actual.strip(),
        })
    test_code = node.get("reasoning_test_code") or node.get("test_code", "")
    failed_test_names = re.findall(r'(?:FAIL|ERROR):\s+(\w+)\s+\(', combined_output)
    failed_test_bodies = extract_test_method_bodies(test_code, failed_test_names[:5])

    detail = error_location.get("detail", "")
    is_compile_issue = (
        error_location.get("error_type") == "structural_violation"
        and "Compile error" in detail
    )

    feedback = {
        "verdict": "reject",
        "summary": detail,
        "compile_ok": not is_compile_issue,
        "unittest_report": {
            "passed": False,
            "output": combined_output,
            "failures": 1,
            "errors": 0,
        },
        "test_code": test_code,
        "failed_assertions": failed_assertions,
        "failed_test_bodies": failed_test_bodies,
    }

    # Pass already-generated sibling methods (excluding the failing one) for context
    method_bodies = state.get("method_bodies", {})
    generated = {k: v for k, v in method_bodies.items() if k != method_name}

    result = agent.invoke({
        "method_node": node,
        "class_plan_ir": ir,
        "generated_methods": generated,
        "feedback": feedback,
    })
    new_body = result.get("method_body", "")

    if not new_body:
        return {}

    updated_bodies = {**method_bodies, method_name: new_body}
    retried = list(state.get("retried_methods", [])) + [method_name]
    return {"method_bodies": updated_bodies, "retried_methods": retried}
