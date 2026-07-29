"""Dispatcher — routes each method in class_plan_ir to the matching method agent.

For each method (in execution_order):
1. Build partial class (real bodies for completed methods, stubs for the rest).
2. Invoke matching method agent to generate a method_body.
3. Verify with class_verifier_agent (compile + per-method unittest).
4. If rejected, retry once with verifier feedback.
5. Accumulate generated_methods for context injection into subsequent agents.
"""

from __future__ import annotations

import re
from typing import Any

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
from auditcoder_class.method_agents._utils import extract_test_method_bodies
from auditcoder_class.class_verifier_agent import build_class_verifier_agent_graph
from auditcoder_class.class_assembler_agent.agent import build_partial_class


def _called_class_methods(test_code: str, method_names: set[str]) -> set[str]:
    return {
        name
        for name in method_names
        if re.search(r'\.' + re.escape(name) + r'\s*\(', test_code)
    }


def _select_verification_test_code(
    node: dict[str, Any],
    class_plan_ir: dict[str, Any],
    available_methods: set[str],
) -> str:
    test_code = node.get("test_code", "")
    if test_code:
        return test_code

    reasoning_test_code = node.get("reasoning_test_code", "")
    if not reasoning_test_code:
        return ""

    method_names = {
        m["method_name"]
        for m in class_plan_ir.get("method_nodes", [])
        if m.get("method_name")
    }
    called_methods = _called_class_methods(reasoning_test_code, method_names)
    if called_methods and called_methods.issubset(available_methods):
        return reasoning_test_code
    return ""


def _augment_retry_feedback(report: dict[str, Any], test_code: str) -> dict[str, Any]:
    augmented = dict(report)
    if test_code:
        augmented["test_code"] = test_code

    unittest_report = report.get("unittest_report", {})
    output = unittest_report.get("output", "") if isinstance(unittest_report, dict) else ""
    failed_assertions = []
    for expected, actual in re.findall(r"AssertionError:\s+(.+?)\s+!=\s+(.+)", output):
        failed_assertions.append({
            "expected": expected.strip(),
            "actual": actual.strip(),
        })
    if failed_assertions:
        augmented["failed_assertions"] = failed_assertions

    failed_test_names = re.findall(r'(?:FAIL|ERROR):\s+(\w+)\s+\(', output)
    failed_test_bodies = extract_test_method_bodies(test_code, failed_test_names[:5])
    if failed_test_bodies:
        augmented["failed_test_bodies"] = failed_test_bodies
    return augmented


def _build_class_verifier_input(
    method_name: str,
    method_body: str,
    class_plan_ir: dict[str, Any],
    generated_methods: dict[str, str],
    test_code: str | None = None,
) -> dict[str, Any]:
    """Build ClassVerifierState input for a single method."""
    # Build partial class: completed methods have real bodies; remaining have stubs
    updated_generated = {**generated_methods, method_name: method_body}
    partial_class_code = build_partial_class(class_plan_ir, updated_generated)

    method_map = {m["method_name"]: m for m in class_plan_ir.get("method_nodes", [])}
    node = method_map.get(method_name, {})

    return {
        "method_name": method_name,
        "partial_class_code": partial_class_code,
        "test_code": node.get("test_code", "") if test_code is None else test_code,
        "import_statements": class_plan_ir.get("import_statements", []),
        "class_name": class_plan_ir.get("class_name", ""),
    }


def dispatch_method_agents(
    class_plan_ir: dict[str, Any],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Execute all methods in class_plan_ir and return generated bodies and reports.

    Returns:
        method_bodies: {method_name → method_body_str}
        verification_reports: {method_name → verification_report_dict}
    """
    _agents = {
        "dp": build_dp_agent_graph(),
        "greedy": build_greedy_agent_graph(),
        "binary_search": build_binary_search_agent_graph(),
        "string": build_string_agent_graph(),
        "segment_tree": build_segment_tree_agent_graph(),
        "monotonic_queue": build_monotonic_queue_agent_graph(),
        "graph_bfs": build_graph_bfs_agent_graph(),
        "graph_dfs": build_graph_dfs_agent_graph(),
        "union_find": build_union_find_agent_graph(),
        "math": build_math_agent_graph(),
        "sorting": build_sorting_agent_graph(),
        "two_pointer": build_two_pointer_agent_graph(),
        "simple": build_simple_agent_graph(),
    }
    _verifier = build_class_verifier_agent_graph()

    method_map: dict[str, dict] = {
        m["method_name"]: m for m in class_plan_ir.get("method_nodes", [])
    }
    generated_methods: dict[str, str] = {}
    method_bodies: dict[str, str] = {}
    verification_reports: dict[str, dict[str, Any]] = {}

    for method_name in class_plan_ir.get("execution_order", []):
        node = method_map.get(method_name)
        if not node:
            continue

        agent_type = node.get("agent_type", "math")
        agent = _agents.get(agent_type, _agents["math"])

        # --- first pass ---
        result = agent.invoke({
            "method_node": node,
            "class_plan_ir": class_plan_ir,
            "generated_methods": dict(generated_methods),
        })
        body = result.get("method_body", "")

        # Skip verification only when neither direct tests nor safe dependency tests are available.
        available_methods = set(generated_methods) | {method_name}
        test_code = _select_verification_test_code(node, class_plan_ir, available_methods)
        if not test_code:
            method_bodies[method_name] = body
            generated_methods[method_name] = body
            verification_reports[method_name] = {"verdict": "skip", "summary": "no test code"}
            continue

        verifier_input = _build_class_verifier_input(
            method_name, body, class_plan_ir, generated_methods, test_code
        )
        verifier_result = _verifier.invoke(verifier_input)
        report = verifier_result.get("verification_report", {})

        # --- retry on rejection ---
        if report.get("verdict") == "reject":
            retry_result = agent.invoke({
                "method_node": node,
                "class_plan_ir": class_plan_ir,
                "generated_methods": dict(generated_methods),
                "feedback": _augment_retry_feedback(report, test_code),
            })
            retry_body = retry_result.get("method_body", "") or body

            retry_verifier_input = _build_class_verifier_input(
                method_name, retry_body, class_plan_ir, generated_methods, test_code
            )
            retry_result_v = _verifier.invoke(retry_verifier_input)
            report = retry_result_v.get("verification_report", report)
            body = retry_body

        method_bodies[method_name] = body
        generated_methods[method_name] = body
        verification_reports[method_name] = report

    return method_bodies, verification_reports
