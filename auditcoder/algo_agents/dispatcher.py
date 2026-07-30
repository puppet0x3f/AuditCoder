"""Dispatcher — routes each leaf task in task_plan_ir to the matching algorithm agent.

Each leaf task is processed as follows:
1. The matching algo agent generates a code snippet.
2. The verifier_agent validates the snippet.
3. If the verdict is "reject", the error report is fed back to the algo agent
   for one bounded retry.
4. A snippet that is still rejected is not released to the assembler.
"""

from __future__ import annotations

from typing import Any

from auditcoder.algo_agents.agents.dp_agent import build_dp_agent_graph
from auditcoder.algo_agents.agents.greedy_agent import build_greedy_agent_graph
from auditcoder.algo_agents.agents.binary_search_agent import build_binary_search_agent_graph
from auditcoder.algo_agents.agents.string_agent import build_string_agent_graph
from auditcoder.algo_agents.agents.segment_tree_agent import build_segment_tree_agent_graph
from auditcoder.algo_agents.agents.monotonic_queue_agent import build_monotonic_queue_agent_graph
from auditcoder.algo_agents.agents.graph_bfs_agent import build_graph_bfs_agent_graph
from auditcoder.algo_agents.agents.graph_dfs_agent import build_graph_dfs_agent_graph
from auditcoder.algo_agents.agents.union_find_agent import build_union_find_agent_graph
from auditcoder.algo_agents.agents.math_agent import build_math_agent_graph
from auditcoder.algo_agents.agents.sorting_agent import build_sorting_agent_graph
from auditcoder.algo_agents.agents.two_pointer_agent import build_two_pointer_agent_graph
from auditcoder.verifier_agent import build_verifier_agent_graph

MAX_GENERATION_ATTEMPTS = 3


def _build_verifier_input(
    task_id: str,
    agent_type: str,
    code_snippet: str,
    node: dict[str, Any],
    task_plan_ir: dict[str, Any],
) -> dict[str, Any]:
    """Construct the VerifierState input dict for a single leaf task."""
    complexity_budget = node.get("complexity_budget", "")
    if isinstance(complexity_budget, str):
        complexity_target: dict[str, Any] = {"time": complexity_budget}
    elif isinstance(complexity_budget, dict):
        complexity_target = dict(complexity_budget)
    else:
        complexity_target = {}
    validation = task_plan_ir.get("validation") or {}
    dispatchable_leaf_ids = {
        candidate.get("id")
        for candidate in task_plan_ir.get("task_nodes", [])
        if candidate.get("is_leaf") and candidate.get("agent_type")
    }

    def scoped_cases(field: str) -> list[dict[str, Any]]:
        cases = validation.get(field, [])
        if not isinstance(cases, list):
            return []
        selected: list[dict[str, Any]] = []
        for case in cases:
            if not isinstance(case, dict):
                continue
            owner = case.get("task_id")
            if owner == task_id or (
                owner is None and len(dispatchable_leaf_ids) == 1
            ):
                selected.append(case)
        return selected

    return {
        "code_snippet": {
            "task_id": task_id,
            "agent_type": agent_type,
            "source_code": code_snippet,
            "explanation": "",
        },
        "task_contract": {
            "task_id": task_id,
            "agent_type": agent_type,
            "interface": node.get("interface") or {},
        },
        "validation_info": {
            "sample_test_cases": scoped_cases("sample_test_cases"),
            "edge_cases": scoped_cases("edge_cases"),
            "invariants": validation.get("invariants", []),
            "scope": (
                "single_leaf_global"
                if len(dispatchable_leaf_ids) == 1
                else "explicit_task_id_only"
            ),
        },
        "complexity_target": complexity_target,
        "problem_info": task_plan_ir.get("problem") or {},
    }


def dispatch_algo_agents(
    task_plan_ir: dict[str, Any],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Execute all leaf tasks in task_plan_ir and return generated code and reports.

    For each leaf task:
    - Generate code with the matching algo agent.
    - Verify with verifier_agent.
    - If rejected, retry once with verifier feedback injected into the agent state.
    - Never return a still-rejected snippet to the assembler.

    Args:
        task_plan_ir: The full TaskPlanIR dict produced by plan_agent.

    Returns:
        A tuple of:
        - code_snippets: dict mapping task_id → final code snippet string.
        - verification_reports: dict mapping task_id → final verification_report dict.
          (first-pass report when passed; retry report when retried)
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
    }
    _verifier = build_verifier_agent_graph()

    node_map: dict[str, dict] = {
        n["id"]: n for n in task_plan_ir.get("task_nodes", []) if "id" in n
    }
    code_snippets: dict[str, str] = {}
    verification_reports: dict[str, dict[str, Any]] = {}

    for task_id in task_plan_ir.get("execution_order", []):
        node = node_map.get(task_id)
        if not node or not node.get("is_leaf"):
            continue

        agent_type = node.get("agent_type")
        if agent_type not in _agents:
            continue

        agent = _agents[agent_type]

        snippet = ""
        report: dict[str, Any] = {}
        feedback: dict[str, Any] | None = None
        attempts = 0
        while attempts < MAX_GENERATION_ATTEMPTS:
            attempts += 1
            agent_state: dict[str, Any] = {
                "task_node": node,
                "task_plan_ir": task_plan_ir,
            }
            if feedback is not None:
                agent_state["feedback"] = feedback
            algo_result = agent.invoke(agent_state)
            snippet = algo_result.get("code_snippet", "") or snippet
            verifier_input = _build_verifier_input(
                task_id, agent_type, snippet, node, task_plan_ir
            )
            verifier_result = _verifier.invoke(verifier_input)
            report = verifier_result.get("verification_report", report)
            if report.get("verdict") != "reject":
                break
            feedback = report

        report = dict(report)
        report["generation_attempts"] = attempts
        report["release_to_assembler"] = bool(
            snippet and report.get("verdict") != "reject"
        )
        if report["release_to_assembler"]:
            code_snippets[task_id] = snippet
        verification_reports[task_id] = report

    return code_snippets, verification_reports
