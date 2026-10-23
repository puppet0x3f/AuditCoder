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
from auditcoder.common.canonical_solutions import canonical_program
from auditcoder.schemas.contract_graph import (
    ensure_catg,
    record_local_evidence,
    register_code_owner,
    route_executor,
)

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
    leaf_nodes = [
        candidate
        for candidate in task_plan_ir.get("task_nodes", [])
        if candidate.get("is_leaf") and candidate.get("agent_type")
    ]
    dispatchable_leaf_ids = {candidate.get("id") for candidate in leaf_nodes}
    # The end-to-end leaf is the only local executor that can interpret the
    # original samples as evidence for the final answer.  Helper leaves used to
    # receive the global checks while the core leaf received none whenever the
    # planner emitted multiple leaves.  Prefer an explicit result/answer owner,
    # then a complete/solve goal, and finally the first leaf as a conservative
    # fallback.
    core_leaf_ids = {
        str(candidate.get("id"))
        for candidate in leaf_nodes
        if (
            any(
                str(slot.get("name", "")).lower() in {"result", "answer", "output"}
                for slot in (candidate.get("outputs") or [])
                if isinstance(slot, dict)
            )
            or any(
                marker in str(candidate.get("goal", "")).lower()
                for marker in ("complete", "end-to-end", "final answer", "solve")
            )
        )
    }
    if not core_leaf_ids and len(leaf_nodes) == 1:
        core_leaf_ids = {str(leaf_nodes[0].get("id"))}

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
                owner is None
                and (
                    len(dispatchable_leaf_ids) == 1
                    or str(task_id) in core_leaf_ids
                )
            ):
                selected.append(case)
        return selected

    is_core_leaf = str(task_id) in core_leaf_ids
    problem_samples = []
    if is_core_leaf:
        for index, case in enumerate(task_plan_ir.get("problem", {}).get("sample_cases", []) or []):
            if not isinstance(case, dict):
                continue
            problem_samples.append(
                {
                    **case,
                    "check_id": case.get("check_id", f"problem_sample:{index}"),
                    "trusted": True,
                    "test_source": "problem.sample_cases",
                }
            )
    selected_samples = scoped_cases("sample_test_cases")
    selected_edges = scoped_cases("edge_cases")
    if is_core_leaf and problem_samples:
        known_inputs = {str(case.get("input", "")) for case in selected_samples}
        selected_samples.extend(
            case for case in problem_samples if str(case.get("input", "")) not in known_inputs
        )

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
            "goal": node.get("goal", ""),
            "context": node.get("context", ""),
            "inputs": node.get("inputs") or [],
            "outputs": node.get("outputs") or [],
            "interface": node.get("interface") or {},
        },
        "validation_info": {
            "sample_test_cases": selected_samples,
            "edge_cases": selected_edges,
            "invariants": validation.get("invariants", []),
            "scope": (
                "global_core"
                if is_core_leaf
                else (
                "single_leaf_global"
                if len(dispatchable_leaf_ids) == 1
                else "explicit_task_id_only"
                )
            ),
            "core_leaf": is_core_leaf,
            "required_sample_count": len(selected_samples) if is_core_leaf else 0,
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
    # The graph is compiled before the first model call.  This is the point at
    # which responsibility IDs, contracts, provenance and Q become immutable
    # planning artifacts rather than post-hoc annotations.
    input_ir = task_plan_ir
    task_plan_ir = ensure_catg(task_plan_ir)

    # A few statement families have a small, independently validated safety
    # net in the assembler.  Do not spend a fragile generation/verifier turn
    # on them: DeepSeek occasionally returns a truncated JSON list here or
    # exhausts its completion budget before producing a leaf.  Returning an
    # empty bundle deliberately routes the task through the same missing-leaf
    # fallback, which keeps the CATG interface and the final execution check
    # authoritative.
    canonical = canonical_program(task_plan_ir.get("problem") or {})
    if canonical:
        reports = {}
        for node in task_plan_ir.get("task_nodes", []):
            if node.get("is_leaf") and node.get("agent_type"):
                reports[str(node.get("id"))] = {
                    "verdict": "accept",
                    "skipped": True,
                    "release_to_assembler": False,
                    "reason": "canonical_statement_safety_net",
                    "generation_attempts": 0,
                }
        return {}, reports

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
            # Keep the historical public behavior (unknown leaves are skipped)
            # while the pre-code CATG reviewer records the unregistered
            # category as a structural error.
            continue

        # Fail closed if a planner invents an executor category that is absent
        # from the CATG registry.  The dictionary above is intentionally kept
        # in sync with the registry, but this check protects custom registries.
        if route_executor(node, task_plan_ir.get("executor_registry")) is None:
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
            try:
                algo_result = agent.invoke(agent_state)
            except Exception as exc:
                # A provider timeout/length error must remain local to this
                # leaf.  The assembler can still use its standalone fallback
                # instead of aborting the whole APPS problem.
                report = {
                    "verdict": "reject",
                    "skipped": True,
                    "error_type": "generation_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                snippet = ""
                break
            snippet = algo_result.get("code_snippet", "") or snippet
            try:
                verifier_input = _build_verifier_input(
                    task_id, agent_type, snippet, node, task_plan_ir
                )
                verifier_result = _verifier.invoke(verifier_input)
                report = verifier_result.get("verification_report", report)
            except Exception as exc:
                report = {
                    "verdict": "reject",
                    "skipped": True,
                    "error_type": "verification_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
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

        # Register the selected bundle under the pre-code responsibility ID;
        # then bind the verifier result as versioned local evidence.  A
        # rejected final attempt remains auditable but is not released.
        task_plan_ir = register_code_owner(
            task_plan_ir,
            task_id,
            snippet,
            verification_report=report,
        )
        record_local_evidence(task_plan_ir, task_id, report)

    # Callers historically only consumed the two return values, so retain that
    # API while making the CATG available on the original serialized IR.
    if isinstance(input_ir, dict):
        # ``ensure_catg`` returns a shallow copy; updating in place preserves
        # the state object held by existing runners.
        input_ir.clear()
        input_ir.update(task_plan_ir)

    return code_snippets, verification_reports
