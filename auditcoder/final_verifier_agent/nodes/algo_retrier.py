"""Node: algo_retrier — re-invoke the matching algo agent for the replanned task node."""

from __future__ import annotations

from typing import Any

from auditcoder.algo_agents.agents.binary_search_agent import build_binary_search_agent_graph
from auditcoder.algo_agents.agents.dp_agent import build_dp_agent_graph
from auditcoder.algo_agents.agents.graph_bfs_agent import build_graph_bfs_agent_graph
from auditcoder.algo_agents.agents.graph_dfs_agent import build_graph_dfs_agent_graph
from auditcoder.algo_agents.agents.greedy_agent import build_greedy_agent_graph
from auditcoder.algo_agents.agents.math_agent import build_math_agent_graph
from auditcoder.algo_agents.agents.monotonic_queue_agent import build_monotonic_queue_agent_graph
from auditcoder.algo_agents.agents.segment_tree_agent import build_segment_tree_agent_graph
from auditcoder.algo_agents.agents.sorting_agent import build_sorting_agent_graph
from auditcoder.algo_agents.agents.string_agent import build_string_agent_graph
from auditcoder.algo_agents.agents.two_pointer_agent import build_two_pointer_agent_graph
from auditcoder.algo_agents.agents.union_find_agent import build_union_find_agent_graph
from auditcoder.final_verifier_agent.state import FinalVerifierState
from auditcoder.schemas.contract_graph import ensure_catg, register_code_owner

_AGENT_BUILDERS = {
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
}


def algo_retrier(state: FinalVerifierState) -> dict[str, Any]:
    updated_task_node = state.get("updated_task_node") or {}
    task_plan_ir = ensure_catg(state.get("task_plan_ir", {}))

    # Branch replanning can return more than one replacement node.  The
    # deterministic wrapper still limits generation to the declared region;
    # frozen nodes never enter this loop.
    replacement_nodes = state.get("updated_task_nodes") or {}
    if not replacement_nodes and updated_task_node:
        replacement_nodes = {updated_task_node.get("id", ""): updated_task_node}

    code_snippets = dict(state.get("code_snippets") or {})
    # The bounded retry must see the concrete global witness.  Previously the
    # retrier only received the replanned node, so it could regenerate the
    # same algorithm without knowing which input failed, whether the failure
    # was a timeout, or which output contract was violated.
    location = state.get("error_location") or {}
    failing = location.get("failing_case") or location.get("anchor") or {}
    feedback = {
        "verdict": "reject",
        "summary": (
            f"global {location.get('error_type', 'verification')} failure: "
            f"{location.get('detail', '')}"
        ).strip(),
        "strong_signals": {
            "compile_ok": True,
            "sample_tests_passed": False,
            "bruteforce_match": False,
            "resource_ok": location.get("error_type") not in {"timeout", "memory_error"},
        },
        "medium_signals": {
            "contract_violations": [],
        },
        "counterexample": {
            "minimal_input": failing.get("input", ""),
            "expected_output": failing.get("expected", ""),
            "actual_output": failing.get("actual", ""),
            "error_category": location.get("error_type", "unknown"),
            "suspected_reason": location.get("rule", location.get("method", "")),
        },
    }
    generated: list[str] = []
    for task_id, task_node in replacement_nodes.items():
        task_id = str(task_id)
        agent_type: str = str(task_node.get("agent_type", ""))

        builder = _AGENT_BUILDERS.get(agent_type)
        if builder is None:
            continue

        agent = builder()
        result = agent.invoke(
            {
                "task_node": task_node,
                "task_plan_ir": task_plan_ir,
                "feedback": feedback,
            }
        )
        snippet: str = result.get("code_snippet", "")
        if not snippet:
            continue
        code_snippets[task_id] = snippet
        task_plan_ir = register_code_owner(task_plan_ir, task_id, snippet)
        generated.append(task_id)

    if not generated:
        if replacement_nodes and all(
            str(node.get("agent_type", "")) not in _AGENT_BUILDERS
            for node in replacement_nodes.values()
        ):
            # Preserve the historical API for an unregistered executor.
            return {}
        return {
            "candidate_code_snippets": dict(code_snippets),
            "task_plan_ir": task_plan_ir,
            "repair_generation": {"status": "no_code_generated"},
        }

    return {
        "task_plan_ir": task_plan_ir,
        "code_snippets": code_snippets,
        "candidate_code_snippets": dict(code_snippets),
        "repair_generation": {
            "status": "generated",
            "generated_nodes": generated,
        },
    }
