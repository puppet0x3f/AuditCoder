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
    task_plan_ir = state.get("task_plan_ir", {})

    task_id: str = updated_task_node.get("id", "")
    agent_type: str = updated_task_node.get("agent_type", "")

    builder = _AGENT_BUILDERS.get(agent_type)
    if builder is None:
        return {}

    agent = builder()
    result = agent.invoke({"task_node": updated_task_node, "task_plan_ir": task_plan_ir})
    snippet: str = result.get("code_snippet", "")

    if not snippet:
        return {}

    code_snippets = dict(state.get("code_snippets") or {})
    code_snippets[task_id] = snippet

    return {"code_snippets": code_snippets}
