"""Replan only the evidence-selected repair region.

The model may propose new hints or a branch-local plan, but the deterministic
wrapper owns identity, contract, parent, dependency, and frozen-complement
checks.  A malformed or over-broad proposal is reduced to the current
responsibility record instead of silently changing the graph boundary.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from langchain_core.messages import HumanMessage, SystemMessage

from auditcoder.common.llm import get_llm, invoke_llm
from auditcoder.common.utils import extract_json
from auditcoder.common.problem_guidance import strategy_card
from auditcoder.final_verifier_agent.audit import code_owner_ids, node_map
from auditcoder.final_verifier_agent.prompts.task_replanner import (
    TASK_REPLANNER_SYSTEM,
    TASK_REPLANNER_USER,
)
from auditcoder.final_verifier_agent.state import FinalVerifierState


# These fields define the pre-code responsibility address and the exposed
# contract.  They remain fixed for node-local repair.  Branch replacement can
# introduce fresh IDs, but never rewrites nodes outside the selected region.
_IMMUTABLE_NODE_FIELDS = {
    "id",
    "goal",
    "type",
    "is_leaf",
    "agent_type",
    "parent",
    "dependencies",
    "inputs",
    "outputs",
    "interface",
    "complexity_budget",
}


def _frozen_node(original: Mapping[str, Any], proposal: Mapping[str, Any]) -> dict[str, Any]:
    updated = {**dict(original), **dict(proposal)}
    for field in _IMMUTABLE_NODE_FIELDS:
        if field in original:
            updated[field] = original[field]
    updated["id"] = str(original.get("id", ""))
    return updated


def _problem_context(task_plan_ir: Mapping[str, Any]) -> str:
    problem = task_plan_ir.get("problem", {}) or {}
    card = strategy_card(problem)
    card_text = f"\n确定性策略卡（必须由失败证据和样例验证）：{card}" if card else ""
    return (
        f"题目：{problem.get('title', '')}\n"
        f"描述：{problem.get('description', '')}\n"
        f"输入格式：{problem.get('input_format', '')}\n"
        f"输出格式：{problem.get('output_format', '')}\n"
        f"约束：{json.dumps(problem.get('constraints', {}), ensure_ascii=False)}\n"
        f"验证不变量：{json.dumps((task_plan_ir.get('validation') or {}).get('invariants', []), ensure_ascii=False)}"
        f"{card_text}"
    )


def _parse_proposal(response_or_proposal: Any) -> dict[str, Any]:
    if isinstance(response_or_proposal, Mapping):
        return dict(response_or_proposal)
    parsed = extract_json(response_or_proposal)
    return parsed if isinstance(parsed, dict) else {}


def task_replanner(state: FinalVerifierState) -> dict[str, Any]:
    error_location = state.get("error_location") or {}
    scope = state.get("repair_scope") or {}
    task_plan_ir = state.get("task_plan_ir", {}) or {}
    node_index = node_map(task_plan_ir)

    owners = error_location.get("owners") or error_location.get("implicated_nodes") or []
    if isinstance(owners, str):
        owners = [owners]
    task_id = str(error_location.get("task_id") or (owners[0] if owners else ""))
    task_node = dict(node_index.get(task_id, {}))
    if not task_node:
        return {
            "repair_plan": {
                "status": "abstained",
                "reason": "located_owner_missing_from_graph",
            }
        }

    failing_case = error_location.get("failing_case", {}) or {}
    scope_text = json.dumps(scope, ensure_ascii=False, indent=2) if scope else "（无）"
    user_content = TASK_REPLANNER_USER.format(
        problem_context=_problem_context(task_plan_ir),
        task_node_json=json.dumps(task_node, ensure_ascii=False, indent=2),
        error_type=error_location.get("error_type", "unknown"),
        error_detail=error_location.get("detail", ""),
        failing_case=json.dumps(failing_case, ensure_ascii=False) if failing_case else "（无）",
        localization_method=error_location.get("method", error_location.get("rule", "unknown")),
        localization_confidence=error_location.get("confidence", "unknown"),
    )
    user_content += (
        "\n\nEvidence-selected repair boundary (authoritative; do not widen it):\n"
        f"{scope_text}\n"
        "Replan only responsibilities in `region`. Preserve every node and code "
        "bundle in `frozen_region`. Preserve all exposed interfaces, provenance "
        "bindings, and downstream calls. If an integration exception is required, "
        "declare its affected owners and supporting evidence before regeneration."
    )

    # Controlled experiments and offline tests can provide a deterministic
    # proposal.  Production uses the normal replanner LLM.
    proposal_input = state.get("repair_proposal")
    if proposal_input is None:
        llm = get_llm(temperature=0.3)
        response = invoke_llm(
            llm,
            [SystemMessage(content=TASK_REPLANNER_SYSTEM), HumanMessage(content=user_content)]
        )
        proposal = _parse_proposal(response)
    else:
        proposal = _parse_proposal(proposal_input)

    region = {str(identifier) for identifier in (scope.get("region") or [task_id])}
    # A single-node proposal is the normal case.  For a branch proposal the
    # model may return ``task_nodes``; all existing nodes are still filtered to
    # the declared region and passed through the identity/contract freeze.
    proposed_nodes = proposal.get("task_nodes")
    updated_nodes = dict(node_index)
    changed_ids: list[str] = []
    if isinstance(proposed_nodes, list):
        for candidate in proposed_nodes:
            if not isinstance(candidate, Mapping):
                continue
            identifier = str(candidate.get("id") or "")
            if identifier not in region or identifier not in node_index:
                continue
            updated_nodes[identifier] = _frozen_node(node_index[identifier], candidate)
            changed_ids.append(identifier)
    else:
        candidate = proposal.get("updated_node", proposal)
        if isinstance(candidate, Mapping):
            updated_nodes[task_id] = _frozen_node(task_node, candidate)
            changed_ids.append(task_id)

    # Do not let a missing/invalid response erase the existing specification.
    if not changed_ids:
        updated_nodes[task_id] = dict(task_node)
        changed_ids = [task_id]

    updated_task_nodes = [
        updated_nodes.get(str(node.get("id")), dict(node))
        for node in task_plan_ir.get("task_nodes", [])
        if isinstance(node, Mapping)
    ]
    updated_ir = {**dict(task_plan_ir), "task_nodes": updated_task_nodes}

    retried_tasks = list(state.get("retried_tasks") or [])
    for identifier in changed_ids:
        if identifier not in retried_tasks:
            retried_tasks.append(identifier)
    code_owners = code_owner_ids(updated_ir, state.get("code_snippets") or {})
    return {
        "updated_task_node": updated_nodes.get(task_id),
        "updated_task_nodes": {
            identifier: updated_nodes[identifier]
            for identifier in changed_ids
            if identifier in updated_nodes
        },
        "task_plan_ir": updated_ir,
        "retried_tasks": retried_tasks,
        "repair_plan": {
            "status": "proposed",
            "region": sorted(region),
            "frozen_region": sorted(set(scope.get("frozen_region") or [])),
            "changed_nodes": sorted(changed_ids),
            "changed_code_owners": sorted(set(changed_ids) & code_owners),
            "integration_exceptions": scope.get("integration_exceptions", []),
            "proposal": proposal,
        },
    }


__all__ = ["task_replanner"]
