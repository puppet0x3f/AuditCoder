"""Node: ir_assembler — 将各节点输出组装为 TaskPlanIR。"""

from __future__ import annotations

import re

from auditcoder.common.execution_contract import parse_execution_contract
from auditcoder.plan_agent.state import PlanState
from auditcoder.schemas.contract_graph import ensure_catg, normalize_task_plan_ir


def ir_assembler(state: PlanState) -> dict:
    """Pure data assembly — no LLM call needed.

    Merges outputs from all upstream nodes into the TaskPlanIR schema.
    """
    problem = state["problem_structured"]
    constraint = state["constraint_analysis"]
    algo_sel = state["algorithm_selection"]
    strategy = state["solution_strategy"]
    task_tree = state["task_tree"]
    interfaces = state.get("interface_definitions", [])
    tests = state["test_cases"]

    algo_selection_list = algo_sel
    if isinstance(algo_sel, dict):
        algo_selection_list = algo_sel.get("algorithm_selection", [])

    # Merge interface definitions into leaf task nodes
    interface_by_id = {}
    if isinstance(interfaces, list):
        for iface in interfaces:
            if isinstance(iface, dict) and "task_id" in iface:
                interface_by_id[iface["task_id"]] = iface

    merged_nodes = []
    for node in task_tree:
        node = dict(node)
        if "id" not in node:
            continue  # drop malformed nodes from LLM output
        task_id = node.get("id", "")
        if task_id in interface_by_id:
            node["interface"] = interface_by_id[task_id].get("interface", interface_by_id[task_id])
        merged_nodes.append(node)

    structured_description = problem.get("description", "")
    original_specification = state.get("problem_raw", "").strip()
    authoritative_description = original_specification or structured_description
    starter_code = str(problem.get("starter_code", "") or "")
    if not starter_code and original_specification:
        match = re.search(
            r"【起始代码模板】.*?```python\s*(.*?)```",
            original_specification,
            flags=re.DOTALL,
        )
        if match:
            starter_code = match.group(1).strip()
    callable_name = str(problem.get("callable_name", "") or "")
    if not callable_name:
        callable_match = re.search(
            r"Solution\.([A-Za-z_]\w*)",
            starter_code,
        )
        if callable_match:
            callable_name = callable_match.group(1)

    ir = {
        "execution_contract": parse_execution_contract(
            state.get("problem_raw", "")
        ),
        "problem": {
            "title": problem.get("title", ""),
            "description": authoritative_description,
            "summary_description": structured_description,
            "original_specification": original_specification,
            "input_format": problem.get("input_format", ""),
            "output_format": problem.get("output_format", ""),
            "starter_code": starter_code,
            "callable_name": callable_name,
            "constraints": problem.get("constraints", {}),
            "sample_cases": problem.get("sample_cases", []),
        },
        "analysis": {
            "problem_category": (
                algo_sel.get("problem_category", "")
                if isinstance(algo_sel, dict)
                else ""
            ),
            "key_observations": (
                algo_sel.get("key_observations", [])
                if isinstance(algo_sel, dict)
                else []
            ),
            "complexity_target": {
                "time": constraint.get("time_complexity_upper_bound", ""),
                "space": constraint.get("space_complexity_upper_bound", ""),
            },
            "algorithm_selection": (
                algo_selection_list
                if isinstance(algo_selection_list, list)
                else []
            ),
            "reasoning_chain": (
                algo_sel.get("reasoning_chain", [])
                if isinstance(algo_sel, dict)
                else []
            ),
        },
        "global_strategy": (
            strategy.get("global_strategy", "")
            if isinstance(strategy, dict) else ""
        ),
        "complexity_budget": {
            "time": constraint.get("time_complexity_upper_bound", ""),
            "space": constraint.get("space_complexity_upper_bound", ""),
        },
        "task_nodes": merged_nodes,
        "root_id": (
            strategy.get("root_id", "S0")
            if isinstance(strategy, dict) else "S0"
        ),
        "execution_order": (
            strategy.get("execution_order", [])
            if isinstance(strategy, dict) else []
        ),
        "validation": {
            "sample_test_cases": (
                tests.get("sample_test_cases", [])
                if isinstance(tests, dict)
                else (tests if isinstance(tests, list) else [])
            ),
            "edge_cases": (
                tests.get("edge_cases", [])
                if isinstance(tests, dict)
                else []
            ),
            "invariants": (
                tests.get("invariants", [])
                if isinstance(tests, dict)
                else []
            ),
        },
    }

    # Allocate responsibility IDs, contracts, provenance, executor routing,
    # and the fixed development-check sequence before any algorithm agent is
    # invoked.  ``normalize_task_plan_ir`` only repairs mechanical omissions;
    # semantic planning choices remain visible for review and repair.
    ir = normalize_task_plan_ir(ir)
    ir = ensure_catg(ir)
    return {
        "task_plan_ir": ir,
        "contract_graph": ir.get("contract_annotated_task_graph"),
        "catg_validation_errors": ir.get("catg_validation_errors", []),
    }
