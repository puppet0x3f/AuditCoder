"""LangGraph state definition for the Plan Agent."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class ReviewResult(TypedDict, total=False):
    status: str       # "pass" | "fail"
    retry_node: str   # 回退目标节点名（仅 status=="fail" 时有值）
    reason: str       # 审查意见


class PlanState(TypedDict, total=False):
    problem_raw: str
    problem_structured: dict[str, Any]
    deep_thinking: dict[str, Any]
    constraint_analysis: dict[str, Any]
    algorithm_selection: list[dict[str, Any]]
    solution_strategy: dict[str, Any]
    task_tree: list[dict[str, Any]]
    interface_definitions: list[dict[str, Any]]
    test_cases: dict[str, Any]
    execution_contract: dict[str, Any]
    task_plan_ir: dict[str, Any]
    contract_graph: dict[str, Any]
    catg_validation_errors: list[str]
    review_result: ReviewResult
    revision_count: int
    messages: Annotated[list, add_messages]
    # Failure context injected by the retry orchestrator into deep_thinker on replan
    failure_context: str
