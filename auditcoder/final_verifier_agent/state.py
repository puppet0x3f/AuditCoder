"""LangGraph state definition for the Final Verifier Agent."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class FinalVerifierState(TypedDict, total=False):
    # --- inputs (set by caller) ---
    problem_id: int
    final_code: str
    task_plan_ir: dict[str, Any]
    code_snippets: dict[str, str]
    verification_reports: dict[str, dict[str, Any]]

    # --- intermediate (set by nodes) ---
    execution_result: dict[str, Any]
    initial_execution_result: dict[str, Any]
    post_repair_execution_result: dict[str, Any]

    # --- retry tracking ---
    retried_tasks: list[str]
    updated_task_node: dict[str, Any] | None

    # --- output ---
    error_location: dict[str, Any] | None
    localization_record: dict[str, Any]
    final_report: dict[str, Any]

    # --- LangGraph messages ---
    messages: Annotated[list, add_messages]
