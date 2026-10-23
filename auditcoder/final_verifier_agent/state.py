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
    # Dataset-backed stdin/stdout cases are authoritative when provided by a
    # benchmark runner.  Planner-derived boundary cases remain useful
    # diagnostics but are advisory because their expected outputs are model
    # generated.
    external_test_cases: list[dict[str, Any]]

    # CATG is produced before code generation and carried through assembly;
    # these aliases make the serialized construction trace available to the
    # deterministic locator without reconstructing it from model text.
    contract_graph: dict[str, Any]
    contract_annotated_task_graph: dict[str, Any]
    responsibility_records: list[dict[str, Any]]
    development_checks: list[dict[str, Any]]

    # --- intermediate (set by nodes) ---
    execution_result: dict[str, Any]
    initial_execution_result: dict[str, Any]
    post_repair_execution_result: dict[str, Any]

    # --- responsibility-preserving audit state ---
    # The graph remains JSON/dict based because TaskPlanIR is serialized, but
    # these fields make the paper's A=(G,S,E,H) trace explicit at runtime.
    ownership_map: dict[str, list[str]]
    registered_units: dict[str, Any]
    evidence_records: list[dict[str, Any]]
    audit_trace: dict[str, Any]
    audit_versions: dict[str, int]
    repair_snapshot: dict[str, Any]
    repair_scope: dict[str, Any] | None
    repair_attempts: int
    max_repair_attempts: int
    repair_history: list[dict[str, Any]]
    repair_event: dict[str, Any]
    repair_decision: str
    rollback_status: str
    candidate_execution_result: dict[str, Any]
    candidate_final_code: str
    candidate_code_snippets: dict[str, str]
    repair_proposal: dict[str, Any]

    # --- retry tracking ---
    retried_tasks: list[str]
    updated_task_node: dict[str, Any] | None

    # --- output ---
    error_location: dict[str, Any] | None
    localization_record: dict[str, Any]
    final_report: dict[str, Any]

    # --- LangGraph messages ---
    messages: Annotated[list, add_messages]
