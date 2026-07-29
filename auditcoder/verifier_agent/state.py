"""LangGraph state definition for the Verifier Agent."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class VerifierState(TypedDict, total=False):
    # --- inputs (set by caller) ---
    code_snippet: dict[str, Any]
    task_contract: dict[str, Any]
    validation_info: dict[str, Any]
    complexity_target: dict[str, Any]
    problem_info: dict[str, Any]

    # --- intermediate reports (set by nodes) ---
    contract_report: dict[str, Any]
    compile_report: dict[str, Any]
    sample_test_report: dict[str, Any]
    bruteforce_report: dict[str, Any]
    complexity_report: dict[str, Any]
    counterexample: dict[str, Any] | None
    weak_signal_report: dict[str, Any]

    # --- final output ---
    verification_report: dict[str, Any]

    # --- LangGraph messages ---
    messages: Annotated[list, add_messages]
