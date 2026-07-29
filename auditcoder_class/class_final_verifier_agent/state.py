"""LangGraph state for the class final verifier agent."""

from __future__ import annotations

from typing import Any, TypedDict


class ClassFinalVerifierState(TypedDict, total=False):
    final_class_code: str
    class_plan_ir: dict[str, Any]
    method_bodies: dict[str, str]
    per_method_reports: dict[str, dict[str, Any]]
    structural_violations: list[dict[str, Any]]
    # intermediate
    execution_result: dict[str, Any]
    error_location: dict[str, Any] | None
    retried_methods: list[str]
    # output
    final_report: dict[str, Any]
