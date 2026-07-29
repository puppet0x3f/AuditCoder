"""LangGraph state for the class assembler agent."""

from __future__ import annotations

from typing import Any, TypedDict


class ClassAssemblerState(TypedDict, total=False):
    class_plan_ir: dict[str, Any]
    method_bodies: dict[str, str]
    final_class_code: str
    structural_violations: list[dict[str, Any]]
