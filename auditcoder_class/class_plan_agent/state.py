"""LangGraph state definition for the class plan agent."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class ClassPlanState(TypedDict, total=False):
    raw_entry: dict[str, Any]
    class_context: dict[str, Any]
    method_thoughts: list[dict[str, Any]]
    method_algo_selection: list[dict[str, Any]]
    execution_order: list[str]
    class_plan_ir: dict[str, Any]
    messages: Annotated[list, add_messages]
