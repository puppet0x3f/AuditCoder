"""LangGraph state shared by all method generation agents."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class MethodAgentState(TypedDict, total=False):
    method_node: dict[str, Any]
    class_plan_ir: dict[str, Any]
    generated_methods: dict[str, str]
    method_body: str
    feedback: dict[str, Any] | None
    messages: Annotated[list, add_messages]
