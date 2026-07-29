"""Pydantic schemas for the class-based multi-algorithm agent pipeline (ClassEval)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MethodInterface(BaseModel):
    method_name: str
    params: list[dict] = Field(default_factory=list)
    return_type: str = "None"
    return_description: str = ""
    method_type: str = "instance"  # "instance", "staticmethod", or "classmethod"


class MethodNode(BaseModel):
    method_name: str
    method_description: str
    agent_type: str
    is_standalone: bool
    lib_dependencies: list[str] = Field(default_factory=list)
    field_dependencies: list[str] = Field(default_factory=list)
    method_dependencies: list[str] = Field(default_factory=list)
    interface: MethodInterface
    hints: list[str] = Field(default_factory=list)
    test_class_name: str = ""
    test_code: str = ""


class ClassPlanIR(BaseModel):
    task_id: str
    class_name: str
    class_description: str
    import_statements: list[str] = Field(default_factory=list)
    class_constructor: str = ""
    fields: list[str] = Field(default_factory=list)
    skeleton: str = ""
    method_nodes: list[MethodNode] = Field(default_factory=list)
    execution_order: list[str] = Field(default_factory=list)
