"""LangGraph state for the class verifier agent."""

from __future__ import annotations

from typing import Any, TypedDict


class ClassVerifierState(TypedDict, total=False):
    # inputs
    method_name: str
    partial_class_code: str
    test_code: str
    import_statements: list[str]
    class_name: str
    # intermediate
    compile_report: dict[str, Any]
    unittest_report: dict[str, Any]
    # output
    verification_report: dict[str, Any]
