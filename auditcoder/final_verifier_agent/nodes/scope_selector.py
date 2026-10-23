"""Deterministic nominal-region and frozen-complement selection."""

from __future__ import annotations

from typing import Any

from auditcoder.final_verifier_agent.audit import normalize_location, select_scope
from auditcoder.final_verifier_agent.state import FinalVerifierState


def scope_selector(state: FinalVerifierState) -> dict[str, Any]:
    location = normalize_location(state.get("error_location"))
    ir = state.get("task_plan_ir", {}) or {}
    snippets = state.get("code_snippets", {}) or {}
    scope = select_scope(location, ir, snippets)
    if scope is None:
        return {
            "repair_scope": None,
            "scope_record": {
                "status": "abstained",
                "reason": "no_supported_local_boundary",
            },
        }
    return {
        "repair_scope": scope,
        "scope_record": {"status": "selected", **scope},
    }


__all__ = ["scope_selector"]
