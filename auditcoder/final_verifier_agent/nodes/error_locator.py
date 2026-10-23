"""Evidence-to-responsibility localization.

The locator is deliberately deterministic.  It consumes the recorded global
run, the graph, the ownership registry, and local evidence; it never asks the
LLM to guess an owner.  Rule priority and abstention semantics follow
Algorithms ``Locate``/``Compat`` in the paper:

``Trace -> Resource -> Local -> abstain``.

The small compatibility wrappers at the bottom retain the API used by older
experiments and tests.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from auditcoder.final_verifier_agent.audit import (
    _build_interface_map,
    _locate_by_traceback,
    _locate_by_verification_reports,
    build_owner_map,
    locate,
    local_records_from_state,
    make_snapshot,
    normalize_location,
)
from auditcoder.final_verifier_agent.state import FinalVerifierState


def _locate_by_timing(*args: Any, **kwargs: Any) -> None:
    """Deprecated compatibility hook.

    Isolated import timing is not executable owner evidence: it never invokes
    the unit under the failing input.  The revised Resource rule therefore
    requires an owner-attributed measurement and this hook intentionally
    abstains.
    """

    return None


def _legacy_location_detail(
    location: Mapping[str, Any],
    execution_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach the old report fields without changing the audit result."""

    result = dict(location)
    result.setdefault("error_type", execution_result.get("error_type", "unknown"))
    anchor = result.get("anchor") or {}
    result.setdefault(
        "detail",
        str(
            anchor.get("stderr")
            or execution_result.get("stderr")
            or anchor.get("actual", "")
        )[:500],
    )
    if anchor:
        result.setdefault(
            "failing_case",
            {
                "case_index": anchor.get("case_index", anchor.get("index")),
                "input": anchor.get("input", anchor.get("stdin", "")),
                "expected": anchor.get("expected", ""),
                "actual": anchor.get("actual", ""),
                "stderr": anchor.get("stderr", ""),
                "timed_out": anchor.get("timed_out", False),
                "exit_code": anchor.get("exit_code"),
                "time_ms": anchor.get("time_ms"),
            },
        )
    if result.get("task_id") and "agent_type" not in result:
        result["agent_type"] = ""
    # The old metrics read ``method``/``confidence`` while the paper uses
    # ``rule``/capitalized grades.  Keep both names in every record.
    result.setdefault("method", str(result.get("rule", "Local")).lower())
    result.setdefault("confidence", "low")
    result.setdefault("rule", result.get("method", "Local"))
    result.setdefault("implicated_nodes", result.get("owners", []))
    return result


def error_locator(state: FinalVerifierState) -> dict[str, Any]:
    execution_result = state.get("execution_result", {}) or {}
    task_plan_ir = state.get("task_plan_ir", {}) or {}
    code_snippets = state.get("code_snippets", {}) or {}
    final_code = str(state.get("final_code", "") or "")

    # Snapshot before any replanner/generator can mutate the candidate.  A
    # rejected transaction is restored from this exact state rather than from
    # a reconstructed graph.
    snapshot = make_snapshot(state)

    if execution_result.get("all_passed", False) or (
        execution_result.get("error_type") in {None, "none"}
        and not execution_result.get("results")
    ):
        return {
            "error_location": None,
            "localization_record": {
                "status": "not_needed",
                "method": None,
                "rule": None,
                "error_type": "none",
            },
            "repair_snapshot": snapshot,
            "repair_scope": None,
            "scope_record": None,
        }

    versions = state.get("audit_versions") or {}
    explicit_owner_map = state.get("ownership_map") or state.get("registered_units")
    owner_map = build_owner_map(
        task_plan_ir,
        code_snippets,
        final_code,
        explicit_owner_map if isinstance(explicit_owner_map, Mapping) else None,
    )
    records = local_records_from_state(state, task_plan_ir)
    location = locate(
        execution_result,
        task_plan_ir,
        records,
        owner_map,
        versions,
        final_code,
        code_snippets,
    )

    if location is None or location.get("status") == "abstained":
        abstention = dict(location or {})
        abstention.setdefault("status", "abstained")
        abstention.setdefault("reason", "no_unique_evidence_supported_owner")
        abstention.setdefault("error_type", execution_result.get("error_type", "unknown"))
        interface_map = _build_interface_map(task_plan_ir)
        rejected_ids = [
            str(task_id)
            for task_id, report in (state.get("verification_reports") or {}).items()
            if isinstance(report, Mapping)
            and str(report.get("verdict", "")).lower()
            in {"reject", "rejected", "fail", "failed"}
        ]
        abstention["rejected_task_ids"] = rejected_ids
        abstention["ambiguous_function_names"] = sorted(
            name for name, owners in interface_map.items() if len(owners) > 1
        )
        return {
            "error_location": None,
            "localization_record": abstention,
            "repair_snapshot": snapshot,
            "repair_scope": None,
            "scope_record": None,
        }

    location = _legacy_location_detail(normalize_location(location), execution_result)
    return {
        "error_location": location,
        "localization_record": {
            "status": "localized",
            **copy.deepcopy(location),
        },
        "repair_snapshot": snapshot,
        "repair_scope": None,
        "scope_record": None,
    }


__all__ = [
    "error_locator",
    "_build_interface_map",
    "_locate_by_traceback",
    "_locate_by_verification_reports",
    "_locate_by_timing",
]
