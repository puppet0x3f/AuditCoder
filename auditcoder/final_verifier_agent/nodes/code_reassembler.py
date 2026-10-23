"""Assemble a staged repair while enforcing the declared frozen complement."""

from __future__ import annotations

from typing import Any

from auditcoder.assembler_agent.agent import build_assembler_agent_graph
from auditcoder.final_verifier_agent.audit import frozen_complement_ok
from auditcoder.final_verifier_agent.state import FinalVerifierState
from auditcoder.schemas.contract_graph import ensure_catg


def code_reassembler(state: FinalVerifierState) -> dict[str, Any]:
    task_plan_ir = state.get("task_plan_ir", {}) or {}
    code_snippets = dict(state.get("code_snippets") or {})
    scope = state.get("repair_scope") or {}
    snapshot = state.get("repair_snapshot") or {}
    before_snippets = dict(snapshot.get("code_snippets") or code_snippets)
    region = set(scope.get("region") or [])

    assembler = build_assembler_agent_graph()
    result = assembler.invoke(
        {"task_plan_ir": task_plan_ir, "code_snippets": code_snippets}
    )
    candidate_ir = result.get("task_plan_ir") or ensure_catg(task_plan_ir)
    new_final_code: str = str(result.get("final_code", "") or "")

    # Direct callers of the original node supplied no transaction context and
    # expect its compact one-field result.  The compiled bounded-repair graph
    # always supplies a scope/snapshot and receives the richer candidate
    # record below.
    if not scope and not snapshot:
        return {"final_code": new_final_code, "task_plan_ir": candidate_ir}

    # The assembler is allowed to emit a candidate program, but the owned code
    # bundles outside R are immutable.  If an agent accidentally returned a
    # complete replacement mapping, retain the pre-repair frozen values and
    # mark the candidate for rejection instead of widening R after the fact.
    candidate_snippets = dict(result.get("code_snippets") or code_snippets)
    frozen_ok, frozen_violations = frozen_complement_ok(
        before_snippets, candidate_snippets, region
    )
    if not frozen_ok:
        candidate_snippets = dict(candidate_snippets)
        for identifier in frozen_violations:
            if identifier in before_snippets:
                candidate_snippets[identifier] = before_snippets[identifier]

    # Preserve the assembler's output in a candidate-only field.  The active
    # state is committed or restored by post_repair_code_runner after rank and
    # boundary checks.
    return {
        "task_plan_ir": candidate_ir,
        "candidate_final_code": new_final_code,
        "candidate_code_snippets": candidate_snippets,
        "final_code": new_final_code,
        "code_snippets": candidate_snippets,
        "assembly_result": result,
        "repair_boundary_check": {
            "frozen_complement_ok": frozen_ok,
            "frozen_violations": frozen_violations,
            "declared_region": sorted(region),
        },
    }


__all__ = ["code_reassembler"]
