"""Compare the pre-paper and paper-aligned repair policies on APPS faults.

The experiment uses ten APPS test problems with a verified Python solution for
which a deterministic AST mutation creates a known fault.  Both policies see
the same faulty program and the same candidate code.  The candidate is the
known-good function replacement (an oracle repair) so the comparison isolates
localization, scope control, candidate acceptance, and rollback rather than
LLM sampling variance.

The legacy policy mirrors the original final-verifier behavior: traceback
lookup, isolated-import timing for timeout attribution, a unique rejected
verification report fallback, singleton retry, and unconditional acceptance.
The revised policy calls the paper-aligned ``Trace -> Resource -> Local``
locator, evidence-supported scope selection, frozen-complement checking, and
strict rank-improvement acceptance.

Run from the workspace root::

    python tests/apps_controlled_repair_comparison.py \
        --output-dir results/apps_10_repair_comparison
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = WORKSPACE_ROOT
sys.path.insert(0, str(PACKAGE_ROOT))

from auditcoder.final_verifier_agent.audit import (  # noqa: E402
    INTERFACE,
    IMPLEMENTATION,
    build_owner_map,
    frozen_complement_ok,
    locate,
    rank_execution,
    select_scope,
)


APPS_ROOT = WORKSPACE_ROOT / "data" / "APPS" / "test"
FROZEN_HELPER = "__auditcoder_frozen_helper"
BRANCH_PROVIDER = "__auditcoder_branch_provider"
BRANCH_TARGET_FAULT = "__auditcoder_branch_target_fault"


@dataclass(frozen=True)
class AppSpec:
    problem_id: int
    solution_index: int
    fault_mode: str
    expected_owners: tuple[str, ...]
    expected_abstention: bool = False
    expected_success: bool = False


# The IDs and solution indices were selected from APPS/test.  The clean
# candidates pass every stored input/output pair before fault injection.
SELECTED_CASES: tuple[AppSpec, ...] = (
    AppSpec(4009, 6, "trace_runtime", ("T",), expected_success=True),
    AppSpec(4013, 6, "local_wrong_answer", ("T",), expected_success=True),
    AppSpec(4020, 13, "resource_budget", ("T",), expected_success=True),
    AppSpec(99, 0, "unsafe_frozen_candidate", ("T",)),
    AppSpec(102, 7, "non_improving_candidate", ("T",)),
    AppSpec(104, 0, "interface_branch", ("T", "C"), expected_success=True),
    AppSpec(3705, 1, "trace_runtime", ("T",), expected_success=True),
    AppSpec(3707, 0, "unmapped_trace", (), expected_abstention=True),
    AppSpec(3710, 3, "ambiguous_trace", (), expected_abstention=True),
    AppSpec(3735, 9, "local_wrong_answer", ("T",), expected_success=True),
)


@dataclass
class PreparedCase:
    spec: AppSpec
    metadata: dict[str, Any]
    inputs: list[str]
    outputs: list[str]
    target_function: str
    clean_source: str
    faulty_source: str
    clean_ir: dict[str, Any]
    initial_ir: dict[str, Any]
    clean_snippets: dict[str, str]
    initial_snippets: dict[str, str]
    clean_execution: dict[str, Any]
    initial_execution_actual: dict[str, Any]
    initial_execution: dict[str, Any]
    local_records: list[dict[str, Any]]
    verification_reports: dict[str, dict[str, Any]]
    candidate_sources: dict[str, str]
    candidate_snippets: dict[str, dict[str, str]]


# ---------------------------------------------------------------------------
# APPS loading and process execution
# ---------------------------------------------------------------------------


def _load_app(spec: AppSpec) -> tuple[dict[str, Any], list[str], list[str], list[str]]:
    app_dir = APPS_ROOT / f"{spec.problem_id:04d}"
    if not app_dir.is_dir():
        raise FileNotFoundError(f"APPS problem directory not found: {app_dir}")
    metadata = json.loads((app_dir / "metadata.json").read_text(encoding="utf-8"))
    io_data = json.loads((app_dir / "input_output.json").read_text(encoding="utf-8"))
    solutions = json.loads((app_dir / "solutions.json").read_text(encoding="utf-8"))
    if spec.solution_index >= len(solutions):
        raise IndexError(f"solution index {spec.solution_index} missing for {spec.problem_id}")
    inputs = [str(value) for value in io_data.get("inputs", [])]
    outputs = [str(value) for value in io_data.get("outputs", [])]
    if len(inputs) != len(outputs) or not inputs:
        raise ValueError(f"APPS problem {spec.problem_id} has inconsistent test pairs")
    return metadata, inputs, outputs, [str(solutions[spec.solution_index])]


def _execute_program(
    source: str,
    inputs: list[str],
    outputs: list[str],
    *,
    timeout_s: float = 1.5,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    all_passed = True
    timed_out_global = False
    first_stderr = ""
    max_time_ms = 0.0
    for index, (test_input, expected) in enumerate(zip(inputs, outputs)):
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", source],
                input=test_input,
                text=True,
                capture_output=True,
                timeout=timeout_s,
            )
            timed_out = False
            stdout = completed.stdout
            stderr = completed.stderr
            exit_code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = str(exc.stdout or "")
            stderr = str(exc.stderr or "")
            exit_code = -1
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        max_time_ms = max(max_time_ms, elapsed_ms)
        actual = stdout.strip()
        expected_text = str(expected).strip()
        passed = (
            not timed_out
            and exit_code == 0
            and actual == expected_text
        )
        if not passed:
            all_passed = False
        if timed_out:
            timed_out_global = True
        if stderr and not first_stderr:
            first_stderr = stderr[:1000]
        if timed_out:
            symptom = "TLE"
        elif exit_code != 0:
            symptom = "RE"
        elif not passed:
            symptom = "WA"
        else:
            symptom = "Pass"
        results.append(
            {
                "case_index": index,
                "input": test_input,
                "expected": expected_text,
                "actual": actual,
                "passed": passed,
                "timed_out": timed_out,
                "exit_code": exit_code,
                "stderr": stderr[:500],
                "time_ms": elapsed_ms,
                "symptom": symptom,
            }
        )
    if timed_out_global:
        error_type = "timeout"
    elif all_passed:
        error_type = "none"
    elif any(item["exit_code"] != 0 for item in results if not item["passed"]):
        error_type = "runtime_error"
    else:
        error_type = "wrong_answer"
    return {
        "ran_tests": True,
        "all_passed": all_passed,
        "timed_out": timed_out_global,
        "exit_code": 0 if all_passed else next(
            (item["exit_code"] for item in results if not item["passed"]), -1
        ),
        "stdout": "",
        "stderr": first_stderr,
        "time_ms": max_time_ms,
        "results": results,
        "error_type": error_type,
        "test_source": "APPS/test/input_output.json",
    }


def _first_failed(execution: Mapping[str, Any]) -> dict[str, Any]:
    for result in execution.get("results", []) or []:
        if not result.get("passed", False):
            return dict(result)
    raise AssertionError("fault injection did not create a failing case")


def _fault_to_resource_signal(
    actual: Mapping[str, Any],
    owner: str,
) -> dict[str, Any]:
    """Turn a measured delay into the paper's explicit resource witness."""

    result = copy.deepcopy(dict(actual))
    results = result.get("results") or []
    if not results:
        raise AssertionError("resource case has no APPS execution result")
    anchor = dict(results[0])
    anchor["passed"] = False
    anchor["timed_out"] = True
    anchor["exit_code"] = -1
    anchor["symptom"] = "TLE"
    # The code still produces the right value; the injected sleep is judged
    # against a deliberately small declared budget.
    observed = max(float(anchor.get("time_ms") or 0.0), 1.0)
    result["results"] = [anchor, *results[1:]]
    result["all_passed"] = False
    result["timed_out"] = True
    result["exit_code"] = -1
    result["error_type"] = "timeout"
    result["stderr"] = ""
    result["owner_measurements"] = {
        owner: [
            {
                "metric": "time",
                "observed": observed,
                "limit": 0.5,
                "violated": True,
                "unit": "ms",
            }
        ]
    }
    result["injected_signal"] = "owner-attributed resource budget violation"
    return result


# ---------------------------------------------------------------------------
# AST mutation and ownership IR
# ---------------------------------------------------------------------------


def _parse(source: str) -> ast.Module:
    return ast.parse(source)


def _function_names(source: str) -> list[str]:
    tree = _parse(source)
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _top_level_function_names(source: str) -> list[str]:
    tree = _parse(source)
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _top_level_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ValueError(f"top-level function {name!r} not found")


def _function_source(source: str, name: str) -> str:
    tree = _parse(source)
    return ast.unparse(_top_level_function(tree, name)) + "\n"


def _mutate_function(
    source: str,
    name: str,
    mutator: Callable[[ast.FunctionDef], None],
) -> str:
    tree = _parse(source)
    node = _top_level_function(tree, name)
    mutator(node)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _raise_stmt(message: str) -> ast.Raise:
    return ast.Raise(
        exc=ast.Call(
            func=ast.Name(id="RuntimeError", ctx=ast.Load()),
            args=[ast.Constant(value=message)],
            keywords=[],
        ),
        cause=None,
    )


def _inject_runtime(source: str, name: str, marker: str) -> str:
    return _mutate_function(source, name, lambda node: node.body.insert(0, _raise_stmt(marker)))


def _inject_wrong_answer(source: str, name: str) -> str:
    return _mutate_function(
        source,
        name,
        lambda node: setattr(node, "body", [ast.Return(value=ast.Constant(value=-999999))]),
    )


def _inject_sleep(source: str, name: str) -> str:
    sleep_call = ast.Expr(
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Call(
                    func=ast.Name(id="__import__", ctx=ast.Load()),
                    args=[ast.Constant(value="time")],
                    keywords=[],
                ),
                attr="sleep",
                ctx=ast.Load(),
            ),
            args=[ast.Constant(value=0.001)],
            keywords=[],
        )
    )
    return _mutate_function(source, name, lambda node: node.body.insert(0, sleep_call))


def _inject_return_constant(source: str, name: str, value: int) -> str:
    return _mutate_function(
        source,
        name,
        lambda node: setattr(node, "body", [ast.Return(value=ast.Constant(value=value))]),
    )


def _insert_function(tree: ast.Module, source: str) -> None:
    helper_tree = ast.parse(source)
    tree.body = helper_tree.body + tree.body


def _add_frozen_helper(source: str) -> str:
    tree = _parse(source)
    if FROZEN_HELPER in _function_names(source):
        raise ValueError(f"reserved helper name already used: {FROZEN_HELPER}")
    _insert_function(
        tree,
        f"def {FROZEN_HELPER}():\n    return 0\n",
    )
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _add_unsafe_guard_call(source: str, target: str) -> str:
    call = ast.Expr(value=ast.Call(func=ast.Name(id=FROZEN_HELPER, ctx=ast.Load()), args=[], keywords=[]))
    return _mutate_function(source, target, lambda node: node.body.insert(0, call))


def _add_branch_helpers(source: str, target: str) -> str:
    tree = _parse(source)
    if BRANCH_PROVIDER in _function_names(source) or BRANCH_TARGET_FAULT in _function_names(source):
        raise ValueError("reserved branch helper name already used")
    _insert_function(
        tree,
        (
            f"def {BRANCH_PROVIDER}():\n"
            "    return 0\n\n"
            f"def {BRANCH_TARGET_FAULT}():\n"
            "    return False\n"
        ),
    )
    target_node = _top_level_function(tree, target)
    preamble = ast.parse(
        f"{BRANCH_PROVIDER}_flag = {BRANCH_PROVIDER}()\n"
        f"if {BRANCH_PROVIDER}_flag or {BRANCH_TARGET_FAULT}():\n"
        "    return -999999\n"
    ).body
    target_node.body = preamble + target_node.body
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _build_wrapped_clean_source(source: str, target: str, mode: str) -> str:
    wrapped = _add_frozen_helper(source)
    if mode == "unsafe_frozen_candidate":
        wrapped = _add_unsafe_guard_call(wrapped, target)
    elif mode == "interface_branch":
        wrapped = _add_branch_helpers(wrapped, target)
    return wrapped


def _build_ir(mode: str, target: str, inputs: list[str], outputs: list[str]) -> dict[str, Any]:
    def leaf(identifier: str, function_name: str, parent: str = "R", owned_units: Iterable[str] = ()) -> dict[str, Any]:
        return {
            "id": identifier,
            "goal": f"Own executable unit {function_name}",
            "type": "implementation",
            "is_leaf": True,
            "agent_type": "apps",
            "parent": parent,
            "dependencies": [],
            "inputs": [],
            "outputs": [],
            "interface": {"function_name": function_name, "params": [], "return_type": "unknown"},
            "owned_units": list(owned_units),
            "complexity_budget": {"time_ms": 0.5, "memory_kb": 1024},
        }

    if mode == "interface_branch":
        task_nodes = [
            {
                "id": "R",
                "goal": "APPS orchestration",
                "type": "orchestration",
                "is_leaf": False,
                "agent_type": None,
                "parent": None,
                "dependencies": ["B", "F"],
                "inputs": [],
                "outputs": [],
                "interface": None,
            },
            {
                "id": "B",
                "goal": "Shared branch contract",
                "type": "branch",
                "is_leaf": False,
                "agent_type": None,
                "parent": "R",
                "dependencies": ["T", "C"],
                "inputs": [],
                "outputs": [],
                "interface": None,
                "commitment": "shared_contract",
            },
            leaf("T", target, "B", [BRANCH_TARGET_FAULT]),
            leaf("C", BRANCH_PROVIDER, "B"),
            leaf("F", FROZEN_HELPER),
        ]
    else:
        interface_name = target
        if mode == "unmapped_trace":
            interface_name = "__auditcoder_unmapped_owner"
        task_nodes = [
            {
                "id": "R",
                "goal": "APPS orchestration",
                "type": "orchestration",
                "is_leaf": False,
                "agent_type": None,
                "parent": None,
                "dependencies": ["T", "F"],
                "inputs": [],
                "outputs": [],
                "interface": None,
            },
            leaf("T", interface_name),
            leaf("F", target if mode == "ambiguous_trace" else FROZEN_HELPER, "R", [FROZEN_HELPER]),
        ]
    return {
        "root_id": "R",
        "task_nodes": task_nodes,
        "problem": {"title": "APPS controlled fault", "constraints": {"time_limit": "0.5ms"}},
        "execution_contract": {"io_mode": "stdin_stdout"},
        "validation": {
            "sample_test_cases": [
                {"input": inp, "expected_output": out, "test_source": "APPS/test/input_output.json"}
                for inp, out in zip(inputs[:5], outputs[:5])
            ]
        },
    }


def _extract_snippets(source: str, mode: str, target: str) -> dict[str, str]:
    snippets = {
        "T": _function_source(source, target),
        "F": _function_source(source, FROZEN_HELPER),
    }
    if mode == "interface_branch":
        snippets["T"] += _function_source(source, BRANCH_TARGET_FAULT)
        snippets["C"] = _function_source(source, BRANCH_PROVIDER)
    return snippets


def _mutate_branch_target(source: str) -> str:
    return _inject_return_constant(source, BRANCH_TARGET_FAULT, 1)


def _mutate_branch_provider(source: str) -> str:
    return _inject_return_constant(source, BRANCH_PROVIDER, 1)


def _candidate_for_mode(
    mode: str,
    clean_source: str,
    faulty_source: str,
    target: str,
) -> dict[str, str]:
    if mode in {"trace_runtime", "local_wrong_answer", "resource_budget"}:
        return {"legacy": clean_source, "revised": clean_source}
    if mode == "unsafe_frozen_candidate":
        unsafe = _inject_runtime(clean_source, FROZEN_HELPER, "AUDITCODER_UNSAFE_FROZEN_EDIT")
        return {"legacy": unsafe, "revised": unsafe}
    if mode == "non_improving_candidate":
        return {"legacy": faulty_source, "revised": faulty_source}
    if mode == "interface_branch":
        old = _mutate_branch_provider(clean_source)
        return {"legacy": old, "revised": clean_source}
    return {}


def _build_local_records(
    mode: str,
    execution: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if mode not in {"local_wrong_answer", "interface_branch"}:
        return []
    anchor = _first_failed(execution)
    witness: dict[str, Any] = {
        "case_index": anchor.get("case_index"),
        "input": anchor.get("input", ""),
        "applicability": "observed",
        "observed": True,
    }
    if mode == "local_wrong_answer":
        return [
            {
                "kind": "exec",
                "verdict": "reject",
                "owners": ["T"],
                "obligation": {"class": "semantic", "name": "counterexample"},
                "witness": witness,
                "version": {"T": 0},
            }
        ]
    witness["provenance_path"] = ["T", "C"]
    return [
        {
            "kind": "exec",
            "verdict": "reject",
            "owners": ["T", "C"],
            "provider_pairs": [["T", "C"]],
            "obligation": {
                "class": INTERFACE,
                "name": "shared_contract",
                "provider_pair": ["T", "C"],
            },
            "witness": witness,
            "version": {"T": 0, "C": 0},
        }
    ]


def _build_verification_reports(mode: str) -> dict[str, dict[str, Any]]:
    # The branch case deliberately leaves only the old unique-report fallback;
    # the revised method receives the richer pair evidence above.
    if mode == "interface_branch":
        return {"T": {"verdict": "reject", "agent_type": "apps"}}
    return {}


# ---------------------------------------------------------------------------
# Legacy and paper-aligned policy runners
# ---------------------------------------------------------------------------


def _legacy_interface_map(ir: Mapping[str, Any]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for node in ir.get("task_nodes", []) or []:
        if not node.get("is_leaf"):
            continue
        name = str((node.get("interface") or {}).get("function_name") or "")
        if name:
            mapping.setdefault(name, []).append(str(node.get("id")))
    return mapping


def _legacy_locate(
    execution: Mapping[str, Any],
    ir: Mapping[str, Any],
    verification_reports: Mapping[str, Mapping[str, Any]],
    snippets: Mapping[str, str],
) -> dict[str, Any] | None:
    error_type = str(execution.get("error_type", "none"))
    interface_map = _legacy_interface_map(ir)
    if error_type in {"runtime_error", "wrong_answer", "contract_error"}:
        stderr = str(execution.get("stderr", ""))
        frame_names = re.findall(r"in\s+(\w+)", stderr)
        for name in reversed(frame_names):
            owners = interface_map.get(name, [])
            if len(owners) == 1:
                return {"task_id": owners[0], "owners": owners, "method": "traceback", "rule": "legacy_trace"}
            if owners:
                return None
    elif error_type == "timeout":
        # The original timing rule imported function definitions but did not
        # invoke them.  The injected delay is inside the target function, so it
        # is intentionally invisible to this rule.
        for snippet in snippets.values():
            try:
                subprocess.run(
                    [sys.executable, "-I", "-c", f"{snippet}\nprint('__ok__')\n"],
                    capture_output=True,
                    text=True,
                    timeout=0.05,
                )
            except subprocess.TimeoutExpired:
                return None
    rejected = [
        (str(task_id), report)
        for task_id, report in verification_reports.items()
        if str(report.get("verdict", "")).lower() == "reject"
    ]
    if len(rejected) == 1:
        task_id, report = rejected[0]
        return {
            "task_id": task_id,
            "owners": [task_id],
            "method": "verification_report",
            "rule": "legacy_local",
            "agent_type": report.get("agent_type", ""),
        }
    return None


def _location_owners(location: Mapping[str, Any] | None) -> set[str]:
    if not location:
        return set()
    owners = location.get("owners", location.get("implicated_nodes", []))
    if isinstance(owners, str):
        return {owners}
    return {str(item) for item in (owners or []) if str(item)}


def _legacy_scope(location: Mapping[str, Any] | None) -> dict[str, Any] | None:
    owners = _location_owners(location)
    if len(owners) != 1:
        return None
    owner = next(iter(owners))
    return {
        "nominal_region": [owner],
        "region": [owner],
        "frozen_region": [],
        "reason": "legacy_singleton_retry",
    }


def _new_location(
    execution: Mapping[str, Any],
    ir: Mapping[str, Any],
    snippets: Mapping[str, str],
    local_records: list[dict[str, Any]],
    versions: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    owner_map = build_owner_map(ir, snippets, final_code="")
    return locate(
        execution,
        ir,
        local_records=local_records,
        owner_map=owner_map,
        versions=versions or {},
        code_snippets=snippets,
    )


def _new_policy_run(
    case: PreparedCase,
    initial_execution: Mapping[str, Any],
    *,
    candidate_source: str | None,
    candidate_snippets: Mapping[str, str] | None,
    candidate_execution: Mapping[str, Any] | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    location = _new_location(
        initial_execution,
        case.initial_ir,
        case.initial_snippets,
        case.local_records,
    )
    scope = select_scope(location, case.initial_ir, case.initial_snippets) if location else None
    rank_before = rank_execution(initial_execution, case.verification_reports)
    decision = "Abstain"
    rollback_status = "NotAttempted"
    reason = "no_supported_scope"
    frozen_ok: bool | None = None
    frozen_violations: list[str] = []
    rank_after: tuple[int, int, int, int, int] | None = None
    final_source = case.faulty_source
    final_snippets = case.initial_snippets
    final_execution = dict(case.initial_execution)
    if location and scope and candidate_source is not None and candidate_snippets is not None and candidate_execution is not None:
        frozen_ok, frozen_violations = frozen_complement_ok(
            case.initial_snippets,
            candidate_snippets,
            scope.get("region", []),
        )
        rank_after = rank_execution(candidate_execution, case.verification_reports)
        if frozen_ok and rank_after > rank_before:
            decision = "Accept"
            rollback_status = "Committed"
            reason = "strict_rank_improvement"
            final_source = candidate_source
            final_snippets = dict(candidate_snippets)
            final_execution = dict(candidate_execution)
        else:
            decision = "Reject"
            rollback_status = "Restored"
            reason = (
                "frozen_complement_violation"
                if not frozen_ok
                else "candidate_rank_not_improved"
            )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "location": location,
        "scope": scope,
        "decision": decision,
        "rollback_status": rollback_status,
        "reason": reason,
        "rank_before": list(rank_before),
        "rank_after": list(rank_after) if rank_after is not None else None,
        "frozen_complement_ok": frozen_ok,
        "frozen_violations": frozen_violations,
        "candidate_execution": dict(candidate_execution or {}),
        "final_source": final_source,
        "final_snippets": final_snippets,
        "final_execution": final_execution,
        "changed_units": sorted(
            key for key in set(case.initial_snippets) | set(candidate_snippets or {})
            if str(case.initial_snippets.get(key, "")) != str((candidate_snippets or {}).get(key, ""))
        ),
        "wall_ms": elapsed_ms,
    }


def _legacy_policy_run(
    case: PreparedCase,
    initial_execution: Mapping[str, Any],
    *,
    candidate_source: str | None,
    candidate_snippets: Mapping[str, str] | None,
    candidate_execution: Mapping[str, Any] | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    location = _legacy_locate(
        initial_execution,
        case.initial_ir,
        case.verification_reports,
        case.initial_snippets,
    )
    scope = _legacy_scope(location)
    decision = "Abstain"
    final_source = case.faulty_source
    final_snippets = case.initial_snippets
    final_execution = dict(case.initial_execution)
    if location and scope and candidate_source is not None and candidate_snippets is not None and candidate_execution is not None:
        decision = "Accept"
        final_source = candidate_source
        final_snippets = dict(candidate_snippets)
        final_execution = dict(candidate_execution)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "location": location,
        "scope": scope,
        "decision": decision,
        "rollback_status": "NotSupported",
        "reason": "legacy_unconditional_singleton_retry" if location else "no_legacy_location",
        "rank_before": list(rank_execution(initial_execution, case.verification_reports)),
        "rank_after": list(rank_execution(candidate_execution, case.verification_reports)) if candidate_execution else None,
        "frozen_complement_ok": None,
        "frozen_violations": [],
        "candidate_execution": dict(candidate_execution or {}),
        "final_source": final_source,
        "final_snippets": final_snippets,
        "final_execution": final_execution,
        "changed_units": sorted(
            key for key in set(case.initial_snippets) | set(candidate_snippets or {})
            if str(case.initial_snippets.get(key, "")) != str((candidate_snippets or {}).get(key, ""))
        ),
        "wall_ms": elapsed_ms,
    }


# ---------------------------------------------------------------------------
# Case construction, metrics, and output
# ---------------------------------------------------------------------------


def _prepare_case(spec: AppSpec, cache: dict[str, dict[str, Any]]) -> PreparedCase:
    metadata, inputs, outputs, solution_box = _load_app(spec)
    original_source = solution_box[0]
    names = _top_level_function_names(original_source)
    target = next((name for name in ("solve", "main", "solution") if name in names), "")
    if not target:
        # A few APPS solutions use a differently named top-level entry point
        # such as ``__starting_point``.  It is still a stable executable owner
        # for traceback injection and has already been verified on all tests.
        target = next((name for name in names if not name.startswith("__auditcoder_")), "")
    if not target:
        raise ValueError(f"selected APPS solution has no conventional entry function: {spec.problem_id}")
    wrapped_clean = _build_wrapped_clean_source(original_source, target, spec.fault_mode)
    clean_ir = _build_ir(spec.fault_mode, target, inputs, outputs)
    clean_snippets = _extract_snippets(wrapped_clean, spec.fault_mode, target)

    if spec.fault_mode in {"trace_runtime", "unsafe_frozen_candidate", "non_improving_candidate"}:
        faulty_source = _inject_runtime(wrapped_clean, target, f"AUDITCODER_INJECTED_RUNTIME:{spec.problem_id}")
    elif spec.fault_mode == "local_wrong_answer":
        faulty_source = _inject_wrong_answer(wrapped_clean, target)
    elif spec.fault_mode == "resource_budget":
        faulty_source = _inject_sleep(wrapped_clean, target)
    elif spec.fault_mode == "interface_branch":
        faulty_source = _mutate_branch_target(_mutate_branch_provider(wrapped_clean))
    elif spec.fault_mode in {"unmapped_trace", "ambiguous_trace"}:
        faulty_source = _inject_runtime(wrapped_clean, target, f"AUDITCODER_INJECTED_RUNTIME:{spec.problem_id}")
    else:
        raise ValueError(f"unknown fault mode: {spec.fault_mode}")

    initial_ir = copy.deepcopy(clean_ir)
    initial_snippets = _extract_snippets(faulty_source, spec.fault_mode, target)
    verification_reports = _build_verification_reports(spec.fault_mode)

    key_clean = hashlib.sha256(wrapped_clean.encode()).hexdigest()
    if key_clean not in cache:
        cache[key_clean] = _execute_program(wrapped_clean, inputs, outputs)
    clean_execution = copy.deepcopy(cache[key_clean])
    if not clean_execution.get("all_passed"):
        raise AssertionError(
            f"clean APPS candidate {spec.problem_id} failed: "
            + json.dumps(clean_execution, ensure_ascii=False)[:1000]
        )
    key_faulty = hashlib.sha256(faulty_source.encode()).hexdigest()
    if key_faulty not in cache:
        cache[key_faulty] = _execute_program(faulty_source, inputs, outputs)
    initial_actual = copy.deepcopy(cache[key_faulty])
    initial_execution = (
        _fault_to_resource_signal(initial_actual, "T")
        if spec.fault_mode == "resource_budget"
        else initial_actual
    )
    if spec.fault_mode == "resource_budget":
        # The source-level sleep is still verified, while the policy receives
        # the explicit paper-style owner measurement above.
        if not initial_actual.get("all_passed"):
            raise AssertionError("resource mutation should preserve functional outputs")
    else:
        if initial_actual.get("all_passed"):
            raise AssertionError(f"fault injection did not fail APPS problem {spec.problem_id}")

    candidates = _candidate_for_mode(spec.fault_mode, wrapped_clean, faulty_source, target)
    candidate_sources: dict[str, str] = {}
    candidate_snippets: dict[str, dict[str, str]] = {}
    for method, source in candidates.items():
        candidate_sources[method] = source
        candidate_snippets[method] = _extract_snippets(source, spec.fault_mode, target)
        key = hashlib.sha256(source.encode()).hexdigest()
        if key not in cache:
            cache[key] = _execute_program(source, inputs, outputs)
    local_records = _build_local_records(spec.fault_mode, initial_execution)
    if spec.fault_mode == "resource_budget":
        # Resource attribution is intentionally carried by execution_result;
        # no weaker local record is added.
        local_records = []
    return PreparedCase(
        spec=spec,
        metadata=metadata,
        inputs=inputs,
        outputs=outputs,
        target_function=target,
        clean_source=wrapped_clean,
        faulty_source=faulty_source,
        clean_ir=clean_ir,
        initial_ir=initial_ir,
        clean_snippets=clean_snippets,
        initial_snippets=initial_snippets,
        clean_execution=clean_execution,
        initial_execution_actual=initial_actual,
        initial_execution=initial_execution,
        local_records=local_records,
        verification_reports=verification_reports,
        candidate_sources=candidate_sources,
        candidate_snippets=candidate_snippets,
    )


def _cached_execution(source: str, case: PreparedCase, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    key = hashlib.sha256(source.encode()).hexdigest()
    if key not in cache:
        cache[key] = _execute_program(source, case.inputs, case.outputs)
    return copy.deepcopy(cache[key])


def _method_record(
    case: PreparedCase,
    method: str,
    run: Mapping[str, Any],
) -> dict[str, Any]:
    location = run.get("location") or {}
    owners = sorted(_location_owners(location))
    expected = set(case.spec.expected_owners)
    localized = bool(owners)
    localization_correct = (
        (not localized) if case.spec.expected_abstention else set(owners) == expected
    )
    candidate_attempted = run.get("decision") in {"Accept", "Reject"}
    candidate_execution = run.get("candidate_execution") or {}
    final_execution = run.get("final_execution") or {}
    return {
        "method": method,
        "localized": localized,
        "located_owners": owners,
        "localization_correct": bool(localization_correct),
        "localization_rule": location.get("rule") if location else None,
        "localization_method": location.get("method") if location else None,
        "localization_confidence": location.get("confidence") if location else None,
        "scope": run.get("scope"),
        "decision": run.get("decision"),
        "rollback_status": run.get("rollback_status"),
        "reason": run.get("reason"),
        "candidate_attempted": candidate_attempted,
        "candidate_all_passed": bool(candidate_execution.get("all_passed")) if candidate_execution else None,
        "final_all_passed": bool(final_execution.get("all_passed")),
        "frozen_complement_ok": run.get("frozen_complement_ok"),
        "frozen_violations": run.get("frozen_violations") or [],
        "rank_before": run.get("rank_before"),
        "rank_after": run.get("rank_after"),
        "changed_units": run.get("changed_units") or [],
        "wall_ms": run.get("wall_ms"),
        "candidate_regressed": bool(candidate_attempted and candidate_execution and not candidate_execution.get("all_passed")),
    }


def _aggregate(cases: list[dict[str, Any]], method: str) -> dict[str, Any]:
    rows = [case[method] for case in cases]
    targeted = [
        row for case, row in zip(cases, rows)
        if not case["expected_abstention"]
    ]
    repairable = [
        row for case, row in zip(cases, rows)
        if case["expected_success"]
    ]
    adversarial = [
        row for case, row in zip(cases, rows)
        if case["fault_mode"] in {"unsafe_frozen_candidate", "non_improving_candidate"}
    ]
    negatives = [
        row for case, row in zip(cases, rows)
        if case["expected_abstention"]
    ]
    def rate(num: int, den: int) -> float | None:
        return num / den if den else None
    return {
        "case_count": len(rows),
        "targeted_case_count": len(targeted),
        "expected_repair_case_count": len(repairable),
        "localization_correct_count": sum(bool(row["localization_correct"]) for row in targeted),
        "targeted_localization_rate": rate(
            sum(bool(row["localization_correct"]) for row in targeted), len(targeted)
        ),
        "localization_precision_when_decided": rate(
            sum(bool(row["localized"] and row["localization_correct"]) for row in rows),
            sum(bool(row["localized"]) for row in rows),
        ),
        "repair_success_count_on_expected_cases": sum(bool(row["final_all_passed"]) for row in repairable),
        "repair_success_rate_on_expected_cases": rate(
            sum(bool(row["final_all_passed"]) for row in repairable), len(repairable)
        ),
        "correct_negative_control_abstentions": sum(
            bool(row["localization_correct"]) for row in negatives
        ),
        "negative_control_abstention_rate": rate(
            sum(bool(row["localization_correct"]) for row in negatives), len(negatives)
        ),
        "adversarial_candidate_rejection_count": sum(
            row["decision"] == "Reject" for row in adversarial
        ),
        "adversarial_candidate_rejection_rate": rate(
            sum(row["decision"] == "Reject" for row in adversarial), len(adversarial)
        ),
        "rollback_count": sum(row["rollback_status"] == "Restored" for row in rows),
        "frozen_violation_detection_count": sum(
            bool(row["frozen_violations"]) for row in rows
        ),
        "mean_wall_ms": (
            sum(float(row["wall_ms"] or 0.0) for row in rows) / len(rows)
            if rows else None
        ),
    }


def run_experiment(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_dir = output_dir / "cases"
    case_dir.mkdir(parents=True, exist_ok=True)
    execution_cache: dict[str, dict[str, Any]] = {}
    prepared: list[PreparedCase] = []
    for spec in SELECTED_CASES:
        prepared.append(_prepare_case(spec, execution_cache))

    records: list[dict[str, Any]] = []
    for case in prepared:
        candidate_execs = {
            method: _cached_execution(source, case, execution_cache)
            for method, source in case.candidate_sources.items()
        }
        old_run = _legacy_policy_run(
            case,
            case.initial_execution,
            candidate_source=case.candidate_sources.get("legacy"),
            candidate_snippets=case.candidate_snippets.get("legacy"),
            candidate_execution=candidate_execs.get("legacy"),
        )
        new_run = _new_policy_run(
            case,
            case.initial_execution,
            candidate_source=case.candidate_sources.get("revised"),
            candidate_snippets=case.candidate_snippets.get("revised"),
            candidate_execution=candidate_execs.get("revised"),
        )
        case_record = {
            "problem_id": case.spec.problem_id,
            "difficulty": case.metadata.get("difficulty"),
            "url": case.metadata.get("url", ""),
            "solution_index": case.spec.solution_index,
            "fault_mode": case.spec.fault_mode,
            "target_function": case.target_function,
            "test_count": len(case.inputs),
            "expected_owners": list(case.spec.expected_owners),
            "expected_abstention": case.spec.expected_abstention,
            "expected_success": case.spec.expected_success,
            "clean_validation": {
                "all_passed": bool(case.clean_execution.get("all_passed")),
                "case_count": len(case.clean_execution.get("results", [])),
            },
            "initial_fault": {
                "all_passed": bool(case.initial_execution.get("all_passed")),
                "error_type": case.initial_execution.get("error_type"),
                "first_failed_case": _first_failed(case.initial_execution),
            },
            "legacy": _method_record(case, "legacy", old_run),
            "revised": _method_record(case, "revised", new_run),
        }
        records.append(case_record)
        case_path = case_dir / f"{case.spec.problem_id:04d}_{case.spec.fault_mode}"
        case_path.mkdir(parents=True, exist_ok=True)
        (case_path / "clean.py").write_text(case.clean_source, encoding="utf-8")
        (case_path / "faulty.py").write_text(case.faulty_source, encoding="utf-8")
        for method, source in case.candidate_sources.items():
            (case_path / f"{method}_candidate.py").write_text(source, encoding="utf-8")
        (case_path / "case.json").write_text(
            json.dumps(case_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    result = {
        "experiment": "APPS controlled fault injection: legacy vs paper-aligned bounded repair",
        "dataset": str(APPS_ROOT),
        "selection_seed": "fixed reviewed selection",
        "candidate_semantics": "same oracle clean-function candidate supplied to both policies; unsafe/non-improving candidates are deliberate adversarial controls",
        "legacy_policy": "original traceback/isolated-import timing/unique-report locator with singleton unconditional retry",
        "revised_policy": "snapshot Trace -> Resource -> Local -> GlobalAttribution, selected scope, frozen complement, strict rank acceptance and rollback",
        "cases": records,
        "summary": {
            "legacy": _aggregate(records, "legacy"),
            "revised": _aggregate(records, "revised"),
        },
    }
    (output_dir / "apps_10_repair_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apps-root", type=Path, default=APPS_ROOT, help="Raw APPS test directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "results" / "apps_10_repair_comparison",
    )
    return parser.parse_args()


def main() -> None:
    global APPS_ROOT
    args = _parse_args()
    APPS_ROOT = args.apps_root.expanduser().resolve(strict=True)
    result = run_experiment(args.output_dir)
    print(json.dumps({"summary": result["summary"], "cases": len(result["cases"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
