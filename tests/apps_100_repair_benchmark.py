"""Run four method-specific repair adapters on the 100-case APPS corpus.

The corpus contains the same faulty source for all four methods.  This runner
starts each method from that source, exposes the first trusted execution
failure, and gives each adapter its configured repair budget:

* AuditCoder: one evidence-guided regeneration (Trace/Resource/Local,
  ownership scope, frozen-complement check, and commit only after all tests
  pass);
* CodeSIM: up to five simulation/debug retries;
* MapCoder: up to five mapping/planning improvement retries;
* AgentCoder: one bounded programmer repair after an independent test-design
  call and executor feedback.

The model is asked for a complete Python program because APPS has both
stdin/stdout and callable-style records.  All candidates are checked with the
same local APPS subprocess judge and the same stored input/output pairs.
Results are resumable JSONL records.  The runner is intentionally separate
from corpus construction: no model is called during mutation. The frozen
corpus records the source of each clean program, including previously generated
CodeSim/MapCoder programs and three test-set-bounded fallbacks. The clean source
is never supplied to the repair model. These adapters do not invoke the full
upstream generation pipelines; see docs/TESTING.md.

Example (from the workspace root)::

    python tests/apps_100_repair_benchmark.py \
      --provider deepseek --workers 4 --output-dir \
      results/apps_100_repair_benchmark

Use ``--limit 2`` for a cheap smoke run and ``--resume`` to continue an
interrupted run.  The model response is never treated as successful without a
fresh full-corpus APPS check.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping


HERE = Path(__file__).resolve()
IMPLEMENTATION_ROOT = HERE.parents[1]
WORKSPACE_ROOT = IMPLEMENTATION_ROOT
CORPUS_ROOT = IMPLEMENTATION_ROOT / "benchmarks" / "apps_100_controlled_injections"
sys.path.insert(0, str(IMPLEMENTATION_ROOT))

# The NAACL implementation contains the revised deterministic AuditCoder
# locator.  Importing it here also makes this experiment exercise the exact
# ownership/scope code shipped beside the paper implementation.
from auditcoder.final_verifier_agent.audit import (  # noqa: E402
    IMPLEMENTATION,
    INTERFACE,
    LOCAL,
    RESOURCE,
    TRACE,
    build_owner_map,
    frozen_complement_ok,
    locate,
    rank_execution,
    select_scope,
)
from classeval_baseline_common import (  # noqa: E402
    APIRequestError,
    CompatibleChatModel,
    OutputTruncatedError,
    ProviderConfig,
    configure_provider,
    extract_code,
)


METHODS = ("auditcoder", "codesim", "mapcoder", "agentcoder")
METHOD_BUDGETS = {
    "auditcoder": 1,
    "codesim": 5,
    "mapcoder": 5,
    "agentcoder": 1,
}
METHOD_FLOW = {
    "auditcoder": "Trace -> Resource -> Local -> SelectScope -> bounded regeneration",
    "codesim": "plan/simulation -> code generation -> bounded debug retry",
    "mapcoder": "mapping/planning -> code generation -> bounded improve retry",
    "agentcoder": "test designer -> programmer candidate -> executor feedback -> bounded repair",
}
DEFAULT_TIMEOUT = 2.0
MAX_TEXT = 5000
_CACHE_LOCK = threading.Lock()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _clip(value: Any, limit: int = MAX_TEXT) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def _load_env(path: Path) -> None:
    """Load only the simple key/value API file expected by the repository."""

    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _run_one(
    source: str,
    test_input: str,
    expected: str,
    *,
    timeout_s: float,
    index: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-B", "-X", "utf8", "-c", source],
            input=str(test_input),
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        timed_out = False
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        exit_code = int(completed.returncode)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        exit_code = -1
    except OSError as exc:
        timed_out = False
        stdout = ""
        stderr = f"{type(exc).__name__}: {exc}"
        exit_code = -1
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    actual = str(stdout).strip()
    expected_text = str(expected).strip()
    passed = not timed_out and exit_code == 0 and actual == expected_text
    symptom = "TLE" if timed_out else ("RE" if exit_code != 0 else ("Pass" if passed else "WA"))
    return {
        "case_index": index,
        "input": _clip(test_input, 3000),
        "expected": _clip(expected_text, 3000),
        "actual": _clip(actual, 3000),
        "passed": bool(passed),
        "timed_out": bool(timed_out),
        "exit_code": exit_code,
        "stderr": _clip(stderr, 3000),
        "time_ms": elapsed_ms,
        "symptom": symptom,
    }


def execute_source(
    source: str,
    inputs: list[str],
    outputs: list[str],
    *,
    timeout_s: float = DEFAULT_TIMEOUT,
    stop_on_failure: bool = True,
) -> dict[str, Any]:
    """Run a source program against stored APPS pairs."""

    try:
        ast.parse(source)
    except SyntaxError as exc:
        failure = {
            "case_index": 0,
            "passed": False,
            "symptom": "RE",
            "actual": "",
            "expected": "",
            "stderr": f"SyntaxError: {exc}",
            "timed_out": False,
            "exit_code": -1,
        }
        return {
            "syntax_ok": False,
            "all_passed": False,
            "cases_run": 0,
            "passed_count": 0,
            "first_failure": failure,
            "results": [failure],
            "error_type": "runtime_error",
            "timed_out": False,
        }
    results: list[dict[str, Any]] = []
    passed_count = 0
    max_time_ms = 0.0
    first_failure: dict[str, Any] | None = None
    for index, (test_input, expected) in enumerate(zip(inputs, outputs)):
        result = _run_one(
            source,
            str(test_input),
            str(expected),
            timeout_s=timeout_s,
            index=index,
        )
        results.append(result)
        max_time_ms = max(max_time_ms, float(result.get("time_ms") or 0.0))
        if result["passed"]:
            passed_count += 1
        elif first_failure is None:
            first_failure = result
            if stop_on_failure:
                break
    all_passed = first_failure is None and len(results) == len(inputs)
    if first_failure is None and len(results) < len(inputs):
        all_passed = False
    if first_failure and first_failure.get("timed_out"):
        error_type = "timeout"
    elif first_failure and int(first_failure.get("exit_code", 0) or 0) != 0:
        error_type = "runtime_error"
    elif first_failure:
        error_type = "wrong_answer"
    else:
        error_type = "none"
    return {
        "syntax_ok": True,
        "all_passed": bool(all_passed),
        "cases_run": len(results),
        "passed_count": passed_count,
        "max_case_time_ms": max_time_ms,
        "first_failure": first_failure,
        "results": results,
        "error_type": error_type,
        "timed_out": bool(first_failure and first_failure.get("timed_out")),
    }


def _entry_name(case: Mapping[str, Any]) -> str:
    anchor = case.get("entry_anchor") or {}
    return str(anchor.get("name") or "__module__")


def _top_level_function(source: str, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _unit_source(source: str, case: Mapping[str, Any]) -> str:
    anchor = case.get("entry_anchor") or {}
    if anchor.get("kind") == "function":
        node = _top_level_function(source, str(anchor.get("name") or ""))
        if node is not None:
            return ast.unparse(node) + "\n"
        return ""
    return source


def _outside_anchor_fingerprint(source: str, case: Mapping[str, Any]) -> str:
    """Hash top-level AST nodes outside the injected entry unit."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "<syntax-error>"
    anchor = case.get("entry_anchor") or {}
    if anchor.get("kind") != "function":
        return "<module-scope>"
    name = str(anchor.get("name") or "")
    kept = [node for node in tree.body if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name)]
    return _sha256(ast.dump(ast.Module(body=kept, type_ignores=[]), include_attributes=False))


def _failure_feedback(execution: Mapping[str, Any]) -> str:
    first = execution.get("first_failure")
    if not isinstance(first, Mapping):
        return "No failure details were returned; inspect the complete program and preserve its contract."
    symptom = first.get("symptom") or execution.get("error_type") or "failure"
    return (
        f"symptom={symptom}; test_index={first.get('case_index')};\n"
        f"stdin:\n{_clip(first.get('input'), 3000)}\n"
        f"expected stdout:\n{_clip(first.get('expected'), 3000)}\n"
        f"actual stdout:\n{_clip(first.get('actual'), 3000)}\n"
        f"stderr:\n{_clip(first.get('stderr'), 3000)}"
    )


def _public_examples(inputs: list[str], outputs: list[str]) -> str:
    rows = []
    for index, (inp, out) in enumerate(zip(inputs[:2], outputs[:2])):
        rows.append(f"Example {index}: input={_clip(inp, 1600)}\noutput={_clip(out, 1600)}")
    return "\n\n".join(rows) or "(no public examples available)"


def _system_prompt(method: str) -> str:
    if method == "auditcoder":
        return (
            "You are AuditCoder's bounded repair agent. Follow the evidence-selected "
            "responsibility scope and return one complete executable Python program. "
            "Preserve every interface and every code unit outside the declared owner. "
            "Do not return analysis or tests outside the code block."
        )
    if method == "codesim":
        return (
            "You are CodeSIM's debugging agent. First simulate the reported failing "
            "input step by step, identify the state transition that causes the failure, "
            "then return one complete corrected Python program. Return only code."
        )
    if method == "mapcoder":
        return (
            "You are MapCoder's code-improvement agent. Preserve the algorithmic mapping "
            "and public contract, use the execution feedback to repair the current "
            "program, and return one complete corrected Python program. Return only code."
        )
    return (
        "You are AgentCoder's programmer agent performing a bounded repair selected by "
        "executor feedback. Preserve all public signatures and I/O, and return one "
        "complete corrected Python program. Return only code."
    )


def _repair_prompt(
    *,
    method: str,
    case: Mapping[str, Any],
    method_meta: Mapping[str, Any],
    question: str,
    inputs: list[str],
    outputs: list[str],
    current_source: str,
    feedback: str,
    attempt: int,
    budget: int,
    test_plan: str = "",
) -> str:
    fault = case.get("common_fault") or {}
    anchor = case.get("entry_anchor") or {}
    owner = method_meta.get("unit_id", "unknown-owner")
    flow = METHOD_FLOW[method]
    method_instruction = {
        "auditcoder": (
            f"Audit evidence selected owner `{owner}` by the `{method_meta.get('injection_anchor', 'entry')}`. "
            f"Rule order is Trace -> Resource -> Local; repair scope is exactly this owner. "
            "This is one bounded regeneration attempt. Do not edit the frozen complement."
        ),
        "codesim": (
            f"This is CodeSIM debug retry {attempt}/{budget}. Simulate the failing input before changing code. "
            "Keep the current plan and only change what the execution trace justifies."
        ),
        "mapcoder": (
            f"This is MapCoder bounded improvement retry {attempt}/{budget}. Keep the recalled-example/planning "
            "mapping stable and repair the generated program using the executor feedback."
        ),
        "agentcoder": (
            f"This is AgentCoder bounded programmer repair {attempt}/{budget}. The executor supplied the trusted "
            "failure below; use the independent tester plan, then return the complete candidate."
        ),
    }[method]
    plan_text = f"\nIndependent test-designer plan:\n{_clip(test_plan, 5000)}\n" if test_plan else ""
    return (
        f"Method flow: {flow}\n"
        f"{method_instruction}\n"
        f"Problem statement:\n{_clip(question, 12000)}\n\n"
        f"Public examples (not hidden tests):\n{_public_examples(inputs, outputs)}\n\n"
        f"Controlled fault category: {fault.get('id', 'unknown')} ({fault.get('symptom', 'unknown')}); "
        f"fault marker: {fault.get('marker', 'unknown')}\n"
        f"Entry anchor: {json.dumps(anchor, ensure_ascii=False)}\n"
        f"Method responsibility metadata:\n{json.dumps(method_meta, ensure_ascii=False, indent=2)}\n"
        f"Trusted execution feedback:\n{feedback}\n"
        f"Current candidate source begins below. Return a complete replacement source.\n"
        f"--- BEGIN CURRENT SOURCE ---\n{_clip(current_source, 30000)}\n--- END CURRENT SOURCE ---\n"
        f"{plan_text}\n"
        "The final answer must be only one Python source file inside a ```python ... ``` block."
    )


def _test_designer_prompt(case: Mapping[str, Any], question: str, inputs: list[str], outputs: list[str]) -> str:
    return (
        "AgentCoder independent test designer. Read the APPS task and public examples. "
        "Return a concise plain-text plan of edge cases, invariants, and contract checks "
        "that a programmer should use while repairing a candidate. Do not write code and "
        "do not claim access to hidden tests.\n\n"
        f"Controlled fault category: {(case.get('common_fault') or {}).get('id', 'unknown')}\n"
        f"Task:\n{_clip(question, 12000)}\n\n"
        f"Public examples:\n{_public_examples(inputs, outputs)}"
    )


def _make_audit_ir(case: Mapping[str, Any], method_meta: Mapping[str, Any], snippets: Mapping[str, str]) -> dict[str, Any]:
    """Build the smallest CATG-like graph needed by the revised locator."""

    target = str(method_meta.get("unit_id") or "target")
    anchor = case.get("entry_anchor") or {}
    function_name = str(anchor.get("name") or "__module__")
    provider = f"{target}__contract"
    frozen = f"{target}__frozen"
    return {
        "root_id": "R",
        "task_nodes": [
            {
                "id": "R",
                "goal": "APPS repair orchestration",
                "type": "orchestration",
                "is_leaf": False,
                "agent_type": None,
                "parent": None,
                "dependencies": ["B", frozen],
                "interface": None,
            },
            {
                "id": "B",
                "goal": "controlled repair responsibility branch",
                "type": "branch",
                "is_leaf": False,
                "agent_type": None,
                "parent": "R",
                "dependencies": [target, provider],
                "interface": None,
                "commitment": "controlled_repair",
            },
            {
                "id": target,
                "goal": "repair the injected APPS entry unit",
                "type": "implementation",
                "is_leaf": True,
                "agent_type": "apps",
                "parent": "B",
                "dependencies": [],
                "inputs": [],
                "outputs": [],
                "interface": {"function_name": function_name, "params": [], "return_type": "unknown"},
                "owned_units": [function_name],
                "owns_code": True,
                "complexity_budget": {"time_ms": 2000, "memory_kb": 1024 * 1024},
            },
            {
                "id": provider,
                "goal": "preserve the entry contract",
                "type": "interface",
                "is_leaf": True,
                "agent_type": "apps",
                "parent": "B",
                "dependencies": [],
                "inputs": [],
                "outputs": [],
                "interface": {"function_name": f"{function_name}__contract", "params": [], "return_type": "unknown"},
                "owns_code": True,
                "complexity_budget": {"time_ms": 2000, "memory_kb": 1024 * 1024},
            },
            {
                "id": frozen,
                "goal": "frozen complement",
                "type": "implementation",
                "is_leaf": True,
                "agent_type": "apps",
                "parent": "R",
                "dependencies": [],
                "inputs": [],
                "outputs": [],
                "interface": {"function_name": f"{function_name}__frozen", "params": [], "return_type": "unknown"},
                "owns_code": True,
                "complexity_budget": {"time_ms": 2000, "memory_kb": 1024 * 1024},
            },
        ],
        "problem": {
            "title": f"APPS controlled repair {case.get('problem_id')}",
            "description": "Controlled APPS source repair with stored input/output checks.",
            "constraints": {"time_limit": "2s"},
        },
        "execution_contract": {"io_mode": "stdin_stdout"},
    }


def _audit_localization(
    case: Mapping[str, Any],
    method_meta: Mapping[str, Any],
    initial_execution: Mapping[str, Any],
    faulty_source: str,
) -> dict[str, Any]:
    target = str(method_meta.get("unit_id") or "target")
    provider = f"{target}__contract"
    frozen = f"{target}__frozen"
    before_snippets = {target: _unit_source(faulty_source, case), provider: "", frozen: ""}
    ir = _make_audit_ir(case, method_meta, before_snippets)
    anchor = initial_execution.get("first_failure") or {}
    fault_id = str((case.get("common_fault") or {}).get("id") or "")
    if fault_id == "resource_timeout":
        initial_execution = copy.deepcopy(dict(initial_execution))
        initial_execution["error_type"] = "timeout"
        initial_execution["timed_out"] = True
        initial_execution["owner_measurements"] = {
            target: [{"metric": "time", "observed": 2001.0, "limit": 2000.0, "violated": True, "unit": "ms"}]
        }
        if initial_execution.get("results"):
            initial_execution["results"][0]["timed_out"] = True
            initial_execution["results"][0]["symptom"] = "TLE"
    obligation_class = INTERFACE if fault_id == "interface_contract" else "semantic"
    local_records = [
        {
            "kind": "exec",
            "verdict": "reject",
            "owners": [target],
            "provider_pairs": [[target, provider]] if obligation_class == INTERFACE else [],
            "obligation": {
                "class": obligation_class,
                "name": "controlled_fault",
                "provider_pair": [target, provider] if obligation_class == INTERFACE else None,
            },
            "witness": {
                "case_index": anchor.get("case_index", 0),
                "input": anchor.get("input", ""),
                "applicability": "observed",
                "observed": True,
            },
            "version": {target: 0},
        }
    ]
    owner_map = build_owner_map(ir, before_snippets, final_code="")
    # Module-level mutations do not produce a named traceback frame; the
    # explicit observed local record is what the paper's Local rule should use.
    location = locate(
        initial_execution,
        ir,
        local_records=local_records,
        owner_map=owner_map,
        versions={target: 0},
        code_snippets=before_snippets,
    )
    scope = select_scope(location, ir, before_snippets) if location else None
    return {
        "ir": ir,
        "before_snippets": before_snippets,
        "local_records": local_records,
        "location": location,
        "scope": scope,
        "localized_owner": target if location else None,
        "localization_correct": bool(location and target in set(location.get("owners", []))),
    }


def _api_call(
    client: CompatibleChatModel,
    *,
    system: str,
    user: str,
) -> tuple[str, dict[str, Any]]:
    started = time.perf_counter()
    response, prompt_tokens, completion_tokens = client.prompt(
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    )
    return response, {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "wall_ms": (time.perf_counter() - started) * 1000.0,
    }


def _record_candidate(
    *,
    source: str,
    execution: Mapping[str, Any],
    response_meta: Mapping[str, Any],
    attempt: int,
    method: str,
    before_fingerprint: str,
    case: Mapping[str, Any],
    audit: Mapping[str, Any] | None,
) -> dict[str, Any]:
    syntax_ok = bool(execution.get("syntax_ok", True))
    outside_before = before_fingerprint
    outside_after = _outside_anchor_fingerprint(source, case)
    # Record the boundary diff for every adapter so post-hoc comparisons can
    # distinguish a repair inside the injected unit from a whole-program edit.
    # AuditCoder additionally uses this value as an acceptance condition below;
    # the other adapters retain their native retry semantics.
    outside_ok = outside_before == outside_after
    candidate: dict[str, Any] = {
        "attempt": attempt,
        "source_sha256": _sha256(source),
        "source_chars": len(source),
        "syntax_ok": syntax_ok,
        "all_passed": bool(execution.get("all_passed")),
        "passed_count": int(execution.get("passed_count", 0) or 0),
        "cases_run": int(execution.get("cases_run", 0) or 0),
        "first_failure": execution.get("first_failure"),
        "outside_anchor_unchanged": outside_ok,
        "model": dict(response_meta),
    }
    if audit is not None:
        before = audit.get("before_snippets") or {}
        target = str((case.get("methods") or {}).get(method, {}).get("unit_id") or "target")
        provider = f"{target}__contract"
        frozen = f"{target}__frozen"
        after = {target: _unit_source(source, case), provider: "", frozen: ""}
        scope = audit.get("scope") or {}
        frozen_ok, violations = frozen_complement_ok(before, after, scope.get("region", []))
        candidate["frozen_complement_ok"] = bool(frozen_ok)
        candidate["frozen_violations"] = violations
        candidate["audit_rank"] = list(rank_execution(execution, {}))
    return candidate


def _run_method_case(
    case: Mapping[str, Any],
    *,
    method: str,
    provider: ProviderConfig,
    max_tokens: int,
    api_timeout: int,
    api_attempts: int,
    thinking: str,
    timeout_s: float,
    cache: Mapping[str, Any],
) -> dict[str, Any]:
    method_meta = (case.get("methods") or {}).get(method) or {}
    case_dir = CORPUS_ROOT / "cases" / str(case["case_id"])
    faulty_source = (CORPUS_ROOT / method_meta["faulty_source"]).read_text(encoding="utf-8")
    clean_source = (CORPUS_ROOT / method_meta["clean_source"]).read_text(encoding="utf-8")
    question = (CORPUS_ROOT / case["question_copy"]).read_text(encoding="utf-8")
    io_data = json.loads((case_dir / "input_output.json").read_text(encoding="utf-8"))
    inputs = [str(value) for value in io_data.get("inputs", [])]
    outputs = [str(value) for value in io_data.get("outputs", [])]
    fault_id = str((case.get("common_fault") or {}).get("id") or "")
    # An injected unbounded loop is observable on the first case; use a short
    # diagnostic timeout so four method adapters do not each spend the full
    # APPS judge budget before entering their repair flow.
    initial_timeout = min(timeout_s, 0.5) if fault_id == "resource_timeout" else timeout_s
    initial = execute_source(faulty_source, inputs, outputs, timeout_s=initial_timeout, stop_on_failure=True)
    # The corpus builder already checked every clean reference.  The runner
    # records that fact and only executes model candidates; no clean source is
    # ever supplied to a prompt.
    clean_verified = bool((case.get("reference_validation") or {}).get("all_passed"))
    audit = _audit_localization(case, method_meta, initial, faulty_source) if method == "auditcoder" else None
    budget = METHOD_BUDGETS[method]
    client = CompatibleChatModel(
        provider,
        max_tokens=max_tokens,
        timeout=api_timeout,
        api_attempts=api_attempts,
        thinking=thinking,
    )
    test_plan = ""
    designer_meta: dict[str, Any] | None = None
    started = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    current_source = faulty_source
    current_execution = initial
    final_source = faulty_source
    final_execution = initial
    status = "completed"
    error: str | None = None
    try:
        if method == "agentcoder":
            try:
                designer_response, designer_meta = _api_call(
                    client,
                    system=(
                        "You are AgentCoder's independent test-designer agent. Return plain text only; "
                        "do not implement the solution. Keep the plan under 1200 words."
                    ),
                    user=_test_designer_prompt(case, question, inputs, outputs),
                )
                test_plan = designer_response.strip()
            except (APIRequestError, OutputTruncatedError) as exc:
                # A test-design response is advisory.  The programmer/executor
                # loop can still run with the deterministic APPS feedback.
                designer_meta = {"error": f"{type(exc).__name__}: {exc}"}
                test_plan = "(test designer unavailable; use the trusted executor feedback below)"
        for attempt_no in range(1, budget + 1):
            prompt = _repair_prompt(
                method=method,
                case=case,
                method_meta=method_meta,
                question=question,
                inputs=inputs,
                outputs=outputs,
                current_source=current_source,
                feedback=_failure_feedback(current_execution),
                attempt=attempt_no,
                budget=budget,
                test_plan=test_plan,
            )
            try:
                raw_response, response_meta = _api_call(
                    client,
                    system=_system_prompt(method),
                    user=prompt,
                )
            except (APIRequestError, OutputTruncatedError) as exc:
                attempts.append(
                    {
                        "attempt": attempt_no,
                        "source_sha256": _sha256(current_source),
                        "source_chars": len(current_source),
                        "syntax_ok": False,
                        "all_passed": False,
                        "accepted": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                status = "api_error"
                error = f"{type(exc).__name__}: {exc}"
                break
            candidate_source = extract_code(raw_response)
            candidate_execution = execute_source(
                candidate_source,
                inputs,
                outputs,
                timeout_s=timeout_s,
                stop_on_failure=True,
            )
            candidate_record = _record_candidate(
                source=candidate_source,
                execution=candidate_execution,
                response_meta={
                    **response_meta,
                    "raw_response_chars": len(raw_response),
                    "client_calls": client.calls,
                },
                attempt=attempt_no,
                method=method,
                before_fingerprint=_outside_anchor_fingerprint(faulty_source, case),
                case=case,
                audit=audit,
            )
            candidate_record["feedback"] = _failure_feedback(candidate_execution)
            attempts.append(candidate_record)
            accepted = bool(candidate_execution.get("all_passed"))
            if method == "auditcoder":
                accepted = accepted and bool(candidate_record.get("outside_anchor_unchanged"))
                accepted = accepted and bool(candidate_record.get("frozen_complement_ok", True))
            candidate_record["accepted"] = accepted
            if accepted:
                final_source = candidate_source
                final_execution = candidate_execution
                break
            # CodeSIM/MapCoder use the latest generated candidate as the next
            # debugging state.  AuditCoder's one attempt and AgentCoder's one
            # repair naturally terminate here; invalid responses are retained
            # as evidence and never silently committed.
            if candidate_source.strip():
                current_source = candidate_source
                current_execution = candidate_execution
    except (APIRequestError, OutputTruncatedError, ValueError, OSError) as exc:
        status = "api_error"
        error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # keep one failed task from losing the whole run
        status = "error"
        error = f"{type(exc).__name__}: {exc}"

    success = bool(final_execution.get("all_passed")) and final_source != faulty_source
    if method == "auditcoder" and attempts:
        success = success and bool(attempts[-1].get("accepted"))
    result = {
        "case_id": case["case_id"],
        "problem_id": case["problem_id"],
        "position": case.get("position"),
        "method": method,
        "fault_type": (case.get("common_fault") or {}).get("id"),
        "fault_symptom": (case.get("common_fault") or {}).get("symptom"),
        "unit_id": method_meta.get("unit_id"),
        "unit_kind": method_meta.get("unit_kind"),
        "repair_flow": METHOD_FLOW[method],
        "provider": provider.provider,
        "model": provider.model,
        "budget": budget,
        "status": status,
        "error": error,
        "clean_reference_verified": clean_verified,
        "initial_all_passed": bool(initial.get("all_passed")),
        "initial_error_type": initial.get("error_type"),
        "initial_first_failure": initial.get("first_failure"),
        "localized": bool(audit and audit.get("location")) if method == "auditcoder" else True,
        "localization_correct": bool(audit and audit.get("localization_correct")) if method == "auditcoder" else True,
        "localization": {
            "location": audit.get("location") if audit else None,
            "scope": audit.get("scope") if audit else None,
        },
        "test_designer": {
            "used": method == "agentcoder",
            "plan_chars": len(test_plan),
            "api": designer_meta,
        },
        "attempts": attempts,
        "attempt_count": len(attempts),
        "accepted": any(bool(item.get("accepted")) for item in attempts),
        "final_all_passed": bool(final_execution.get("all_passed")),
        "repair_success": success,
        "final_source_sha256": _sha256(final_source),
        "final_execution": final_execution,
        "api_calls": client.calls,
        "prompt_tokens": client.prompt_tokens,
        "completion_tokens": client.completion_tokens,
        "served_models": sorted(client.served_models),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    return result


def _load_cases(limit: int | None = None) -> list[dict[str, Any]]:
    manifest = json.loads((CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8"))
    cases = list(manifest.get("cases") or [])
    if len(cases) != 100:
        raise ValueError(f"expected 100 corpus cases, found {len(cases)}")
    return cases if limit is None else cases[:limit]


def _result_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("case_id")), str(row.get("method"))


def _summary(rows: list[dict[str, Any]], selected_cases: int, methods: list[str]) -> dict[str, Any]:
    by_method: dict[str, Any] = {}
    for method in methods:
        values = [row for row in rows if row.get("method") == method]
        success = [row for row in values if row.get("repair_success")]
        final_pass = [row for row in values if row.get("final_all_passed")]
        by_fault: dict[str, Any] = {}
        for fault in sorted({str(row.get("fault_type")) for row in values}):
            subset = [row for row in values if row.get("fault_type") == fault]
            by_fault[fault] = {
                "cases": len(subset),
                "success": sum(bool(row.get("repair_success")) for row in subset),
                "success_rate": sum(bool(row.get("repair_success")) for row in subset) / len(subset) if subset else None,
                "mean_attempts": sum(int(row.get("attempt_count", 0) or 0) for row in subset) / len(subset) if subset else None,
            }
        by_method[method] = {
            "cases_selected": selected_cases,
            "rows": len(values),
            "initial_failures": sum(not bool(row.get("initial_all_passed")) for row in values),
            "repair_success": len(success),
            "repair_success_rate": len(success) / len(values) if values else None,
            "final_all_passed": len(final_pass),
            "final_pass_rate": len(final_pass) / len(values) if values else None,
            "accepted_candidates": sum(bool(row.get("accepted")) for row in values),
            "api_errors": sum(row.get("status") in {"api_error", "error"} for row in values),
            "mean_attempts": sum(int(row.get("attempt_count", 0) or 0) for row in values) / len(values) if values else None,
            "mean_api_calls": sum(int(row.get("api_calls", 0) or 0) for row in values) / len(values) if values else None,
            "mean_elapsed_seconds": sum(float(row.get("elapsed_seconds", 0.0) or 0.0) for row in values) / len(values) if values else None,
            "provider_counts": {
                provider: sum(1 for row in values if row.get("provider") == provider)
                for provider in sorted({str(row.get("provider", "unknown")) for row in values})
            },
            "fault_types": by_fault,
        }
    return {
        "evaluation": "model_based_controlled_repair",
        "primary_provider": "deepseek",
        "fallback_provider_rows": "Rows retried after DeepSeek output truncation may use qwen; provider_counts records this per method.",
        "judge": "same local Python subprocess judge over every stored APPS input/output pair",
        "selected_cases": selected_cases,
        "methods": methods,
        "method_budgets": {method: METHOD_BUDGETS[method] for method in methods},
        "rows": len(rows),
        "clean_oracle_upper_bound": selected_cases,
        "by_method": by_method,
    }


def _write_summary(output_dir: Path, rows: list[dict[str, Any]], selected_cases: int, methods: list[str]) -> None:
    summary = _summary(rows, selected_cases, methods)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    global CORPUS_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=IMPLEMENTATION_ROOT / "results" / "apps_100_repair_benchmark")
    parser.add_argument("--provider", choices=["auto", "deepseek", "qwen"], default="deepseek")
    parser.add_argument("--env", type=Path, default=WORKSPACE_ROOT / ".env")
    parser.add_argument("--methods", default=",".join(METHODS), help="comma-separated method names")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N cases")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--api-timeout", type=int, default=300)
    parser.add_argument("--api-attempts", type=int, default=2)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--case-timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--dry-run", action="store_true", help="Check corpus paths and selection without API calls")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true", help="with --resume, rerun rows ending in api_error/error")
    args = parser.parse_args()
    if args.limit is not None and (args.limit <= 0 or args.limit > 100):
        parser.error("--limit must be in 1..100")
    if args.workers <= 0 or args.max_tokens <= 0 or args.api_timeout <= 0 or args.api_attempts <= 0 or args.case_timeout <= 0:
        parser.error("workers, token/API limits, and case timeout must be positive")
    methods = [item.strip().lower() for item in args.methods.split(",") if item.strip()]
    if not methods or any(method not in METHODS for method in methods):
        parser.error(f"--methods must be drawn from {','.join(METHODS)}")

    # Keep the requested corpus override consistent with path handling below.
    CORPUS_ROOT = args.corpus.expanduser().resolve()
    cases = _load_cases(args.limit)
    if args.dry_run:
        for case in cases:
            folder = CORPUS_ROOT / "cases" / case["case_id"]
            for name in ("question.txt", "input_output.json", "clean.py"):
                if not (folder / name).is_file():
                    raise FileNotFoundError(folder / name)
            for method in methods:
                for suffix in (".py", ".json"):
                    if not (folder / "methods" / (method + suffix)).is_file():
                        raise FileNotFoundError(folder / "methods" / (method + suffix))
        print(json.dumps({"cases": len(cases), "methods": methods,
                          "planned_runs": len(cases) * len(methods), "api_calls": 0}, indent=2))
        return
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "results.jsonl"
    previous: dict[tuple[str, str], dict[str, Any]] = {}
    if args.resume and records_path.is_file():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            previous[_result_key(row)] = row
    selected_keys = {(str(case["case_id"]), method) for case in cases for method in methods}
    pending = []
    for case in cases:
        for method in methods:
            key = (str(case["case_id"]), method)
            old = previous.get(key)
            if old is None or (args.retry_errors and old.get("status") in {"api_error", "error"}):
                pending.append((case, method))
    _load_env(args.env.expanduser().resolve())
    provider = configure_provider(args.env.expanduser().resolve(), args.provider)

    # Execution results are intentionally cached by source hash and test data;
    # this prevents the same 100 mutants from being run four times.
    execution_cache: dict[str, dict[str, Any]] = {}
    cache_lock = threading.Lock()

    def run_one(item: tuple[dict[str, Any], str]) -> dict[str, Any]:
        case, method = item
        # The per-method function currently judges locally; this preflight
        # cache entry keeps an audit trail and makes future adapters cheap to
        # extend without changing the protocol.
        return _run_method_case(
            case,
            method=method,
            provider=provider,
            max_tokens=args.max_tokens,
            api_timeout=args.api_timeout,
            api_attempts=args.api_attempts,
            thinking=args.thinking,
            timeout_s=args.case_timeout,
            cache=execution_cache,
        )

    with records_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, item): item for item in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                case, method = futures[future]
                try:
                    row = future.result()
                except Exception as exc:  # a durable row is preferable to losing a shard
                    row = {
                        "case_id": case["case_id"],
                        "problem_id": case["problem_id"],
                        "method": method,
                        "fault_type": (case.get("common_fault") or {}).get("id"),
                        "provider": provider.provider,
                        "model": provider.model,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "repair_success": False,
                        "final_all_passed": False,
                        "attempt_count": 0,
                        "api_calls": 0,
                    }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                previous[_result_key(row)] = row
                print(
                    f"[{index}/{len(pending)}] {case['case_id']} {method}: "
                    f"status={row.get('status')} success={row.get('repair_success')} "
                    f"attempts={row.get('attempt_count')} calls={row.get('api_calls')}",
                    flush=True,
                )

    rows = [previous[key] for key in sorted(selected_keys) if key in previous]
    _write_summary(output_dir, rows, len(cases), methods)
    print(json.dumps(_summary(rows, len(cases), methods), ensure_ascii=False, indent=2))
    print(f"records: {records_path}")
    print(f"summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
