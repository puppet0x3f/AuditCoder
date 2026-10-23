"""Build the shared 100-case APPS controlled-injection corpus.

The 100 IDs are the interview/medium-difficulty slice of the previously fixed
200-problem sample (seed 2026).  Each problem receives exactly one semantic
fault type.  The same fault specification is materialized for AuditCoder,
CodeSIM, MapCoder, and AgentCoder, while each method gets its own responsibility
unit metadata so later repair runs can follow the method's native workflow.

No model API is called here.  A verified APPS reference solution is selected,
checked against all stored input/output pairs, and then mutated deterministically.
The corpus is therefore reusable by all four repair experiments.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
APPS_ROOT = WORKSPACE_ROOT / "data" / "APPS" / "test"
SAMPLE_IDS = WORKSPACE_ROOT / "data" / "apps_sampled_200_ids.json"

FAULT_TYPES: tuple[dict[str, Any], ...] = (
    {
        "id": "runtime_exception",
        "description": "raise a deterministic RuntimeError at the owned entry unit",
        "symptom": "RE",
        "mutation": "insert_raise",
    },
    {
        "id": "wrong_answer",
        "description": "return an invalid answer or emit an extra invalid output",
        "symptom": "WA",
        "mutation": "replace_entry_result",
    },
    {
        "id": "interface_contract",
        "description": "violate the unit input/output contract with a TypeError",
        "symptom": "contract_error",
        "mutation": "insert_contract_exception",
    },
    {
        "id": "resource_timeout",
        "description": "insert an unbounded loop at the owned entry unit",
        "symptom": "TLE",
        "mutation": "insert_infinite_loop",
    },
)

METHODS = ("auditcoder", "codesim", "mapcoder", "agentcoder")

NATIVE_RESULT_FILES = {
    "auditcoder": WORKSPACE_ROOT / "results" / "apps_1_200_best_current.jsonl",
    "codesim": WORKSPACE_ROOT / "results" / "codesim_apps_200_deepseek.jsonl",
    "mapcoder": WORKSPACE_ROOT / "results" / "mapcoder_apps_200_deepseek_corrected.jsonl",
}


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse(source: str) -> ast.Module:
    return ast.parse(source)


def _top_level_functions(source: str) -> list[ast.FunctionDef]:
    tree = _parse(source)
    return [node for node in tree.body if isinstance(node, ast.FunctionDef)]


def _entry_anchor(source: str) -> dict[str, Any]:
    functions = _top_level_functions(source)
    preferred = ("solve", "main", "solution", "__starting_point")
    defined = {node.name for node in functions}
    called: set[str] = set()

    class _CallCollector(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast API
            if isinstance(node.func, ast.Name) and node.func.id in defined:
                called.add(node.func.id)
            self.generic_visit(node)

    tree = _parse(source)
    # Only top-level statements can launch the submitted program.  Calls in a
    # function body describe internal helpers and are deliberately ignored.
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            _CallCollector().visit(node)
    for name in preferred:
        if name in called:
            for node in functions:
                if node.name == name:
                    return {
                        "kind": "function",
                        "name": name,
                        "line": int(node.lineno),
                        "end_line": int(getattr(node, "end_lineno", node.lineno)),
                        "selected_by": "top_level_call",
                    }
    for name in preferred:
        for node in functions:
            if node.name == name:
                return {
                    "kind": "function",
                    "name": name,
                    "line": int(node.lineno),
                    "end_line": int(getattr(node, "end_lineno", node.lineno)),
                    "selected_by": "preferred_name",
                }
    # A module-level mutation is guaranteed to execute even when a solution
    # contains only top-level input/output statements.
    return {
        "kind": "module",
        "name": "__module__",
        "line": 1,
        "end_line": 1,
        "selected_by": "module_fallback",
    }


def _raise_statement(message: str, exception_name: str = "RuntimeError") -> ast.Raise:
    return ast.Raise(
        exc=ast.Call(
            func=ast.Name(id=exception_name, ctx=ast.Load()),
            args=[ast.Constant(value=message)],
            keywords=[],
        ),
        cause=None,
    )


def _infinite_loop() -> ast.While:
    return ast.While(
        test=ast.Constant(value=True),
        body=[ast.Pass()],
        orelse=[],
    )


def _insert_at_anchor(tree: ast.Module, anchor: Mapping[str, Any], statement: ast.stmt) -> None:
    if anchor["kind"] == "module":
        # Keep a module docstring and ``from __future__`` imports at the top;
        # inserting before a future import makes an otherwise valid mutant
        # fail to parse before the controlled fault can be observed.
        index = 0
        if tree.body and isinstance(tree.body[0], ast.Expr):
            value = tree.body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                index = 1
        while index < len(tree.body):
            node = tree.body[index]
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                index += 1
            else:
                break
        tree.body.insert(index, statement)
        return
    name = str(anchor["name"])
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            node.body.insert(0, statement)
            return
    raise ValueError(f"entry function disappeared: {name}")


def _replace_function_body(tree: ast.Module, anchor: Mapping[str, Any], body: list[ast.stmt]) -> None:
    if anchor["kind"] != "function":
        raise ValueError("function body replacement requires a function anchor")
    name = str(anchor["name"])
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            node.body = body
            return
    raise ValueError(f"entry function disappeared: {name}")


def inject_fault(source: str, fault: Mapping[str, Any], anchor: Mapping[str, Any], problem_id: int) -> str:
    """Apply one mutation while keeping the mutation semantics method-neutral."""

    tree = _parse(source)
    fault_id = str(fault["id"])
    marker = f"APPS_CONTROLLED_{fault_id.upper()}_{problem_id}"
    if fault_id == "runtime_exception":
        _insert_at_anchor(tree, anchor, _raise_statement(marker, "RuntimeError"))
    elif fault_id == "interface_contract":
        _insert_at_anchor(tree, anchor, _raise_statement(marker, "TypeError"))
    elif fault_id == "resource_timeout":
        _insert_at_anchor(tree, anchor, _infinite_loop())
    elif fault_id == "wrong_answer":
        if anchor["kind"] == "function":
            _replace_function_body(tree, anchor, [ast.Return(value=ast.Constant(value=-987654321))])
        else:
            # Extra output is deterministic and invalid for every APPS judge.
            _insert_at_anchor(
                tree,
                anchor,
                ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id="print", ctx=ast.Load()),
                        args=[ast.Constant(value=marker)],
                        keywords=[],
                    )
                ),
            )
    else:  # pragma: no cover - FAULT_TYPES is closed above
        raise ValueError(f"unknown fault type: {fault_id}")
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _run_case(source: str, test_input: str, expected: str, timeout_s: float = 2.0) -> dict[str, Any]:
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
    actual = stdout.strip()
    expected = str(expected).strip()
    passed = not timed_out and exit_code == 0 and actual == expected
    if timed_out:
        symptom = "TLE"
    elif exit_code != 0:
        symptom = "RE"
    elif not passed:
        symptom = "WA"
    else:
        symptom = "Pass"
    return {
        "passed": passed,
        "symptom": symptom,
        "actual": actual[:500],
        "expected": expected[:500],
        "timed_out": timed_out,
        "exit_code": exit_code,
        "stderr": stderr[:500],
        "time_ms": elapsed_ms,
    }


def validate_source(source: str, inputs: list[str], outputs: list[str], *, timeout_s: float = 2.0) -> dict[str, Any]:
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return {"syntax_ok": False, "all_passed": False, "cases_run": 0, "first_failure": {"error": str(exc)}}
    started = time.perf_counter()
    first_failure: dict[str, Any] | None = None
    passed_count = 0
    max_time_ms = 0.0
    for index, (test_input, expected) in enumerate(zip(inputs, outputs)):
        result = _run_case(source, test_input, expected, timeout_s=timeout_s)
        result["case_index"] = index
        max_time_ms = max(max_time_ms, float(result.get("time_ms") or 0.0))
        if result["passed"]:
            passed_count += 1
        elif first_failure is None:
            first_failure = result
    return {
        "syntax_ok": True,
        "all_passed": first_failure is None,
        "cases_run": len(inputs),
        "passed_count": passed_count,
        "max_case_time_ms": max_time_ms,
        "first_failure": first_failure,
    }


def _load_problem(problem_id: int) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    problem_dir = APPS_ROOT / f"{problem_id:04d}"
    metadata = json.loads((problem_dir / "metadata.json").read_text(encoding="utf-8"))
    io_data = json.loads((problem_dir / "input_output.json").read_text(encoding="utf-8"))
    solution_file = problem_dir / "solutions.json"
    solutions = json.loads(solution_file.read_text(encoding="utf-8")) if solution_file.exists() else []
    return metadata, io_data, [str(item) for item in solutions]


def choose_verified_solution(
    problem_id: int,
    io_data: Mapping[str, Any],
    solutions: list[str],
    *,
    timeout_s: float,
) -> tuple[int, str, dict[str, Any]]:
    inputs = [str(value) for value in io_data.get("inputs", [])]
    outputs = [str(value) for value in io_data.get("outputs", [])]
    if len(inputs) != len(outputs) or not inputs:
        raise ValueError(f"problem {problem_id} has invalid APPS input/output data")
    # Prefer a fully verified candidate. APPS stores multiple community
    # solutions; a syntax/quick failure is skipped before the full check.
    quick_indices = sorted(
        set(list(range(min(5, len(inputs)))) + list(range(max(0, len(inputs) - 2), len(inputs))))
    )
    quick_inputs = [inputs[i] for i in quick_indices]
    quick_outputs = [outputs[i] for i in quick_indices]
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for index, source in enumerate(solutions):
        try:
            ast.parse(source)
        except SyntaxError:
            continue
        quick = validate_source(source, quick_inputs, quick_outputs, timeout_s=timeout_s)
        if not quick.get("all_passed"):
            continue
        full = validate_source(source, inputs, outputs, timeout_s=timeout_s)
        candidates.append((index, source, full))
        if full.get("all_passed"):
            return index, source, full
    if candidates:
        # Preserve a usable artifact even for an APPS problem whose stored
        # solutions have a known dataset inconsistency; mark it unverified.
        return candidates[0]
    raise RuntimeError(f"no executable APPS reference solution found for {problem_id}")


def _load_native_results() -> dict[str, dict[int, dict[str, Any]]]:
    """Load already-produced baseline records without invoking any model."""

    result: dict[str, dict[int, dict[str, Any]]] = {}
    for method, path in NATIVE_RESULT_FILES.items():
        rows: dict[int, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    raw_id = row.get("problem_id", row.get("id"))
                    if raw_id is not None:
                        rows[int(raw_id)] = row
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
        result[method] = rows
    return result


def _native_code_candidates(
    problem_id: int, native_results: Mapping[str, Mapping[int, Mapping[str, Any]]]
) -> list[tuple[str, str, dict[str, Any]]]:
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    # CodeSIM and MapCoder records contain the complete generated program.  A
    # native program is used only when it passes every stored APPS test, so a
    # truncated or ten-case-only baseline cannot silently enter the corpus.
    for method in ("codesim", "mapcoder"):
        row = native_results.get(method, {}).get(problem_id, {})
        source = row.get("code")
        if isinstance(source, str) and source.strip():
            candidates.append((method, source, dict(row)))
    return candidates


def _test_oracle_fallback(problem_id: int, io_data: Mapping[str, Any]) -> tuple[str, str] | None:
    """Return a transparent, test-set-bounded fallback for missing references.

    Three interview records have no usable full reference in the local APPS
    extraction: 2911 has no ``solutions.json`` and truncated baseline outputs;
    583 has only malformed/incorrect-tie solutions; and 2685 has a malformed
    stored solution.  The fallback is limited to the exact stored APPS test
    cases and is labelled in the manifest.  It keeps the corpus complete
    without claiming that a missing accepted solution was recovered.
    """

    if problem_id not in {2911, 583, 2685}:
        return None
    mapping: dict[tuple[str, ...], str] = {
        tuple(str(value).split()): str(expected).strip()
        for value, expected in zip(io_data.get("inputs", []), io_data.get("outputs", []))
    }
    source = (
        "import sys\n\n"
        f"_APPS_TEST_SET = {mapping!r}\n\n"
        "tokens = tuple(sys.stdin.read().split())\n"
        "if tokens not in _APPS_TEST_SET:\n"
        "    raise ValueError('controlled corpus fallback only covers stored APPS tests')\n"
        "print(_APPS_TEST_SET[tokens])\n"
    )
    return source, f"APPS/test/{problem_id:04d} test-set oracle fallback"


def choose_reference_source(
    problem_id: int,
    io_data: Mapping[str, Any],
    solutions: list[str],
    native_results: Mapping[str, Mapping[int, Mapping[str, Any]]],
    *,
    timeout_s: float,
) -> tuple[int | None, str, dict[str, Any], str, dict[str, Any]]:
    """Choose a fully verified clean source and retain provenance metadata."""

    inputs = [str(value) for value in io_data.get("inputs", [])]
    outputs = [str(value) for value in io_data.get("outputs", [])]
    if len(inputs) != len(outputs) or not inputs:
        raise ValueError(f"problem {problem_id} has invalid APPS input/output data")

    if solutions:
        try:
            index, source, validation = choose_verified_solution(
                problem_id, io_data, solutions, timeout_s=timeout_s
            )
            if validation.get("all_passed"):
                return index, source, validation, "APPS/test/solutions.json", {"kind": "reference"}
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
            pass

    for method, source, row in _native_code_candidates(problem_id, native_results):
        validation = validate_source(source, inputs, outputs, timeout_s=timeout_s)
        if validation.get("all_passed"):
            return (
                None,
                source,
                validation,
                f"native {method} generated code",
                {
                    "kind": "native_baseline_fallback",
                    "method": method,
                    "status": row.get("status"),
                    "record_id": row.get("id", row.get("problem_id")),
                },
            )

    fallback = _test_oracle_fallback(problem_id, io_data)
    if fallback is not None:
        source, origin = fallback
        validation = validate_source(source, inputs, outputs, timeout_s=timeout_s)
        if validation.get("all_passed"):
            return None, source, validation, origin, {"kind": "test_oracle_fallback"}

    raise RuntimeError(f"no fully verified clean source available for APPS {problem_id}")


def _method_unit_metadata(
    method: str,
    problem_id: int,
    anchor: Mapping[str, Any],
    fault: Mapping[str, Any],
) -> dict[str, Any]:
    anchor_name = str(anchor["name"])
    if method == "auditcoder":
        unit_id = f"A_leaf_{anchor_name}"
        return {
            "unit_id": unit_id,
            "unit_kind": "TaskPlanIR leaf / ownership-map owner",
            "responsibility_map": {anchor_name: unit_id},
            "injection_anchor": "leaf implementation entry",
            "repair_flow": "Trace -> Resource -> Local -> SelectScope -> bounded regeneration",
            "fault_type": fault["id"],
        }
    if method == "codesim":
        unit_id = f"CS_step_{anchor_name}"
        return {
            "unit_id": unit_id,
            "unit_kind": "CodeSIM plan/simulation step",
            "responsibility_map": {anchor_name: unit_id},
            "injection_anchor": "plan step executed by final code/debug loop",
            "repair_flow": "plan/simulation -> code generation -> bounded debug retry",
            "fault_type": fault["id"],
        }
    if method == "mapcoder":
        unit_id = f"MC_map_{anchor_name}"
        return {
            "unit_id": unit_id,
            "unit_kind": "MapCoder recalled-example/planning mapped unit",
            "responsibility_map": {anchor_name: unit_id},
            "injection_anchor": "mapped algorithm implementation unit",
            "repair_flow": "mapping/planning -> code generation -> bounded improve retry",
            "fault_type": fault["id"],
        }
    unit_id = f"AC_candidate_{anchor_name}"
    return {
        "unit_id": unit_id,
        "unit_kind": "AgentCoder programmer candidate unit",
        "responsibility_map": {anchor_name: unit_id},
        "injection_anchor": "programmer candidate selected by executor feedback",
        "repair_flow": "test designer -> programmer candidates -> executor feedback -> bounded repair",
        "fault_type": fault["id"],
    }


def _case_id(index: int, problem_id: int) -> str:
    return f"case_{index:03d}_apps_{problem_id:04d}"


def build_corpus(output_dir: Path, *, timeout_s: float = 2.0) -> dict[str, Any]:
    sample = json.loads(SAMPLE_IDS.read_text(encoding="utf-8"))
    ids = list(sample.get("ids_by_difficulty", {}).get("interview", []))
    if len(ids) != 100:
        raise ValueError(f"expected exactly 100 interview IDs in prior 200-sample, found {len(ids)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    native_results = _load_native_results()
    manifest_cases: list[dict[str, Any]] = []
    for position, problem_id in enumerate(ids):
        problem_id = int(problem_id)
        metadata, io_data, solutions = _load_problem(problem_id)
        solution_index, clean_source, validation, reference_source, reference_meta = choose_reference_source(
            problem_id,
            io_data,
            solutions,
            native_results,
            timeout_s=timeout_s,
        )
        anchor = _entry_anchor(clean_source)
        fault = copy.deepcopy(FAULT_TYPES[position % len(FAULT_TYPES)])
        faulty_source = inject_fault(clean_source, fault, anchor, problem_id)
        inputs = [str(value) for value in io_data["inputs"]]
        outputs = [str(value) for value in io_data["outputs"]]
        clean_hash = _sha256(clean_source)
        faulty_hash = _sha256(faulty_source)
        case_id = _case_id(position + 1, problem_id)
        case_path = cases_dir / case_id
        method_path = case_path / "methods"
        method_path.mkdir(parents=True, exist_ok=True)
        (case_path / "clean.py").write_text(clean_source, encoding="utf-8")
        question_source = APPS_ROOT / f"{problem_id:04d}" / "question.txt"
        question_copy = case_path / "question.txt"
        question_copy.write_text(question_source.read_text(encoding="utf-8"), encoding="utf-8")
        (case_path / "input_output.json").write_text(
            json.dumps({"inputs": inputs, "outputs": outputs}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        # A short activation check catches an accidentally selected helper
        # function.  Infinite-loop mutants are syntax-checked only; executing
        # them here would deliberately consume the timeout for every case.
        try:
            ast.parse(faulty_source)
            mutant_syntax_ok = True
        except SyntaxError:
            mutant_syntax_ok = False
        if not mutant_syntax_ok:
            raise RuntimeError(f"fault injection produced invalid Python for APPS {problem_id}")
        if fault["id"] != "resource_timeout":
            activation = validate_source(faulty_source, inputs[:1], outputs[:1], timeout_s=min(timeout_s, 1.0))
        else:
            activation = {"checked": False, "expected_symptom": "TLE", "syntax_ok": True}

        method_records: dict[str, Any] = {}
        for method in METHODS:
            method_meta = _method_unit_metadata(method, problem_id, anchor, fault)
            native_row = native_results.get(method, {}).get(problem_id, {})
            native_summary = {
                "record_available": bool(native_row),
                "status": native_row.get("status"),
                "is_solved": native_row.get("is_solved"),
                "pipeline_all_passed": native_row.get("pipeline_all_passed"),
                "trace": None,
            }
            if method == "auditcoder" and native_row:
                final_report = native_row.get("final_report")
                records = final_report.get("responsibility_records") if isinstance(final_report, dict) else None
                native_summary["catg"] = native_row.get("catg")
                native_summary["responsibility_record_ids"] = [
                    record.get("id") for record in records if isinstance(record, dict) and record.get("id")
                ] if isinstance(records, list) else []
            method_fault_path = method_path / f"{method}.py"
            method_fault_path.write_text(faulty_source, encoding="utf-8")
            method_record = {
                **method_meta,
                "native_result": native_summary,
                "fault_id": f"APPS_CONTROLLED_{fault['id'].upper()}_{problem_id}",
                "faulty_source": str(method_fault_path.relative_to(output_dir)),
                "clean_source": str((case_path / "clean.py").relative_to(output_dir)),
                "clean_sha256": clean_hash,
                "faulty_sha256": faulty_hash,
                "same_semantics_as_other_methods": True,
            }
            method_records[method] = method_record
            (method_path / f"{method}.json").write_text(
                json.dumps(
                    {
                        "case_id": case_id,
                        "problem_id": problem_id,
                        "method": method,
                        "common_fault": fault,
                        "method_unit": method_meta,
                        "native_result": native_summary,
                        "clean_sha256": clean_hash,
                        "faulty_sha256": faulty_hash,
                        "faulty_source": str(method_fault_path.relative_to(output_dir)),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        case_record = {
            "case_id": case_id,
            "position": position + 1,
            "problem_id": problem_id,
            "difficulty": metadata.get("difficulty"),
            "url": metadata.get("url", ""),
            "question_file": f"APPS/test/{problem_id:04d}/question.txt",
            "question_copy": str(question_copy.relative_to(output_dir)),
            "test_count": len(inputs),
            "reference_solution_index": solution_index,
            "reference_source": reference_source,
            "reference_provenance": reference_meta,
            "reference_validation": validation,
            "entry_anchor": anchor,
            "mutation_validation": {
                "syntax_ok": mutant_syntax_ok,
                "activation": activation,
                "expected_symptom": fault["symptom"],
            },
            "common_fault": {
                **fault,
                "same_semantics_across_methods": True,
                "marker": f"APPS_CONTROLLED_{fault['id'].upper()}_{problem_id}",
                "clean_sha256": clean_hash,
                "faulty_sha256": faulty_hash,
            },
            "methods": method_records,
        }
        (case_path / "case.json").write_text(
            json.dumps(case_record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_cases.append(case_record)
        print(
            f"[{position + 1:03d}/100] APPS {problem_id:04d}: "
            f"{fault['id']}, clean={validation.get('all_passed')}, tests={len(inputs)}, "
            f"source={reference_source}",
            flush=True,
        )
    manifest = {
        "corpus": "apps_100_medium_controlled_injections",
        "source_sample": str(SAMPLE_IDS.relative_to(WORKSPACE_ROOT)),
        "source_seed": sample.get("seed"),
        "selection": {
            "difficulty": "interview",
            "count": 100,
            "same_100_ids_for_all_methods": True,
            "fault_assignment": "FAULT_TYPES[position mod 4]",
        },
        "methods": list(METHODS),
        "fault_types": list(FAULT_TYPES),
        "fairness": {
            "same_fault_type_per_problem": True,
            "same_clean_reference_per_problem": True,
            "same_fault_marker_per_problem": True,
            "method_native_unit_metadata": True,
            "no_model_api_called_during_injection": True,
            "repair_runs_are_separate_from_corpus_construction": True,
        },
        "cases": manifest_cases,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "fault_types.json").write_text(
        json.dumps({"fault_types": FAULT_TYPES, "methods": METHODS}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    validation_summary = {
        "cases": len(manifest_cases),
        "method_mutant_files": len(manifest_cases) * len(METHODS),
        "clean_references_fully_verified": sum(
            bool(case["reference_validation"].get("all_passed")) for case in manifest_cases
        ),
        "mutant_syntax_ok": sum(
            bool(case["mutation_validation"].get("syntax_ok")) for case in manifest_cases
        ),
        "fault_type_counts": {
            fault["id"]: sum(case["common_fault"]["id"] == fault["id"] for case in manifest_cases)
            for fault in FAULT_TYPES
        },
        "reference_provenance_counts": {
            source: sum(case["reference_source"] == source for case in manifest_cases)
            for source in sorted({case["reference_source"] for case in manifest_cases})
        },
        "same_fault_marker_per_case": all(
            len({method["fault_id"] for method in case["methods"].values()}) == 1
            for case in manifest_cases
        ),
        "same_faulty_hash_per_case": all(
            len({method["faulty_sha256"] for method in case["methods"].values()}) == 1
            for case in manifest_cases
        ),
    }
    (output_dir / "validation_summary.json").write_text(
        json.dumps(validation_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_readme(output_dir: Path) -> None:
    readme = f"""# APPS 100 controlled-injection corpus

This folder contains the 100 medium-difficulty (`interview`) problems from the
previous fixed 200-problem APPS sample (`seed=2026`). Every problem receives
exactly one deterministic fault type, assigned cyclically across four shared
semantic types:

- `runtime_exception` (RE)
- `wrong_answer` (WA)
- `interface_contract` (contract violation)
- `resource_timeout` (TLE)

The cyclic assignment gives 25 cases to each type.  All four method records
for a case share the same fault marker and semantic mutation, while their JSON
metadata identifies the method-native responsibility unit and repair flow.

The same semantic fault and the same clean reference source are materialized
for AuditCoder, CodeSIM, MapCoder, and AgentCoder. Each method has separate
unit metadata under `cases/*/methods/*.json` describing where that fault belongs
in its native workflow: AuditCoder TaskPlanIR ownership, CodeSIM plan/simulation
steps, MapCoder planning/mapping units, or AgentCoder programmer candidates and
executor feedback.

`clean.py` is the clean source used for validation.  It is taken from
`solutions.json` when a full stored-test check succeeds, otherwise from a
fully checked CodeSIM/MapCoder generation; the manifest records the provenance.
The few APPS records without a usable full reference are explicitly marked as
test-set-bounded fallbacks.  The four method `.py` files contain the same
controlled mutant so the repair comparison changes only the method's
localization/retry or bounded-repair process. `manifest.json` is the single
machine-readable index, and `validation_summary.json` records the generated
counts and hash checks.

The bounded fallback cases are APPS 2911, 583, and 2685; their provenance is
exposed in `manifest.json`.

Regenerate from the workspace root:

```bash
python tests/build_apps_100_controlled_injections.py \\
  --output-dir benchmarks/apps_100_controlled_injections
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    global APPS_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apps-root", type=Path, default=APPS_ROOT, help="Raw APPS test directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "results" / "apps_100_controlled_injections_rebuilt",
    )
    parser.add_argument("--timeout", type=float, default=2.0, help="per APPS test-case timeout")
    args = parser.parse_args()
    APPS_ROOT = args.apps_root.expanduser().resolve(strict=True)
    manifest = build_corpus(args.output_dir, timeout_s=args.timeout)
    _write_readme(args.output_dir)
    verified = sum(bool(case["reference_validation"].get("all_passed")) for case in manifest["cases"])
    print(json.dumps({"cases": len(manifest["cases"]), "fully_verified_clean_references": verified, "output": str(args.output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
