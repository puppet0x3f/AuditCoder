#!/usr/bin/env python3
"""Run ClassEval with the local AuditCoder implementation.

This adapter keeps the class-level benchmark harness separate from the older
``AuditCoder/auditcoder_class`` checkout.  It uses the NAACL implementation's
CATG ownership records, deterministic evidence locator, scope selector, rank
and repair event.  ClassEval's unittest suites are treated as trusted
benchmark checks.  At most one evidence-guided method repair is attempted for
each class; there is no second whole-problem retry.

Examples::

    # Validate the downloaded dataset without an API call
    python tests/run_classeval_auditcoder.py \
        --dry-run

    # One-task smoke run (requires the configured provider API key)
    python tests/run_classeval_auditcoder.py \
        --ids ClassEval_0 --output results/classeval_smoke.jsonl

    # Full 100-class run; the repair budget is fixed at one
    python tests/run_classeval_auditcoder.py \
        --count 100 --output results/classeval_auditcoder_one_repair.jsonl

    # Rerun only failed classes with an analysis pass and the same one repair
    python tests/run_classeval_auditcoder.py \
        --ids ClassEval_0,ClassEval_15 --reasoning --resume
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import re
import sys
import textwrap
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


HERE = Path(__file__).resolve()
IMPLEMENTATION_ROOT = HERE.parents[1]
WORKSPACE_ROOT = IMPLEMENTATION_ROOT
DEFAULT_DATA = WORKSPACE_ROOT / "data" / "ClassEval_data.json"
DEFAULT_OUTPUT = IMPLEMENTATION_ROOT / "results" / "classeval_auditcoder_one_repair.jsonl"
MAX_REPAIR_ATTEMPTS = 1

# Ensure every import below comes from this repository.
sys.path.insert(0, str(IMPLEMENTATION_ROOT))

from dotenv import load_dotenv  # noqa: E402

from auditcoder.common.llm import get_llm, invoke_llm  # noqa: E402
from auditcoder.common.utils import extract_json  # noqa: E402
from auditcoder.final_verifier_agent.audit import (  # noqa: E402
    frozen_complement_ok,
    rank_execution,
    repair_event,
)
from auditcoder.final_verifier_agent.nodes.error_locator import error_locator  # noqa: E402
from auditcoder.final_verifier_agent.nodes.final_report_assembler import (  # noqa: E402
    final_report_assembler,
)
from auditcoder.final_verifier_agent.nodes.scope_selector import scope_selector  # noqa: E402
from auditcoder.schemas.contract_graph import (  # noqa: E402
    catg_summary,
    ensure_catg,
    register_code_owner,
    register_assembly_units,
)
from auditcoder.verifier_agent.sandbox import SandboxExecutor  # noqa: E402


_FENCED_CODE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def load_classeval(path: Path) -> list[dict[str, Any]]:
    """Load and validate the official ClassEval JSON file."""

    if not path.is_file():
        raise FileNotFoundError(f"ClassEval data file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"ClassEval data must be a JSON list: {path}")
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict) or not item.get("task_id"):
            raise ValueError(f"ClassEval row {index} has no task_id")
        task_id = str(item["task_id"])
        if task_id in seen:
            raise ValueError(f"duplicate ClassEval task_id: {task_id}")
        seen.add(task_id)
        rows.append(item)
    return rows


def _task_sort_key(item: Mapping[str, Any]) -> tuple[int, str]:
    task_id = str(item.get("task_id", ""))
    match = re.search(r"(\d+)$", task_id)
    return (int(match.group(1)) if match else -1, task_id)


def select_entries(
    entries: list[dict[str, Any]],
    *,
    start: int = 0,
    count: int | None = None,
    end: int | None = None,
    ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    ordered = sorted(entries, key=_task_sort_key)
    if ids is not None:
        by_id = {str(item["task_id"]): item for item in ordered}
        return [by_id[item] for item in ids if item in by_id]
    selected = ordered[start : end if end is not None else None]
    return selected if count is None else selected[:count]


def _canonical_method_name(class_name: str, method_name: str) -> str:
    """Undo Python's source-level name mangling in test call expressions."""

    prefix = f"_{class_name.lstrip('_')}"
    if method_name.startswith(prefix) and "__" in method_name[len(prefix) :]:
        suffix = method_name[len(prefix) :]
        if suffix.startswith("__"):
            return suffix
    return method_name


def _class_test_method_calls(
    entry: Mapping[str, Any],
) -> dict[str, list[tuple[str, str]]]:
    """Find class-under-test method calls in each unittest class.

    A few ClassEval records exercise helper methods only from an integration
    test (for example ``set_cookies`` and ``matches_requirements``), so those
    methods are absent from ``methods_info``.  The test AST gives us a small,
    deterministic supplement without using the hidden solution code.
    """

    source = str(entry.get("test", ""))
    class_name = str(entry.get("class_name", ""))
    if not source or not class_name:
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    instance_names: set[str] = set()
    instance_attrs: set[str] = set()
    for node in ast.walk(tree):
        value: ast.AST | None = None
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            value = node.value
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            targets = [node.target]
        if not isinstance(value, ast.Call):
            continue
        if isinstance(value.func, ast.Name) and value.func.id == class_name:
            for target in targets:
                if isinstance(target, ast.Name):
                    instance_names.add(target.id)
                elif (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    instance_attrs.add(target.attr)

    result: dict[str, list[tuple[str, str]]] = {}
    for test_class in (node for node in tree.body if isinstance(node, ast.ClassDef)):
        calls: list[tuple[int, str]] = []
        for node in ast.walk(test_class):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            receiver = node.func.value
            is_instance_call = (
                isinstance(receiver, ast.Name) and receiver.id in instance_names
            ) or (
                isinstance(receiver, ast.Attribute)
                and isinstance(receiver.value, ast.Name)
                and receiver.value.id == "self"
                and receiver.attr in instance_attrs
            ) or (
                isinstance(receiver, ast.Call)
                and isinstance(receiver.func, ast.Name)
                and receiver.func.id == class_name
            )
            is_class_call = isinstance(receiver, ast.Name) and receiver.id == class_name
            if not (is_instance_call or is_class_call):
                continue
            method_name = _canonical_method_name(class_name, str(node.func.attr))
            if method_name.startswith("assert") or method_name in {
                "setUp",
                "tearDown",
            }:
                continue
            calls.append((int(getattr(node, "lineno", 0)), method_name))
        if calls:
            # ast.walk is not source ordered; use the recorded line numbers.
            calls.sort()
            result[test_class.name] = [
                (method_name, test_class.name) for _, method_name in calls
            ]
    return result


def _method_info(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return declared methods plus helper methods exercised by the tests."""

    declared = [
        dict(item)
        for item in (entry.get("methods_info") or [])
        if isinstance(item, Mapping) and item.get("method_name")
    ]
    known = {str(item["method_name"]) for item in declared}
    calls = _class_test_method_calls(entry)
    helper_classes: dict[str, str] = {}
    for test_class, methods in calls.items():
        for method_name, _ in methods:
            if method_name not in known and method_name not in helper_classes:
                helper_classes[method_name] = test_class
    for method_name, test_class in helper_classes.items():
        declared.append(
            {
                "method_name": method_name,
                "method_description": (
                    f"def {method_name}(self, *args, **kwargs):\n"
                    "    Helper method observed in the trusted ClassEval integration tests. "
                    "Infer its exact behavior from those tests and the calling public methods."
                ),
                "test_class": test_class,
                "test_code": "",
                "dependencies": {"method_dependencies": []},
                "inferred_helper": True,
            }
        )
        known.add(method_name)
    return declared


def _method_parameters(description: str, method_name: str) -> list[dict[str, str]]:
    """Extract parameter names from ClassEval's method declaration."""

    match = re.search(
        rf"^\s*def\s+{re.escape(method_name)}\s*\((.*?)\)\s*:",
        description or "",
        flags=re.MULTILINE | re.DOTALL,
    )
    params_text = match.group(1) if match else "self"
    try:
        function = ast.parse(f"def {method_name}({params_text}):\n    pass").body[0]
    except SyntaxError:
        return [{"name": "self", "type": "Any", "description": "class instance"}]
    params: list[dict[str, str]] = []
    all_args = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
    for argument in all_args:
        params.append(
            {
                "name": argument.arg,
                "type": ast.unparse(argument.annotation) if argument.annotation else "Any",
                "description": "class instance" if argument.arg == "self" else "method argument",
            }
        )
    if function.args.vararg:
        params.append({"name": "*" + function.args.vararg.arg, "type": "Any", "description": "variadic arguments"})
    if function.args.kwarg:
        params.append({"name": "**" + function.args.kwarg.arg, "type": "Any", "description": "keyword arguments"})
    return params


def build_class_plan(entry: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    """Create a CATG-compatible class plan before code is generated.

    Each ClassEval method receives a stable responsibility ID.  The class
    constructor/skeleton remains an assembler concern, while each method is a
    code-owning leaf that the bounded repair can address independently.
    """

    methods = _method_info(entry)
    id_by_method = {
        str(method["method_name"]): f"S{index + 1}"
        for index, method in enumerate(methods)
    }
    method_index = {str(method["method_name"]): index for index, method in enumerate(methods)}
    method_by_id = {identifier: name for name, identifier in id_by_method.items()}
    root_id = "S0"
    leaf_ids = list(method_by_id)
    root = {
        "id": root_id,
        "goal": f"Assemble the complete {entry.get('class_name', '')} class.",
        "type": "orchestration",
        "is_leaf": False,
        "agent_type": None,
        "parent": None,
        "dependencies": leaf_ids,
        "inputs": [],
        "outputs": [{"name": "class_code", "type": "python class", "description": "complete class"}],
        "hints": ["Preserve the supplied class name, constructor, imports, method signatures, and shared state."],
        "complexity_budget": "within the ClassEval tests",
        "context": str(entry.get("class_description", "")),
    }
    nodes = [root]
    for method in methods:
        name = str(method["method_name"])
        identifier = id_by_method[name]
        dependencies = [
            id_by_method[dependency]
            for dependency in (method.get("dependencies") or {}).get("method_dependencies", [])
            # ClassEval contains a few mutually recursive helper pairs.  Keep
            # the declared relation in the method context, but retain only
            # the earlier-to-later edge in CATG so the repair graph remains a
            # valid DAG; the source-level class still preserves the calls.
            if (
                dependency in id_by_method
                and dependency != name
                and method_index.get(str(dependency), -1) < method_index.get(name, 0)
            )
        ]
        params = _method_parameters(str(method.get("method_description", "")), name)
        node = {
            "id": identifier,
            "goal": f"Implement method {name} for the {entry.get('class_name', '')} class.",
            "type": "class_method",
            "is_leaf": True,
            "agent_type": "class_method",
            "parent": root_id,
            "dependencies": dependencies,
            "inputs": [p for p in params if p["name"] != "self"],
            "outputs": [{"name": "result", "type": "Any", "description": "method return value"}],
            "interface": {
                "function_name": name,
                "params": params,
                "return_type": "Any",
                "return_description": "ClassEval method result",
                "preconditions": [],
                "postconditions": [],
                "invariants": [],
            },
            "hints": [
                str(method.get("method_description", "")).strip()[:8000],
                "Emit a complete method definition including self; preserve shared instance state and method dependencies.",
            ],
            "complexity_budget": "within the supplied tests",
            "context": (
                f"Class: {entry.get('class_name', '')}\n"
                f"Fields: {', '.join(map(str, entry.get('fields', []) or [])) or '(none)'}\n"
                f"Method dependencies: {', '.join(map(str, (method.get('dependencies') or {}).get('method_dependencies', []))) or '(none)'}"
            ),
        }
        nodes.append(node)

    problem_description = "\n".join(
        [
            str(entry.get("class_description", "")).strip(),
            f"Class name: {entry.get('class_name', '')}",
            "Methods:\n" + "\n".join(
                f"- {m.get('method_name')}: {str(m.get('method_description', '')).strip()}"
                for m in methods
            ),
        ]
    )
    task_plan_ir: dict[str, Any] = {
        "problem": {
            "title": str(entry.get("class_name", "")),
            "description": problem_description,
            "original_specification": problem_description,
            "input_format": "ClassEval unittest calls",
            "output_format": "A complete Python class",
            "starter_code": str(entry.get("skeleton", "")),
            "constraints": {"time_limit": "30s", "memory_limit": "256MB"},
            "sample_cases": [],
        },
        "analysis": {
            "problem_category": "class-level Python implementation",
            "key_observations": ["Methods share constructor state and may call one another."],
            "complexity_target": {"time": "within tests", "space": "within tests"},
            "algorithm_selection": [],
            "reasoning_chain": [],
        },
        "global_strategy": "Implement each owned method against the supplied class contract, then preserve the frozen methods during repair.",
        "complexity_budget": {"time": "within tests", "space": "within tests"},
        "task_nodes": nodes,
        "root_id": root_id,
        "execution_order": leaf_ids,
        "validation": {"sample_test_cases": [], "edge_cases": [], "invariants": []},
        "executor_registry": {"class_method": "agent_class_method"},
        "class_eval": {
            "class_name": str(entry.get("class_name", "")),
            "imports": list(entry.get("import_statement", []) or []),
            "test": str(entry.get("test", "")),
            "test_classes": list(entry.get("test_classes", []) or []),
            "test_to_method": {
                str(method.get("test_class")): str(method.get("method_name"))
                for method in methods
                if method.get("test_class") and method.get("method_name")
            },
        },
    }
    return ensure_catg(task_plan_ir), id_by_method, method_by_id


def _format_class_prompt(entry: Mapping[str, Any], ir: Mapping[str, Any]) -> str:
    methods = _method_info(entry)
    lines = [
        "ClassEval task. First reason through every method and the shared state; then emit only the requested JSON.",
        f"Class name: {entry.get('class_name', '')}",
        "Class description:",
        str(entry.get("class_description", "")).strip(),
        "",
        "Fields and constructor contract:",
        json.dumps(
            {
                "fields": list(entry.get("fields", []) or []),
                "constructor": str(entry.get("class_constructor", "")).strip(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "Required imports:",
        "\n".join(str(value) for value in (entry.get("import_statement") or [])) or "(none)",
        "",
        "Class skeleton and constructor:",
        "```python",
        str(entry.get("skeleton", "")).strip(),
        "```",
        "",
        "Methods and stable AuditCoder responsibility IDs:",
    ]
    for method in methods:
        name = str(method.get("method_name", ""))
        node_id = next(
            (str(node.get("id")) for node in ir.get("task_nodes", []) if (node.get("interface") or {}).get("function_name") == name),
            "",
        )
        lines.extend(
            [
                f"[{node_id}] {name}",
                str(method.get("method_description", "")).strip(),
                f"Dependencies: {json.dumps(method.get('dependencies', {}), ensure_ascii=False)}",
                "",
            ]
        )
        method_tests = str(method.get("test_code", "")).strip()
        if method_tests:
            lines.extend(
                [
                    f"Trusted tests focused on {name}:",
                    "```python",
                    method_tests[:12000],
                    "```",
                    "",
                ]
            )
    helper_names = [
        str(method.get("method_name"))
        for method in methods
        if method.get("inferred_helper")
    ]
    if helper_names:
        lines.extend(
            [
                "Additional helper methods called by the trusted tests (they are required even though the original methods_info omits them):",
                ", ".join(helper_names),
                "Infer their signatures and exact behavior from the test code and the public methods that call them.",
                "",
            ]
        )
    test_code = str(entry.get("test", ""))
    if test_code:
        lines.extend(["Trusted unittest suite:", "```python", test_code[:24000], "```"])
    return "\n".join(lines)


def _format_analysis_prompt(entry: Mapping[str, Any], ir: Mapping[str, Any]) -> str:
    """Compact context for the optional analysis call.

    The coding prompt intentionally repeats focused and full tests.  Repeating
    both in the analysis request can consume the provider's completion budget
    on long classes before it closes the JSON object.
    """

    methods = _method_info(entry)
    lines = [
        "ClassEval analysis task",
        f"Class name: {entry.get('class_name', '')}",
        "Class description:",
        str(entry.get("class_description", "")).strip(),
        "",
        "Fields:",
        json.dumps(list(entry.get("fields", []) or []), ensure_ascii=False),
        "Constructor:",
        str(entry.get("class_constructor", "")).strip(),
        "",
        "Required imports:",
        "\n".join(str(value) for value in (entry.get("import_statement") or [])) or "(none)",
        "",
        "Methods and focused test evidence:",
    ]
    for method in methods:
        lines.extend(
            [
                f"[{method.get('method_name')}]",
                str(method.get("method_description", "")).strip()[:4000],
                f"Dependencies: {json.dumps(method.get('dependencies', {}), ensure_ascii=False)}",
            ]
        )
        focused = str(method.get("test_code", "")).strip()
        if focused:
            lines.extend(["Focused tests:", focused[:4000]])
        lines.append("")
    full_test = str(entry.get("test", ""))
    if full_test:
        lines.extend(["Full integration test (prefix):", full_test[:12000]])
    return "\n".join(lines)


def _extract_python(value: Any) -> str:
    """Extract code from JSON fields or markdown-wrapped model output."""

    if isinstance(value, Mapping):
        for key in ("final_code", "class_code", "method_code", "code_snippet", "code"):
            if value.get(key):
                return _extract_python(value[key])
        value = value.get("raw", "")
    text = str(value or "").strip()
    match = _FENCED_CODE.search(text)
    if match:
        text = match.group(1).strip()
    return text


def _invoke_json(system_prompt: str, user_prompt: str, *, max_tokens: int) -> Any:
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm(temperature=0.2, json_mode=True, max_tokens=max_tokens)
    response = invoke_llm(
        llm,
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
    )
    try:
        return extract_json(response)
    except ValueError:
        return {"raw": getattr(response, "content", str(response))}


_CLASS_REASONER_SYSTEM = """You are the analysis stage of AuditCoder for a ClassEval Python class.
Return exactly one JSON object: {\"analysis\": \"...\"}. Do not output Python
code. In at most 1500 words, explain concisely but concretely the exact behavior required by every
method, shared-state invariants, helper methods called only by integration
tests, edge cases, and any numeric/library compatibility traps. Use the
trusted focused tests as the oracle and flag ambiguity instead of inventing a
new API. The next stage will use this analysis to write the complete class.
"""


def analyze_class(entry: Mapping[str, Any], ir: Mapping[str, Any]) -> str:
    parsed = _invoke_json(
        _CLASS_REASONER_SYSTEM,
        _format_analysis_prompt(entry, ir)
        + "\n\nReturn the implementation analysis now as JSON.",
        max_tokens=3000,
    )
    if isinstance(parsed, Mapping):
        return str(parsed.get("analysis") or parsed.get("raw") or "").strip()
    return str(parsed or "").strip()


_CLASS_GENERATOR_SYSTEM = """You are the class implementation generator inside AuditCoder.
Return exactly one JSON object: {\"final_code\": \"...\"}.
The value must be complete executable Python source containing the requested
class, its constructor, every required method, and any imports it needs.
Preserve every public method name/signature and shared instance state from the
skeleton. Do not include markdown or explanation outside the JSON object.
Use the trusted unittest suite as the behavioral specification, including the
integration tests and helper methods that may not appear in methods_info. The
stable responsibility IDs identify method ownership; implement methods
independently but make their dependencies and state updates consistent. Before
returning, mentally check every focused test and the final integration suite.
Do not add test classes, top-level execution, placeholder returns, or guessed
APIs. Keep all imports required by the supplied skeleton.
"""

_CLASS_RECOVERY_SYSTEM = """You are the syntax-recovery stage inside AuditCoder.
Return exactly one JSON object: {\"final_code\": \"...\"}. The previous
response was not valid Python. Reconstruct the complete requested class from
the supplied skeleton, contracts, tests, and malformed response. Preserve all
method signatures and behavior; return executable Python only inside the JSON
value. Do not add explanations, markdown, tests, or top-level execution.
"""


def generate_class_code(
    entry: Mapping[str, Any],
    ir: Mapping[str, Any],
    *,
    analysis: str = "",
) -> str:
    prompt = _format_class_prompt(entry, ir)
    if analysis:
        prompt += (
            "\n\nIndependent implementation analysis (check it against the tests before coding):\n"
            + analysis[:16000]
        )
    parsed = _invoke_json(
        _CLASS_GENERATOR_SYSTEM,
        prompt + "\n\nReturn the complete class implementation now as JSON.",
        max_tokens=12000,
    )
    code = _extract_python(parsed)
    if not code:
        raise ValueError("class generator returned no code")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        recovery_prompt = (
            prompt
            + "\n\nMalformed candidate (repair syntax while preserving its useful implementation):\n"
            + code[:18000]
            + f"\n\nParser error: {exc}\nReturn the complete corrected class now as JSON."
        )
        recovered = _invoke_json(
            _CLASS_RECOVERY_SYSTEM,
            recovery_prompt,
            max_tokens=12000,
        )
        code = _extract_python(recovered)
        try:
            tree = ast.parse(code)
        except SyntaxError as recovery_exc:
            raise ValueError(
                f"class generator returned invalid Python after syntax recovery: {recovery_exc}"
            ) from recovery_exc
    class_name = str(entry.get("class_name", ""))
    if not any(isinstance(node, ast.ClassDef) and node.name == class_name for node in tree.body):
        raise ValueError(f"generated code does not define class {class_name!r}")
    return code


def _method_sources(source: str, class_name: str) -> dict[str, str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    class_node = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name),
        None,
    )
    if class_node is None:
        return {}
    lines = source.splitlines()
    result: dict[str, str] = {}
    for node in class_node.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = min(
            [getattr(node, "lineno", 1)]
            + [getattr(dec, "lineno", getattr(node, "lineno", 1)) for dec in node.decorator_list]
        )
        end = getattr(node, "end_lineno", node.lineno)
        result[node.name] = textwrap.dedent("\n".join(lines[start - 1 : end])).strip()
    return result


def _extract_method_candidate(value: Any, method_name: str) -> str:
    source = _extract_python(value)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    lines = source.splitlines()
    for class_node in tree.body:
        if not isinstance(class_node, ast.ClassDef):
            continue
        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                node.name == method_name
                or (
                    method_name.startswith("__")
                    and node.name.endswith(method_name)
                    and node.name.startswith("_")
                )
            ):
                start = min([node.lineno] + [getattr(dec, "lineno", node.lineno) for dec in node.decorator_list])
                end = getattr(node, "end_lineno", node.lineno)
                result = textwrap.dedent("\n".join(lines[start - 1 : end])).strip()
                if node.name != method_name:
                    result = re.sub(
                        rf"(\b(?:async\s+)?def\s+){re.escape(node.name)}(\s*\()",
                        rf"\g<1>{method_name}\g<2>",
                        result,
                        count=1,
                    )
                return result
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name == method_name
            or (
                method_name.startswith("__")
                and node.name.endswith(method_name)
                and node.name.startswith("_")
            )
        ):
            start = min([node.lineno] + [getattr(dec, "lineno", node.lineno) for dec in node.decorator_list])
            end = getattr(node, "end_lineno", node.lineno)
            result = textwrap.dedent("\n".join(lines[start - 1 : end])).strip()
            if node.name != method_name:
                result = re.sub(
                    rf"(\b(?:async\s+)?def\s+){re.escape(node.name)}(\s*\()",
                    rf"\g<1>{method_name}\g<2>",
                    result,
                    count=1,
                )
            return result
    return ""


def _replace_method(source: str, class_name: str, method_name: str, replacement: str) -> str:
    candidate = _extract_method_candidate(replacement, method_name)
    if not candidate:
        raise ValueError(f"repair output does not contain method {method_name}")
    candidate_tree = ast.parse(candidate)
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name for node in candidate_tree.body):
        raise ValueError(f"repair output has no top-level method {method_name}")
    original_tree = ast.parse(source)
    class_node = next(
        (node for node in original_tree.body if isinstance(node, ast.ClassDef) and node.name == class_name),
        None,
    )
    if class_node is None:
        raise ValueError(f"class {class_name!r} missing from current code")
    target = next(
        (node for node in class_node.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name),
        None,
    )
    lines = source.splitlines()
    indented = textwrap.indent(textwrap.dedent(candidate).strip(), "    ")
    if target is None:
        # Test-only helpers are omitted from a number of ClassEval skeletons.
        # A bounded repair may add such a helper while leaving all existing
        # methods byte-for-byte frozen.
        insert_at = getattr(class_node, "end_lineno", len(lines))
        if insert_at > 0 and lines[insert_at - 1].strip() == "":
            insert_at -= 1
        insertion = [""] + indented.splitlines()
        lines[insert_at:insert_at] = insertion
    else:
        start = min([target.lineno] + [getattr(dec, "lineno", target.lineno) for dec in target.decorator_list])
        end = getattr(target, "end_lineno", target.lineno)
        lines[start - 1 : end] = indented.splitlines()
    result = "\n".join(lines)
    ast.parse(result)
    return result


_REPAIR_SYSTEM = """You are the bounded method repair agent inside AuditCoder.
Return exactly one JSON object: {\"method_code\": \"...\"}.
Return only a complete Python method definition for the named method,
including self and any local helper/import assumptions already available in the
class. Preserve the method signature and shared state. Repair the reported
failure using the trusted test evidence. Do not output a class, markdown, or
any other method; the caller will splice this method into the frozen class.
"""

_REPAIR_RECOVERY_SYSTEM = """You are the output-format recovery stage inside AuditCoder.
Return exactly one JSON object: {\"method_code\": \"...\"}. Emit one complete
Python method definition for the named target, preserving its exact signature
and decorator. The previous response had the wrong shape or method name; fix
that formatting and the reported behavior. Do not output a class, prose, or
markdown outside the JSON value.
"""


def repair_method(
    entry: Mapping[str, Any],
    ir: Mapping[str, Any],
    current_code: str,
    method_name: str,
    failure: Mapping[str, Any],
) -> str:
    node = next(
        (
            node
            for node in ir.get("task_nodes", [])
            if (node.get("interface") or {}).get("function_name") == method_name
        ),
        {},
    )
    user_prompt = "\n".join(
        [
            _format_class_prompt(entry, ir),
            "",
            f"Current complete class code:\n```python\n{current_code}\n```",
            "",
            f"Repair target responsibility: {node.get('id', '')}",
            f"Method: {method_name}",
            f"Failure evidence:\n{json.dumps(dict(failure), ensure_ascii=False, indent=2)[:10000]}",
            "Return only the corrected method definition as JSON.",
        ]
    )
    parsed = _invoke_json(_REPAIR_SYSTEM, user_prompt, max_tokens=7000)
    candidate = _extract_python(parsed)
    extracted = _extract_method_candidate(candidate, method_name) if candidate else ""
    if extracted:
        return extracted

    # A malformed JSON/code wrapper is a formatting failure within the same
    # single bounded repair, not a second repair attempt.  Recover it with a
    # compact schema-only call before recording the repair as rejected.
    recovery = _invoke_json(
        _REPAIR_RECOVERY_SYSTEM,
        user_prompt
        + "\n\nThe previous output did not contain a parseable target method. "
        + f"Emit exactly `def {method_name}` (with its decorator if required) now.",
        max_tokens=7000,
    )
    recovered = _extract_python(recovery)
    extracted = _extract_method_candidate(recovered, method_name) if recovered else ""
    if not extracted:
        raise ValueError(f"repair output does not contain method {method_name}")
    return extracted


def _test_script(entry: Mapping[str, Any], code: str, test_class: str | None = None) -> str:
    imports = "\n".join(str(item) for item in (entry.get("import_statement") or []))
    test_code = str(entry.get("test", ""))
    test_classes = _usable_test_classes(entry)
    class_list = ", ".join(repr(item) for item in test_classes)
    if test_class is None:
        suite_builder = f"""
    for cls_name in [{class_list}]:
        cls = globals().get(cls_name)
        if cls is None:
            raise RuntimeError(f"missing ClassEval test class: {{cls_name}}")
        suite.addTests(loader.loadTestsFromTestCase(cls))
"""
    else:
        suite_builder = f"""
    cls = globals().get({test_class!r})
    if cls is None:
        raise RuntimeError({f'missing ClassEval test class: {test_class}'!r})
    suite.addTests(loader.loadTestsFromTestCase(cls))
"""
    return "\n\n".join(
        part
        for part in [
            imports,
            code,
            "import unittest",
            test_code,
            f"""
if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
{textwrap.indent(textwrap.dedent(suite_builder).strip(), '    ')}
    if suite.countTestCases() == 0:
        raise RuntimeError('ClassEval selected no tests')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    import sys
    sys.exit(0 if result.wasSuccessful() else 1)
""",
        ]
        if part
    )


def _usable_test_classes(entry: Mapping[str, Any]) -> list[str]:
    """Drop fixture-only classes that contain no ``test_*`` method."""

    source = str(entry.get("test", ""))
    requested = [str(item).strip() for item in (entry.get("test_classes") or [])]
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return requested
    names_with_cases = {
        node.name.strip()
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name.startswith("test")
            for child in node.body
        )
    }
    return [name for name in requested if name in names_with_cases]


def _parse_test_count(stdout: str, stderr: str) -> int:
    match = re.search(r"Ran (\d+) test", f"{stdout}\n{stderr}")
    return int(match.group(1)) if match else 0


def _integration_failure_method(
    entry: Mapping[str, Any],
    test_class: str,
    stderr: str,
    known_methods: set[str],
) -> str:
    """Attribute an integration assertion to its preceding class call."""

    source = str(entry.get("test", ""))
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    failure_match = re.search(
        r"(?:FAIL|ERROR):\s+(\w+)\s+\(__main__\.[^)]+\)",
        stderr or "",
    )
    failure_name = failure_match.group(1) if failure_match else ""
    class_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name.strip() == test_class.strip()
        ),
        None,
    )
    if class_node is None:
        return ""
    function_node = next(
        (
            node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and (not failure_name or node.name == failure_name)
        ),
        None,
    )
    if function_node is None:
        return ""
    calls: list[tuple[int, str]] = []
    assertions: list[int] = []
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        name = str(node.func.attr)
        line = int(getattr(node, "lineno", 0))
        if name.startswith("assert"):
            assertions.append(line)
        elif name in known_methods:
            calls.append((line, name))
    if not calls:
        return ""
    calls.sort()
    assertions.sort()
    if assertions:
        before = [item for item in calls if item[0] < assertions[0]]
        if before:
            return before[-1][1]
    return calls[-1][1]


def run_class_tests(
    entry: Mapping[str, Any],
    code: str,
    *,
    method_to_id: Mapping[str, str],
    timeout: float,
    memory_limit_mb: int = 1024,
) -> dict[str, Any]:
    """Run the full ClassEval suite and each TestCase independently."""

    # BLAS-backed imports can exceed a 256 MB address-space cap before tests
    # start.  Keep the evaluator bounded while leaving room for declared
    # scientific/document dependencies.
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, "1")
    executor = SandboxExecutor(
        default_timeout_s=timeout,
        default_memory_limit_mb=memory_limit_mb,
    )
    full = executor.run(
        _test_script(entry, code),
        timeout_s=timeout,
        memory_limit_mb=memory_limit_mb,
    )
    test_to_method = {
        str(method.get("test_class")).strip(): str(method.get("method_name"))
        for method in _method_info(entry)
        if method.get("test_class") and method.get("method_name")
    }
    # Prefer the actual call pattern when ClassEval's metadata accidentally
    # associates a method with a fixture class (PDFHandler is one example).
    for test_class, calls in _class_test_method_calls(entry).items():
        names = [name for name, _ in calls if name in method_to_id]
        unique_names = list(dict.fromkeys(names))
        if len(unique_names) == 1:
            test_to_method[test_class.strip()] = unique_names[0]
    known_methods = set(method_to_id) | {
        str(method.get("method_name")) for method in _method_info(entry)
    }
    results: list[dict[str, Any]] = []
    for index, test_class in enumerate(_usable_test_classes(entry)):
        test_class = str(test_class).strip()
        per = executor.run(
            _test_script(entry, code, test_class),
            timeout_s=timeout,
            memory_limit_mb=memory_limit_mb,
        )
        method_name = test_to_method.get(test_class, "")
        if not method_name and (per.exit_code != 0 or per.timed_out):
            method_name = _integration_failure_method(
                entry,
                test_class,
                per.stderr or "",
                known_methods,
            )
        owner = method_to_id.get(method_name, "")
        passed = per.exit_code == 0 and not per.timed_out
        stderr = per.stderr or ""
        # A unittest assertion often has no frame in generated code.  The
        # ClassEval test-class/method relation is deterministic benchmark
        # evidence, so expose it as a generated frame for AuditCoder's Trace
        # locator without guessing from the failure text.
        if not passed and owner and method_name:
            stderr += f'\n  File "<classeval-generated>", line 1, in {method_name}\n'
        results.append(
            {
                "case_index": index,
                "test_class": test_class,
                "method_name": method_name,
                "owner": owner,
                "owners": [owner] if owner else [],
                "input": test_class,
                "expected": "PASS",
                "actual": "PASS" if passed else "FAIL",
                "passed": passed,
                "timed_out": per.timed_out,
                "exit_code": per.exit_code,
                "stdout": (per.stdout or "")[:12000],
                "stderr": stderr[:12000],
                "time_ms": per.time_ms,
                "symptom": "Pass" if passed else ("TLE" if per.timed_out else "RE" if per.exit_code not in (0, None) and "FAIL" not in stderr else "WA"),
                "test_source": "ClassEval.unittest",
                "trusted": True,
            }
        )

    full_test_count = _parse_test_count(full.stdout, full.stderr)
    all_passed = (
        bool(results)
        and full_test_count > 0
        and full.exit_code == 0
        and not full.timed_out
        and all(bool(item["passed"]) for item in results)
    )
    method_pass_rates: dict[str, bool] = {}
    grouped: dict[str, list[bool]] = {}
    for item in results:
        method_name = str(item.get("method_name") or "")
        if method_name:
            grouped.setdefault(method_name, []).append(bool(item["passed"]))
    for method in _method_info(entry):
        name = str(method.get("method_name"))
        method_pass_rates[name] = bool(grouped.get(name)) and all(grouped[name])
    trusted_failures = [
        {
            "case_index": item["case_index"],
            "test_class": item["test_class"],
            "input": item["input"],
            "expected": item["expected"],
            "actual": item["actual"],
            "owner": item.get("owner", ""),
        }
        for item in results
        if not item["passed"]
    ]
    error_type = (
        "none"
        if all_passed
        else "timeout"
        if full.timed_out or any(item["timed_out"] for item in results)
        else "wrong_answer"
    )
    return {
        "all_passed": all_passed,
        "test_count": full_test_count,
        "pass_count": sum(method_pass_rates.values()),
        "method_pass_rates": method_pass_rates,
        "results": results,
        "trusted_test_count": len(results),
        "trusted_failures": trusted_failures,
        "advisory_failures": [],
        "error_type": error_type,
        "stdout": full.stdout,
        "stderr": full.stderr,
        "exit_code": full.exit_code,
        "timed_out": full.timed_out,
        "time_ms": full.time_ms,
        "test_source": "ClassEval.unittest",
    }


def _verification_reports(
    execution: Mapping[str, Any],
    *,
    ir: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for item in execution.get("results", []) or []:
        owner = str(item.get("owner") or "")
        if not owner:
            continue
        reports[owner] = {
            "verdict": "accept" if item.get("passed") else "reject",
            "summary": f"ClassEval {item.get('test_class', '')}: {'passed' if item.get('passed') else 'failed'}",
            "agent_type": "class_method",
            "owners": [owner],
            "obligation": {"class": "semantic", "name": "ClassEval method behavior"},
            "witness": {
                "case_index": item.get("case_index"),
                "input": item.get("input", ""),
                "expected": item.get("expected", "PASS"),
                "actual": item.get("actual", "FAIL"),
                "applicability": "observed",
                "observed": True,
            },
            "version": {owner: 0},
        }
    return reports


def _evidence_records(execution: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in execution.get("results", []) or []:
        owner = str(item.get("owner") or "")
        if not owner:
            continue
        records.append(
            {
                "id": f"classeval:{item.get('case_index', 0)}:{owner}",
                "owners": [owner],
                "version": {owner: 0},
                "verdict": "Pass" if item.get("passed") else "Reject",
                "obligation": {"class": "semantic", "name": "ClassEval method behavior"},
                "witness": {
                    "case_index": item.get("case_index"),
                    "input": item.get("input", ""),
                    "expected": item.get("expected", "PASS"),
                    "actual": item.get("actual", "FAIL"),
                    "applicability": "observed",
                    "observed": True,
                },
                "kind": "exec",
                "source": "ClassEval.unittest",
                "current": True,
            }
        )
    return records


def _register_runtime_state(
    ir: dict[str, Any],
    code: str,
    snippets: Mapping[str, str],
    reports: Mapping[str, Mapping[str, Any]],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    result = copy.deepcopy(ir)
    for task_id, snippet in snippets.items():
        result = register_code_owner(
            result,
            str(task_id),
            str(snippet),
            verification_report=reports.get(str(task_id)),
        )
    result = register_assembly_units(result, code)
    result["evidence_records"] = copy.deepcopy(evidence)
    return result


def _append_repair_event(ir: dict[str, Any], event: Mapping[str, Any], region: Iterable[str]) -> dict[str, Any]:
    result = copy.deepcopy(ir)
    region_ids = {str(identifier) for identifier in region}
    for record in result.get("responsibility_records", []) or []:
        if str(record.get("id")) in region_ids:
            record.setdefault("repair_history", []).append(copy.deepcopy(dict(event)))
    for node in result.get("task_nodes", []) or []:
        if str(node.get("id")) in region_ids:
            node.setdefault("repair_history", []).append(copy.deepcopy(dict(event)))
    return ensure_catg(result, ownership_map=result.get("ownership_map"))


def _bump_versions(ir: dict[str, Any], region: Iterable[str]) -> tuple[dict[str, Any], dict[str, int]]:
    """Advance CATG versions for the owners changed by an accepted repair."""

    result = copy.deepcopy(ir)
    region_ids = {str(identifier) for identifier in region}
    versions: dict[str, int] = {}
    for record in result.get("responsibility_records", []) or []:
        identifier = str(record.get("id", ""))
        if identifier in region_ids:
            version = int(record.get("version", 0) or 0) + 1
            record["version"] = version
            versions[identifier] = version
    for node in result.get("task_nodes", []) or []:
        identifier = str(node.get("id", ""))
        if identifier in versions:
            node["version"] = versions[identifier]
    return ensure_catg(result, ownership_map=result.get("ownership_map")), versions


def _set_evidence_versions(
    evidence: list[dict[str, Any]],
    versions: Mapping[str, int],
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for item in evidence:
        record = copy.deepcopy(item)
        owners = record.get("owners") or []
        previous = record.get("version") if isinstance(record.get("version"), Mapping) else {}
        record["version"] = {
            str(owner): int(versions.get(str(owner), previous.get(str(owner), 0) or 0))
            for owner in owners
        }
        updated.append(record)
    return updated


def _state(
    *,
    code: str,
    ir: dict[str, Any],
    snippets: Mapping[str, str],
    reports: Mapping[str, Mapping[str, Any]],
    execution: Mapping[str, Any],
    evidence: list[dict[str, Any]],
    max_repair_attempts: int,
    audit_versions: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    graph = ir.get("contract_annotated_task_graph", ir.get("contract_graph", {}))
    return {
        "problem_id": None,
        "final_code": code,
        "task_plan_ir": ir,
        "code_snippets": dict(snippets),
        "verification_reports": dict(reports),
        "execution_result": dict(execution),
        "initial_execution_result": dict(execution),
        "external_test_cases": [],
        "max_repair_attempts": max_repair_attempts,
        "repair_attempts": 0,
        "repair_history": [],
        "audit_versions": dict(audit_versions or {}),
        "ownership_map": ir.get("ownership_map", {}),
        "registered_units": ir.get("ownership_map", {"units": []}),
        "contract_graph": graph,
        "contract_annotated_task_graph": graph,
        "responsibility_records": ir.get("responsibility_records", []),
        "development_checks": ir.get("development_checks", []),
        "evidence_records": evidence,
        "audit_trace": {"G": graph, "S": ir.get("responsibility_records", []), "E": evidence, "H": []},
        "retried_tasks": [],
    }


def run_one(
    entry: dict[str, Any],
    *,
    timeout: float,
    memory_limit_mb: int,
    reasoning: bool,
    quiet: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    task_id = str(entry["task_id"])
    result: dict[str, Any] = {
        "problem_id": task_id,
        "task_id": task_id,
        "class_name": entry.get("class_name", ""),
        "status": "error",
        "max_repair_attempts": MAX_REPAIR_ATTEMPTS,
    }
    try:
        ir, id_by_method, method_by_id = build_class_plan(entry)
        generation_analysis = analyze_class(entry, ir) if reasoning else ""
        initial_code = generate_class_code(entry, ir, analysis=generation_analysis)
        initial_snippets_by_name = _method_sources(initial_code, str(entry.get("class_name", "")))
        initial_snippets = {
            id_by_method[name]: source
            for name, source in initial_snippets_by_name.items()
            if name in id_by_method
        }
        initial_tests = run_class_tests(
            entry,
            initial_code,
            method_to_id=id_by_method,
            timeout=timeout,
            memory_limit_mb=memory_limit_mb,
        )
        initial_reports = _verification_reports(initial_tests, ir=ir)
        initial_evidence = _evidence_records(initial_tests)
        ir = _register_runtime_state(ir, initial_code, initial_snippets, initial_reports, initial_evidence)
        state = _state(
            code=initial_code,
            ir=ir,
            snippets=initial_snippets,
            reports=initial_reports,
            execution=initial_tests,
            evidence=initial_evidence,
            max_repair_attempts=MAX_REPAIR_ATTEMPTS,
        )
        state["problem_id"] = task_id

        location_result = error_locator(state)
        state.update(location_result)
        location = location_result.get("error_location")
        scope_result: dict[str, Any] = {}
        if location:
            scope_result = scope_selector({**state, "error_location": location})
            state.update(scope_result)

        final_code = initial_code
        final_tests = initial_tests
        final_reports = initial_reports
        final_evidence = initial_evidence
        final_ir = ir
        final_versions: dict[str, int] = {}
        history: list[dict[str, Any]] = []
        repair_attempts = 0
        repair_decision = "NotAttempted"
        rollback_status = "NotApplicable"
        repaired_owner = ""

        scope = scope_result.get("repair_scope") or {}
        if location and scope and repair_attempts < MAX_REPAIR_ATTEMPTS:
            region = [str(identifier) for identifier in scope.get("region", [])]
            repaired_owner = region[0] if len(region) == 1 else ""
            method_name = method_by_id.get(repaired_owner, "")
            if method_name:
                repair_attempts = 1
                failure = (location.get("failing_case") or location.get("anchor") or {})
                try:
                    candidate_method = repair_method(
                        entry,
                        ir,
                        initial_code,
                        method_name,
                        failure,
                    )
                    candidate_code = _replace_method(
                        initial_code,
                        str(entry.get("class_name", "")),
                        method_name,
                        candidate_method,
                    )
                    candidate_snippets = dict(initial_snippets)
                    candidate_snippets[repaired_owner] = _method_sources(
                        candidate_code,
                        str(entry.get("class_name", "")),
                    ).get(method_name, candidate_method)
                    boundary_ok, frozen_violations = frozen_complement_ok(
                        initial_snippets,
                        candidate_snippets,
                        region,
                    )
                    candidate_tests = run_class_tests(
                        entry,
                        candidate_code,
                        method_to_id=id_by_method,
                        timeout=timeout,
                        memory_limit_mb=memory_limit_mb,
                    )
                    candidate_reports = _verification_reports(candidate_tests, ir=ir)
                    candidate_evidence = _evidence_records(candidate_tests)
                    candidate_ir = _register_runtime_state(
                        ir,
                        candidate_code,
                        candidate_snippets,
                        candidate_reports,
                        candidate_evidence,
                    )
                    candidate_state = _state(
                        code=candidate_code,
                        ir=candidate_ir,
                        snippets=candidate_snippets,
                        reports=candidate_reports,
                        execution=candidate_tests,
                        evidence=candidate_evidence,
                        max_repair_attempts=MAX_REPAIR_ATTEMPTS,
                    )
                    accepted = boundary_ok and rank_execution(
                        candidate_tests,
                        candidate_reports,
                    ) > rank_execution(initial_tests, initial_reports)
                    event = repair_event(
                        location=location,
                        scope=scope,
                        before_state=state,
                        candidate_state=candidate_state,
                        candidate_execution=candidate_tests,
                        decision="Accept" if accepted else "Reject",
                        rollback_status="NotApplicable" if accepted else "Restored",
                        attempt=repair_attempts,
                        reason=(
                            "strict_rank_improvement"
                            if accepted and not frozen_violations
                            else "frozen_complement_violation"
                            if frozen_violations
                            else "candidate_rank_not_improved"
                        ),
                    )
                    history.append(event)
                    if accepted:
                        final_code, final_tests = candidate_code, candidate_tests
                        final_reports = candidate_reports
                        final_ir, final_versions = _bump_versions(candidate_ir, region)
                        final_evidence = _set_evidence_versions(candidate_evidence, final_versions)
                        final_ir["evidence_records"] = copy.deepcopy(final_evidence)
                        final_ir = _append_repair_event(final_ir, event, region)
                        repair_decision = "Accept"
                    else:
                        final_ir = _append_repair_event(ir, event, region)
                        repair_decision = "Reject"
                        rollback_status = "Restored"
                except Exception as exc:
                    repair_decision = "Reject"
                    rollback_status = "Restored"
                    event = repair_event(
                        location=location,
                        scope=scope,
                        before_state=state,
                        candidate_state=state,
                        candidate_execution={"all_passed": False, "error_type": "repair_error"},
                        decision="Reject",
                        rollback_status="Restored",
                        attempt=repair_attempts,
                        reason=f"repair_error:{type(exc).__name__}: {exc}",
                    )
                    history.append(event)
                    final_ir = _append_repair_event(ir, event, region)
            else:
                repair_decision = "Abstained"

        final_state = _state(
            code=final_code,
            ir=final_ir,
            snippets=(
                {
                    id_by_method[name]: source
                    for name, source in _method_sources(final_code, str(entry.get("class_name", ""))).items()
                    if name in id_by_method
                }
            ),
            reports=final_reports,
            execution=final_tests,
            evidence=final_evidence,
            max_repair_attempts=MAX_REPAIR_ATTEMPTS,
            audit_versions=final_versions,
        )
        final_state.update(
            {
                "problem_id": task_id,
                "error_location": location,
                "localization_record": location_result.get("localization_record"),
                "scope_record": scope_result.get("scope_record"),
                "repair_scope": scope,
                "repair_history": history,
                "repair_attempts": repair_attempts,
                "repair_decision": repair_decision,
                "rollback_status": rollback_status,
                "retried_tasks": [repaired_owner] if repaired_owner else [],
                "initial_execution_result": initial_tests,
                "post_repair_execution_result": final_tests,
                "audit_trace": {
                    "G": final_ir.get("contract_annotated_task_graph", final_ir.get("contract_graph", {})),
                    "S": final_ir.get("responsibility_records", []),
                    "E": final_evidence,
                    "H": history,
                },
            }
        )
        report = final_report_assembler(final_state)["final_report"]
        result.update(
            {
                "status": "ok",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_s": round(time.monotonic() - started, 3),
                "initial_class_code": initial_code,
                "final_class_code": final_code,
                "generation_strategy": "analysis_then_code" if reasoning else "direct_code",
                "generation_analysis": generation_analysis if reasoning else None,
                "initial_tests": initial_tests,
                "final_tests": final_tests,
                "pipeline_verdict": report.get("verdict", "unknown"),
                "pipeline_report": report,
                "localization": location_result.get("localization_record"),
                "repair_scope": scope_result.get("scope_record"),
                "repair_attempts": repair_attempts,
                "repair_decision": repair_decision,
                "rollback_status": rollback_status,
                "catg": catg_summary(final_ir),
            }
        )
        if not quiet:
            print(
                f"[{task_id}] {entry.get('class_name', '')}: "
                f"{final_tests.get('pass_count', 0)}/{len(final_tests.get('method_pass_rates', {}))} methods, "
                f"verdict={report.get('verdict')} repair={repair_decision}"
            )
    except Exception as exc:
        result.update(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_s": round(time.monotonic() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=30),
            }
        )
        if not quiet:
            print(f"[{task_id}] ERROR: {exc}")
    return result


def _configure_environment(env_path: Path, provider: str | None) -> str:
    if env_path.exists():
        load_dotenv(env_path, override=False)
    aliases = {"QWEN_API": "QWEN_API_KEY", "DEEPSEEK_API": "DEEPSEEK_API_KEY", "GLM_API": "GLM_API_KEY"}
    for old, new in aliases.items():
        if not os.getenv(new) and os.getenv(old):
            os.environ[new] = os.environ[old]
    chosen = (provider or os.getenv("AUDITCODER_LLM_PROVIDER") or "").strip().lower()
    if not chosen:
        chosen = next(
            (name for name in ("deepseek", "qwen", "glm") if os.getenv(f"{name.upper()}_API_KEY")),
            "",
        )
    if chosen not in {"deepseek", "qwen", "glm"}:
        raise ValueError("configure AUDITCODER_LLM_PROVIDER and its API key")
    key_name = f"{chosen.upper()}_API_KEY"
    if not os.getenv(key_name):
        raise ValueError(f"missing {key_name}; configure it in {env_path}")
    os.environ["AUDITCODER_LLM_PROVIDER"] = chosen
    return chosen


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("task_id"):
            records[str(item["task_id"])] = item
    return records


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in records if item.get("status") == "ok"]
    passed = [item for item in completed if item.get("final_tests", {}).get("all_passed")]
    methods_run = sum(len(item.get("final_tests", {}).get("method_pass_rates", {})) for item in completed)
    methods_passed = sum(item.get("final_tests", {}).get("pass_count", 0) for item in completed)
    repairs = [item for item in completed if item.get("repair_attempts", 0)]
    accepted = [item for item in repairs if item.get("repair_decision") == "Accept"]
    localized = [item for item in completed if (item.get("localization") or {}).get("status") == "localized"]
    return {
        "problems": len(records),
        "completed": len(completed),
        "problem_passed": len(passed),
        "problem_pass_rate": len(passed) / len(completed) if completed else 0.0,
        "methods_run": methods_run,
        "methods_passed": methods_passed,
        "method_pass_rate": methods_passed / methods_run if methods_run else 0.0,
        "repair_attempted": len(repairs),
        "repair_accepted": len(accepted),
        "localized": len(localized),
        "status_counts": dict(Counter(str(item.get("status", "unknown")) for item in records)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--env", type=Path, default=WORKSPACE_ROOT / ".env")
    parser.add_argument("--provider", choices=["deepseek", "qwen", "glm"], default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--ids", help="comma-separated ClassEval task IDs")
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds per unittest subprocess")
    parser.add_argument(
        "--memory-limit-mb",
        type=int,
        default=1024,
        help="address-space limit for each unittest subprocess (default: 1024; scientific imports need more than 256)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="run a concise analysis call before class generation (useful for rerunning hard failures)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if (
        args.start < 0
        or (args.count is not None and args.count < 0)
        or args.timeout <= 0
        or args.memory_limit_mb <= 0
    ):
        parser.error("start/count must be nonnegative; timeout and memory-limit-mb must be positive")
    input_path = args.input.expanduser().resolve()
    entries = load_classeval(input_path)
    requested = [value.strip() for value in args.ids.split(",") if value.strip()] if args.ids else None
    selected = select_entries(entries, start=args.start, count=args.count, end=args.end, ids=requested)
    if not selected:
        parser.error("no ClassEval entries selected")

    print(f"selected {len(selected)} ClassEval classes from {input_path}")
    print(f"AuditCoder implementation: {IMPLEMENTATION_ROOT}")
    print(f"bounded repair attempts per class: {MAX_REPAIR_ATTEMPTS}")
    if args.dry_run:
        print(f"ids: {[item['task_id'] for item in selected[:10]]}{' ...' if len(selected) > 10 else ''}")
        print(f"methods: {sum(len(_method_info(item)) for item in selected)}")
        print("dataset validation: OK")
        return

    provider = _configure_environment(args.env.expanduser().resolve(), args.provider)
    print(f"provider={provider}; repair budget={MAX_REPAIR_ATTEMPTS}")
    output_path = args.output.expanduser().resolve()
    summary_path = args.summary.expanduser().resolve() if args.summary else output_path.with_suffix(".summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    previous = _load_checkpoint(output_path) if args.resume else {}
    if not args.resume:
        output_path.write_text("", encoding="utf-8")
    pending = [item for item in selected if str(item["task_id"]) not in previous]
    with output_path.open("a", encoding="utf-8") as handle:
        for entry in pending:
            record = run_one(
                entry,
                timeout=args.timeout,
                memory_limit_mb=args.memory_limit_mb,
                reasoning=args.reasoning,
                quiet=args.quiet,
            )
            if args.save_artifacts and record.get("status") == "ok":
                artifact_dir = output_path.with_name(output_path.stem + "_artifacts")
                artifact_dir.mkdir(parents=True, exist_ok=True)
                artifact_path = artifact_dir / f"{record['task_id']}.json"
                artifact_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                record["artifact_path"] = str(artifact_path)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            previous[str(record["task_id"])] = record

    selected_records = [previous[str(item["task_id"])] for item in selected if str(item["task_id"]) in previous]
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "input": str(input_path),
            "implementation_root": str(IMPLEMENTATION_ROOT),
            "provider": provider,
            "max_repair_attempts": MAX_REPAIR_ATTEMPTS,
            "timeout": args.timeout,
            "memory_limit_mb": args.memory_limit_mb,
            "reasoning": args.reasoning,
        },
        "selected_count": len(selected),
        "processed_count": len(selected_records),
        "overall": _aggregate(selected_records),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"finished: {summary['processed_count']}/{summary['selected_count']} processed")
    print(f"overall: {json.dumps(summary['overall'], ensure_ascii=False)}")
    print(f"records: {output_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
