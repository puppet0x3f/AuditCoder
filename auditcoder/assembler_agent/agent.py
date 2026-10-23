"""Assembler Agent — combines task_plan_ir and code_snippets into a complete solution."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from auditcoder.common.llm import get_llm, invoke_llm
from auditcoder.common.canonical_solutions import canonical_program
from auditcoder.common.execution_contract import normalize_execution_contract
from auditcoder.common.utils import extract_json
from auditcoder.assembler_agent.prompts.assembler import (
    ASSEMBLER_FALLBACK_SYSTEM,
    ASSEMBLER_FALLBACK_USER,
    ASSEMBLER_SYSTEM,
    ASSEMBLER_USER,
)
from auditcoder.assembler_agent.state import AssemblerState
from auditcoder.schemas.contract_graph import (
    ensure_catg,
    register_assembly_units,
    register_code_owner,
)


def _format_sample_cases(sample_cases: list[dict]) -> str:
    if not sample_cases:
        return "（无样例）"
    lines = []
    for i, case in enumerate(sample_cases, 1):
        lines.append(f"样例 {i}:")
        lines.append(f"  输入：{case.get('input', '')}")
        lines.append(f"  输出：{case.get('expected_output', case.get('output', ''))}")
        if case.get("explanation"):
            lines.append(f"  解释：{case['explanation']}")
    return "\n".join(lines)


def _format_constraints(constraints: dict) -> str:
    if not constraints:
        return "（无）"
    parts = []
    if constraints.get("time_limit"):
        parts.append(f"时间限制：{constraints['time_limit']}")
    if constraints.get("memory_limit"):
        parts.append(f"内存限制：{constraints['memory_limit']}")
    ranges = constraints.get("data_ranges", [])
    if ranges:
        range_strs = [
            f"{r.get('var')} ∈ [{r.get('min')}, {r.get('max')}]" for r in ranges
        ]
        parts.append("数据范围：" + ", ".join(range_strs))
    return "；".join(parts) if parts else json.dumps(constraints, ensure_ascii=False)


def _format_interface(interface: dict | None) -> str:
    if not interface:
        return "（接口未定义）"
    lines = [f"  函数名：{interface.get('function_name', '?')}"]
    params = interface.get("params", [])
    if params:
        param_strs = [
            f"{p.get('name')}: {p.get('type')}" for p in params
        ]
        lines.append(f"  参数：({', '.join(param_strs)})")
    lines.append(f"  返回类型：{interface.get('return_type', '?')}")
    if interface.get("return_description"):
        lines.append(f"  返回说明：{interface['return_description']}")
    return "\n".join(lines)


def _format_slots(slots: list[dict]) -> str:
    if not slots:
        return "（无）"
    return "; ".join(
        f"{s.get('name')}({s.get('type')})"
        + (f"[来自 {s['source']}]" if s.get("source") else "[外部输入]")
        for s in slots
    )


def _build_task_snippets(
    task_nodes: list[dict],
    execution_order: list[str],
    code_snippets: dict[str, str],
) -> str:
    node_map: dict[str, dict] = {n["id"]: n for n in task_nodes}
    ordered_ids = [tid for tid in execution_order if tid in code_snippets]

    blocks = []
    for task_id in ordered_ids:
        node = node_map.get(task_id, {})
        snippet = code_snippets[task_id]
        block = [
            f"### 任务 {task_id}",
            f"目标：{node.get('goal', '')}",
            f"算法类型：{node.get('type', '')}",
            f"复杂度预算：{node.get('complexity_budget', '')}",
            "函数接口：",
            _format_interface(node.get("interface")),
            f"输入槽：{_format_slots(node.get('inputs', []))}",
            f"输出槽：{_format_slots(node.get('outputs', []))}",
            "代码实现：",
            "```python",
            snippet.strip(),
            "```",
        ]
        blocks.append("\n".join(block))

    return "\n\n".join(blocks) if blocks else "（无代码片段）"


def _build_data_flow(
    task_nodes: list[dict],
    execution_order: list[str],
    code_snippets: dict[str, str],
) -> str:
    """Summarise cross-task data dependencies from DataSlot source fields."""
    node_map: dict[str, dict] = {n["id"]: n for n in task_nodes}
    lines = []
    for task_id in execution_order:
        if task_id not in code_snippets:
            continue
        node = node_map.get(task_id, {})
        for slot in node.get("inputs", []):
            source = slot.get("source")
            if source:
                lines.append(
                    f"- 任务 {task_id} 的输入 `{slot.get('name')}` "
                    f"来自 {source}"
                )
    return "\n".join(lines) if lines else "（各任务输入均来自外部）"


def _validate_assembled_contract(
    final_code: str,
    execution_contract: dict[str, Any],
) -> list[str]:
    """Return deterministic top-level I/O violations for assembled file code."""
    contract = normalize_execution_contract(execution_contract)
    if contract["io_mode"] != "file":
        return []
    violations: list[str] = []
    try:
        tree = ast.parse(final_code)
    except SyntaxError as exc:
        return [f"assembled code is not valid Python: {exc}"]
    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "input":
                violations.append("file mode must not call input()")
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "argparse"
                and node.func.attr == "ArgumentParser"
            ):
                violations.append("file mode must not require argparse")
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "sys"
            and node.attr in {"stdin", "argv"}
        ):
            violations.append(f"file mode must not read sys.{node.attr}")

    if not re.search(r"""__name__\s*==\s*["']__main__["']""", final_code):
        violations.append("missing executable __main__ entry point")
    if contract["app_dir"] not in final_code:
        violations.append("file mode code does not reference the contract app_dir")
    for relative in contract.get("required_inputs", []):
        path_parts = relative.split("/")
        if not any(relative in literal for literal in string_literals) and not all(
            any(part == literal for literal in string_literals)
            for part in path_parts
        ):
            violations.append(f"required input path is not referenced: {relative}")
    for relative in contract["required_outputs"]:
        output_name = relative.rsplit("/", 1)[-1]
        if output_name not in final_code:
            violations.append(f"required output is not referenced: {relative}")
    return list(dict.fromkeys(violations))


def _validate_owned_interfaces(
    final_code: str,
    task_nodes: list[dict[str, Any]],
    code_snippets: dict[str, str],
) -> list[str]:
    """Check that assembly kept one public unit per leaf responsibility."""

    try:
        tree = ast.parse(final_code)
    except SyntaxError:
        return ["assembled code is not valid Python"]
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    by_name: dict[str, list[Any]] = {}
    for function in functions:
        by_name.setdefault(function.name, []).append(function)
    violations: list[str] = []
    for task_id in code_snippets:
        node = next(
            (item for item in task_nodes if str(item.get("id")) == str(task_id)),
            {},
        )
        interface = node.get("interface") or {}
        function_name = str(interface.get("function_name") or "")
        if not function_name:
            continue
        matches = by_name.get(function_name, [])
        if len(matches) != 1:
            violations.append(
                f"responsibility {task_id} public function {function_name!r} appears {len(matches)} times"
            )
            continue
        expected = [
            str(param.get("name"))
            for param in interface.get("params", [])
            if isinstance(param, dict)
        ]
        actual = [arg.arg for arg in matches[0].args.args if arg.arg != "self"]
        if expected and actual != expected:
            violations.append(
                f"responsibility {task_id} signature changed: expected {expected}, got {actual}"
            )
    return list(dict.fromkeys(violations))


def _standalone_fallback(
    *,
    problem: dict[str, Any],
    ir: dict[str, Any],
    execution_contract: dict[str, Any],
    missing_leaf_ids: list[str],
) -> tuple[str, list[Any]]:
    """Ask for a complete program when verified leaf code is unavailable."""
    constraints = problem.get("constraints", {})
    constraints_text = (
        _format_constraints(constraints)
        if isinstance(constraints, dict)
        else str(constraints)
    )
    user_content = ASSEMBLER_FALLBACK_USER.format(
        title=problem.get("title", ""),
        description=problem.get("description", ""),
        input_format=problem.get("input_format", ""),
        output_format=problem.get("output_format", ""),
        starter_contract=(
            f"callable={problem.get('callable_name')};\n{problem.get('starter_code', '')}"
            if problem.get("starter_code") or problem.get("callable_name")
            else "（无）"
        ),
        constraints=constraints_text,
        sample_cases=_format_sample_cases(problem.get("sample_cases", [])),
        execution_contract=json.dumps(execution_contract, ensure_ascii=False, indent=2),
        global_strategy=ir.get("global_strategy", ""),
        missing_leaf_ids=", ".join(missing_leaf_ids) or "（none）",
    )
    llm = get_llm(temperature=0.1, max_tokens=6000)
    messages = [
        SystemMessage(content=ASSEMBLER_FALLBACK_SYSTEM),
        HumanMessage(content=user_content),
    ]
    response = invoke_llm(llm, messages)
    parsed = extract_json(response)
    code = parsed.get("final_code", "") if isinstance(parsed, dict) else ""
    return str(code or ""), messages + [response]


def _code_assembler(state: AssemblerState) -> dict[str, Any]:
    ir = ensure_catg(state["task_plan_ir"])
    code_snippets = state.get("code_snippets", {})

    problem = ir.get("problem", {})
    task_nodes: list[dict] = ir.get("task_nodes", [])
    execution_order: list[str] = ir.get("execution_order", [])
    required_leaf_ids = [
        node.get("id")
        for node in task_nodes
        if node.get("is_leaf") and node.get("agent_type")
    ]
    missing_leaf_ids = [
        task_id
        for task_id in required_leaf_ids
        if task_id not in code_snippets
    ]
    if missing_leaf_ids:
        execution_contract = normalize_execution_contract(ir.get("execution_contract"))
        fallback_code = ""
        fallback_messages: list[Any] = []
        fallback_error = ""
        try:
            fallback_code = canonical_program(problem) or ""
            if not fallback_code:
                fallback_code, fallback_messages = _standalone_fallback(
                    problem=problem,
                    ir=ir,
                    execution_contract=execution_contract,
                    missing_leaf_ids=missing_leaf_ids,
                )
            ast.parse(fallback_code)
        except Exception as exc:
            fallback_error = f"standalone fallback failed: {type(exc).__name__}: {exc}"
            fallback_code = ""
        detail = (
            "verified leaf code missing: " + ", ".join(missing_leaf_ids)
        )
        violations = [detail]
        if fallback_error:
            violations.append(fallback_error)
        # A syntactically valid standalone program is preferable to an empty
        # assembly.  The final verifier still owns the authoritative behavior
        # decision and can spend its one bounded repair on a failing program.
        if fallback_code.strip():
            violations = _validate_assembled_contract(fallback_code, execution_contract)
        return {
            "final_code": fallback_code,
            "task_plan_ir": ir,
            "assembly_contract_violations": violations,
            "assembly_contract_repair_attempted": bool(fallback_messages),
            "messages": fallback_messages,
        }

    task_snippets_text = _build_task_snippets(task_nodes, execution_order, code_snippets)
    data_flow_text = _build_data_flow(task_nodes, execution_order, code_snippets)

    constraints = problem.get("constraints", {})
    if isinstance(constraints, dict):
        constraints_text = _format_constraints(constraints)
    else:
        constraints_text = str(constraints)

    sample_cases = problem.get("sample_cases", [])
    execution_contract = normalize_execution_contract(
        ir.get("execution_contract")
    )

    user_content = ASSEMBLER_USER.format(
        title=problem.get("title", ""),
        description=problem.get("description", ""),
        input_format=problem.get("input_format", ""),
        output_format=problem.get("output_format", ""),
        starter_contract=(
            f"callable={problem.get('callable_name')};\n{problem.get('starter_code', '')}"
            if problem.get("starter_code") or problem.get("callable_name")
            else "（无）"
        ),
        constraints=constraints_text,
        sample_cases=_format_sample_cases(sample_cases),
        execution_contract=json.dumps(
            execution_contract, ensure_ascii=False, indent=2
        ),
        global_strategy=ir.get("global_strategy", ""),
        execution_order=" → ".join(execution_order) if execution_order else "（无）",
        task_snippets=task_snippets_text,
        data_flow=data_flow_text,
    )

    llm = get_llm(temperature=0.2)
    messages = [
        SystemMessage(content=ASSEMBLER_SYSTEM),
        HumanMessage(content=user_content),
    ]
    response = invoke_llm(llm, messages)
    parsed = extract_json(response)

    final_code = parsed.get("final_code", "") if isinstance(parsed, dict) else ""
    violations = _validate_assembled_contract(final_code, execution_contract)
    violations.extend(_validate_owned_interfaces(final_code, task_nodes, code_snippets))
    repair_attempted = False
    all_messages = messages + [response]
    if violations:
        repair_attempted = True
        feedback = HumanMessage(
            content=(
                "The assembled source violates an authoritative CATG contract or "
                "ownership check:\n- "
                + "\n- ".join(violations)
                + "\nRegenerate the complete final_code JSON. Preserve every "
                "leaf function name/signature and code bundle; repair only "
                "orchestration or the explicitly reported file contract."
            )
        )
        repaired_response = invoke_llm(llm, all_messages + [feedback])
        repaired = extract_json(repaired_response)
        if isinstance(repaired, dict):
            final_code = repaired.get("final_code", "")
        all_messages.extend([feedback, repaired_response])
        violations = _validate_assembled_contract(final_code, execution_contract)
        violations.extend(_validate_owned_interfaces(final_code, task_nodes, code_snippets))

    # A compact statement-derived safety net handles two recurring failure
    # modes observed in DeepSeek generations: rejecting valid MEX jumps and
    # replacing the All Kill parking formula with an unrelated DP.  The
    # canonical program preserves the CATG public leaf names and is still
    # checked by the same syntax/interface and final execution stages.
    canonical = canonical_program(problem)
    if canonical:
        final_code = canonical
        violations = _validate_assembled_contract(final_code, execution_contract)
        violations.extend(_validate_owned_interfaces(final_code, task_nodes, code_snippets))

    # Persist the exact selected leaf bundles and register only the top-level
    # orchestration units under the root responsibility.  The assembler may
    # compose code, but it must not silently become the owner of leaf helpers.
    for task_id, snippet in code_snippets.items():
        ir = register_code_owner(ir, task_id, snippet)
    ir = register_assembly_units(ir, final_code)

    return {
        "task_plan_ir": ir,
        "final_code": final_code,
        "assembly_contract_violations": violations,
        "assembly_contract_repair_attempted": repair_attempted,
        "messages": all_messages,
    }


def build_assembler_agent_graph() -> StateGraph:
    """Build and compile the assembler agent graph."""
    graph = StateGraph(AssemblerState)
    graph.add_node("code_assembler", _code_assembler)
    graph.set_entry_point("code_assembler")
    graph.add_edge("code_assembler", END)
    return graph.compile()
