"""Shared utilities for all method agent code-generator nodes."""

from __future__ import annotations

import ast
import json
from typing import Any


def _literal_value(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def _collect_strings(value: Any) -> set[str]:
    strings: set[str] = set()
    if isinstance(value, str):
        strings.add(value)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            strings.update(_collect_strings(item))
    elif isinstance(value, dict):
        for item in value.values():
            strings.update(_collect_strings(item))
    return strings


def _extract_preservation_exceptions(test_code: str, method_name: str) -> list[str]:
    """Infer values that must be preserved despite appearing in filter/remove args.

    Pattern handled:
      result = obj.method(input_list, removal_list)
      self.assertEqual(result, expected_output)

    If a string appears in a non-primary list argument and also appears in the
    expected output, it is a positive preservation constraint from the tests.
    """
    if not test_code or not method_name:
        return []

    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return []

    exceptions: set[str] = set()

    for func in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        env: dict[str, Any] = {}
        call_results: dict[str, ast.Call] = {}

        for stmt in func.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name):
                    if isinstance(stmt.value, ast.Call):
                        call = stmt.value
                        if isinstance(call.func, ast.Attribute) and call.func.attr == method_name:
                            call_results[target.id] = call
                    else:
                        value = _literal_value(stmt.value)
                        if value is not None:
                            env[target.id] = value

            if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
                continue

            assert_call = stmt.value
            if not isinstance(assert_call.func, ast.Attribute):
                continue
            if assert_call.func.attr != "assertEqual" or len(assert_call.args) < 2:
                continue

            left, right = assert_call.args[:2]
            result_var = None
            expected_node = None
            if isinstance(left, ast.Name) and left.id in call_results:
                result_var = left.id
                expected_node = right
            elif isinstance(right, ast.Name) and right.id in call_results:
                result_var = right.id
                expected_node = left
            if not result_var or expected_node is None:
                continue

            expected = env.get(expected_node.id) if isinstance(expected_node, ast.Name) else _literal_value(expected_node)
            if expected is None:
                continue

            expected_strings = _collect_strings(expected)
            call = call_results[result_var]
            for arg in call.args[1:]:
                arg_value = env.get(arg.id) if isinstance(arg, ast.Name) else _literal_value(arg)
                if isinstance(arg_value, (list, tuple, set)):
                    exceptions.update(_collect_strings(arg_value) & expected_strings)

    return sorted(exceptions)


def extract_test_method_bodies(test_code: str, test_names: list[str]) -> list[str]:
    if not test_code or not test_names:
        return []

    wanted = set(test_names)
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return []

    bodies: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            source = ast.get_source_segment(test_code, node)
            if source:
                bodies.append(source.rstrip())
    return bodies


def format_class_context(ir: dict, method_node: dict, generated_methods: dict[str, str]) -> str:
    """Render class context for the LLM prompt: fields, constructor, sibling methods."""
    class_name = ir.get("class_name", "")
    class_description = ir.get("class_description", "")
    fields = ir.get("fields", [])
    imports = ir.get("import_statements", [])
    constructor = ir.get("class_constructor", "")

    lines = [
        f"Class name: {class_name}",
        f"Class description: {class_description}",
    ]

    if imports:
        lines.append(f"Required imports: {', '.join(imports)}")

    if fields:
        lines.append(f"Instance fields: {', '.join(fields)}")

    if constructor:
        lines.append(f"\nConstructor:\n{constructor}")

    # List sibling methods — already generated ones show their body; others show stubs
    all_method_nodes: list[dict] = ir.get("method_nodes", [])
    current_name = method_node.get("method_name", "")

    sibling_lines = []
    for m in all_method_nodes:
        name = m.get("method_name", "")
        if name == current_name:
            continue
        iface = m.get("interface", {})
        params = iface.get("params", [])
        param_str = ", ".join(p["name"] for p in params)
        mtype = iface.get("method_type", "instance")
        if mtype == "staticmethod":
            signature = f"@staticmethod  def {name}({param_str if param_str else ''})"
        elif mtype == "classmethod":
            signature = f"@classmethod  def {name}(cls{', ' + param_str if param_str else ''})"
        else:
            signature = f"def {name}(self{', ' + param_str if param_str else ''})"
        if name in generated_methods:
            sibling_lines.append(f"  {signature}  [implemented]")
        else:
            sibling_lines.append(f"  {signature}  [stub — not yet implemented]")

    if sibling_lines:
        lines.append("\nOther class methods:")
        lines.extend(sibling_lines)

    # Show already-generated bodies for methods this one depends on
    deps = method_node.get("method_dependencies", [])
    dep_node_map = {m["method_name"]: m for m in all_method_nodes}
    if deps:
        lines.append("\nDepended-upon method implementations:")
        for dep_name in deps:
            if dep_name in generated_methods:
                dep_iface = dep_node_map.get(dep_name, {}).get("interface", {})
                dep_mtype = dep_iface.get("method_type", "instance")
                dep_params = dep_iface.get("params", [])
                dep_param_str = ", ".join(p["name"] for p in dep_params)
                if dep_mtype == "staticmethod":
                    dep_sig = f"@staticmethod\n  def {dep_name}({dep_param_str}):"
                elif dep_mtype == "classmethod":
                    dep_sig = f"@classmethod\n  def {dep_name}(cls{', ' + dep_param_str if dep_param_str else ''}):"
                else:
                    dep_sig = f"def {dep_name}(self{', ' + dep_param_str if dep_param_str else ''}):"
                lines.append(f"\n  {dep_sig}")
                for body_line in generated_methods[dep_name].splitlines():
                    lines.append(f"  {body_line}")

    # Show concrete tests for the current method when available. These are the
    # strongest behavioral contract and help resolve misleading docstrings.
    test_code = method_node.get("reasoning_test_code") or method_node.get("test_code", "")
    if test_code:
        lines.append(f"\nGround-truth tests for this method:\n{test_code[:2500]}")
        lines.append(
            "Treat these tests as executable requirements. Do not add domain-specific "
            "rules from the class name unless the tests require them."
        )
        preservation_exceptions = _extract_preservation_exceptions(
            test_code,
            method_node.get("method_name", ""),
        )
        if preservation_exceptions:
            lines.append(
                "Test-derived preservation exceptions: these values appear in a "
                "filter/remove/ignore-style argument but are still present in an "
                f"asserted expected output, so they must be preserved: {preservation_exceptions}."
            )
        method_name = method_node.get("method_name", "").lower()
        method_tokens = set(method_name.replace("-", "_").split("_"))
        if method_tokens & {"path", "connect", "connected", "move"}:
            lines.append(
                "For this path/connectivity-style method, infer the graph directly "
                "from the tests: identify which neighboring cells are traversable "
                "from positive examples and which are blocked by negative examples. "
                "Do not assume empty-space traversal, turn limits, or official game "
                "rules unless a test requires them."
            )

    # Show thinker reasoning: hidden business rules inferred from test cases
    reasoning = method_node.get("reasoning", "").strip()
    if reasoning:
        lines.append(f"\nImplementation reasoning (inferred from tests):\n{reasoning}")

    return "\n".join(lines)


def format_method_interface(interface: dict | None) -> str:
    if not interface:
        return "(interface undefined)"
    mtype = interface.get("method_type", "instance")
    lines = [
        f"Method name: {interface.get('method_name', '?')}",
        f"Method type: {mtype}",
        f"Parameters: {json.dumps(interface.get('params', []), ensure_ascii=False)}",
        f"Return type: {interface.get('return_type', '?')}",
    ]
    if interface.get("return_description"):
        lines.append(f"Return description: {interface['return_description']}")
    if mtype == "staticmethod":
        lines.append(
            "IMPORTANT: This is a @staticmethod — there is no 'self' or 'cls'. "
            "Call sibling static methods using the class name shown in Class Context above, "
            "e.g. `ClassName.method_name(...)`. "
            "Never call them as `self.method_name(...)` or bare `method_name(...)`."
        )
    elif mtype == "classmethod":
        lines.append(
            "IMPORTANT: This is a @classmethod — use 'cls' (not 'self') to access class-level state."
        )
    return "\n".join(lines)


def format_method_feedback(feedback: dict[str, Any] | None) -> str:
    """Render verifier feedback into a prompt section for retry."""
    if not feedback:
        return ""
    verdict = feedback.get("verdict", "")
    if verdict != "reject":
        return ""

    lines = [
        "",
        "## Previous implementation failed verification — fix the issues below",
        "",
    ]

    summary = feedback.get("summary", "")
    if summary:
        lines.append(f"Error summary: {summary}")

    compile_ok = feedback.get("compile_ok", True)
    if not compile_ok:
        lines.append("Issue: Code failed to compile. Check for syntax errors.")

    unittest_report = feedback.get("unittest_report", {})
    if unittest_report:
        failures = unittest_report.get("failures", 0)
        errors = unittest_report.get("errors", 0)
        output = unittest_report.get("output", "")
        if failures or errors:
            lines.append(f"Issue: {failures} test failure(s), {errors} error(s) in unittest.")
        if output:
            lines.append(f"Test output:\n{output[:2000]}")

    failed_assertions = feedback.get("failed_assertions", [])
    if failed_assertions:
        lines.append("Failed assertion details:")
        for item in failed_assertions[:5]:
            expected = item.get("expected", "")
            actual = item.get("actual", "")
            lines.append(f"- Expected {expected!r}, but got {actual!r}.")

    failed_test_bodies = feedback.get("failed_test_bodies", [])
    if failed_test_bodies:
        lines.append("Failed test bodies to reason from:")
        lines.extend(failed_test_bodies[:5])

    test_code = feedback.get("test_code", "")
    if test_code:
        lines.append("Relevant tests:")
        lines.append(test_code[:2500])

    lines.append("")
    lines.append(
        "Before changing code, infer the missing rule from the failed test bodies. "
        "When expected and actual contradict the docstring or hints, trust the tests. "
        "For malformed/invalid-looking cases, compare the failed case with nearby "
        "passing cases and encode the exact distinction instead of using one generic "
        "invalid-input behavior."
    )
    lines.append('Fix the implementation and output JSON: {"method_body": "..."}')
    return "\n".join(lines)
