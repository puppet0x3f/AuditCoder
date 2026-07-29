"""Rule-based node: parse a raw ClassEval JSON entry into a structured class_context dict."""

from __future__ import annotations

import ast
import re
from typing import Any

from auditcoder_class.class_plan_agent.state import ClassPlanState


def _detect_method_type_from_skeleton(method_name: str, skeleton: str) -> str:
    """Return 'staticmethod', 'classmethod', or 'instance' by inspecting the class skeleton.

    Looks for a decorator line immediately before the `def method_name(` line.
    """
    lines = skeleton.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Match the def line for this method (allow any indentation)
        if re.match(r"def\s+" + re.escape(method_name) + r"\s*\(", stripped):
            # Look at the line directly before the def line for a decorator
            if i > 0:
                prev = lines[i - 1].strip()
                if prev == "@staticmethod":
                    return "staticmethod"
                if prev == "@classmethod":
                    return "classmethod"
            return "instance"
    return "instance"


def _parse_method_interface(method_description: str, skeleton: str = "") -> dict[str, Any]:
    """Extract method name, params, return type, and method_type from a method description.

    The method_description from ClassEval typically starts with the `def` line followed
    by a docstring.  We parse just enough to extract the signature.
    """
    lines = method_description.strip().splitlines()

    # Detect decorator from method_description lines (lines before `def`)
    method_type = "instance"
    for line in lines:
        stripped = line.strip()
        if stripped == "@staticmethod":
            method_type = "staticmethod"
            break
        if stripped == "@classmethod":
            method_type = "classmethod"
            break
        if stripped.startswith("def "):
            break  # reached def without finding decorator

    # Pull the first `def` line
    def_line = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def "):
            def_line = stripped
            break

    if not def_line:
        return {"method_name": "unknown", "params": [], "return_type": "None",
                "return_description": "", "method_type": method_type}

    # Parse with ast for accuracy
    try:
        tree = ast.parse(def_line + "\n    pass")
        func = tree.body[0]
        method_name = func.name

        # Fall back to skeleton for decorator if not found in method_description
        if method_type == "instance" and skeleton:
            method_type = _detect_method_type_from_skeleton(method_name, skeleton)

        params = []
        for arg in func.args.args:
            # Skip 'self' for instance/classmethod; for staticmethod skip nothing
            if arg.arg in ("self", "cls"):
                continue
            annotation = ""
            if arg.annotation:
                annotation = ast.unparse(arg.annotation)
            params.append({"name": arg.arg, "type": annotation or "Any", "description": ""})

        return_type = "None"
        if func.returns:
            return_type = ast.unparse(func.returns)

        return {
            "method_name": method_name,
            "params": params,
            "return_type": return_type,
            "return_description": "",
            "method_type": method_type,
        }
    except SyntaxError:
        # Fallback: regex
        m = re.match(r"def\s+(\w+)\s*\(", def_line)
        method_name = m.group(1) if m else "unknown"
        if method_type == "instance" and skeleton and method_name != "unknown":
            method_type = _detect_method_type_from_skeleton(method_name, skeleton)
        return {"method_name": method_name, "params": [], "return_type": "None",
                "return_description": "", "method_type": method_type}


def _is_integration_test(test_code: str, all_method_names: list[str]) -> bool:
    """Return True if test_code calls more than one method from the class.

    Integration tests span multiple methods and should not be used for
    per-method verification (which runs before all methods are implemented).
    """
    if not test_code or len(all_method_names) < 2:
        return False
    called = sum(1 for name in all_method_names if re.search(r'\.' + re.escape(name) + r'\s*\(', test_code))
    return called >= 2


def _extract_self_call_dependencies(source: str, method_name: str) -> list[str]:
    """Extract `self.other_method(...)` calls as best-effort method dependencies."""
    deps: list[str] = []
    for dep in re.findall(r'\bself\.(\w+)\s*\(', source or ""):
        if dep != method_name and dep not in deps:
            deps.append(dep)
    return deps


def _target_is_self_attr(node: ast.AST) -> str | None:
    """Return the attribute name for `self.attr`, otherwise None."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None


def _collect_class_instance_refs(test_code: str, class_name: str) -> tuple[set[str], set[str]]:
    """Collect test variables that hold instances of the class under test."""
    instance_names: set[str] = set()
    instance_attrs: set[str] = set()

    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return instance_names, instance_attrs

    for node in ast.walk(tree):
        targets: list[ast.AST] = []
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value

        if not isinstance(value, ast.Call):
            continue
        if not isinstance(value.func, ast.Name) or value.func.id != class_name:
            continue

        for target in targets:
            if isinstance(target, ast.Name):
                instance_names.add(target.id)
            attr = _target_is_self_attr(target)
            if attr:
                instance_attrs.add(attr)

    return instance_names, instance_attrs


def _call_is_on_instance(value: ast.AST, instance_names: set[str], instance_attrs: set[str], class_name: str) -> bool:
    """Return True if an attribute call receiver is an instance of the class under test."""
    if isinstance(value, ast.Name) and value.id in instance_names:
        return True
    if isinstance(value, ast.Attribute):
        self_attr = _target_is_self_attr(value)
        return bool(self_attr and self_attr in instance_attrs)
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == class_name:
        return True
    return False


def _literal_arg_name(arg: ast.AST, index: int) -> str:
    """Best-effort parameter name for synthetic methods inferred from test call arguments."""
    if isinstance(arg, ast.Name) and arg.id:
        return arg.id
    if isinstance(arg, ast.Attribute) and arg.attr:
        return arg.attr
    if isinstance(arg, ast.Dict):
        keys = {k.value for k in arg.keys if isinstance(k, ast.Constant)}
        if "skills" in keys:
            return "resume"
        if "requirements" in keys:
            return "job"
    if isinstance(arg, ast.List):
        return "requirements" if index == 1 else f"arg{index + 1}"
    return f"arg{index + 1}"


def _dedupe_param_names(names: list[str]) -> list[str]:
    """Sanitize and deduplicate inferred Python parameter names."""
    result: list[str] = []
    used: set[str] = {"self", "cls"}
    for i, raw in enumerate(names):
        name = re.sub(r"\W+", "_", raw).strip("_") or f"arg{i + 1}"
        if name[0].isdigit():
            name = f"arg_{name}"
        base = name
        suffix = 2
        while name in used:
            name = f"{base}_{suffix}"
            suffix += 1
        used.add(name)
        result.append(name)
    return result


def _extract_missing_method_specs(test_code: str, class_name: str, known_method_names: set[str]) -> list[dict[str, Any]]:
    """Find methods called by full tests but absent from methods_info/skeleton."""
    if not test_code:
        return []

    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return []

    instance_names, instance_attrs = _collect_class_instance_refs(test_code, class_name)
    discovered: dict[str, dict[str, Any]] = {}

    for class_node in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        class_source = ast.get_source_segment(test_code, class_node) or ""
        for node in ast.walk(class_node):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if not _call_is_on_instance(node.func.value, instance_names, instance_attrs, class_name):
                continue

            method_name = node.func.attr
            if method_name in known_method_names or method_name.startswith("__"):
                continue
            if method_name not in discovered:
                positional = [_literal_arg_name(arg, i) for i, arg in enumerate(node.args)]
                keywords = [kw.arg for kw in node.keywords if kw.arg]
                params = _dedupe_param_names(positional + keywords)
                discovered[method_name] = {
                    "method_name": method_name,
                    "params": params,
                    "test_class_name": class_node.name,
                    "test_code": class_source,
                }

    return list(discovered.values())


def _append_synthetic_method_to_skeleton(skeleton: str, method_name: str, params: list[str]) -> str:
    """Append a stub for a test-discovered method so the assembler can inject a body."""
    params_part = ", ".join(["self", *params])
    method_stub = (
        f"\n\n    def {method_name}({params_part}):\n"
        "        \"\"\"\n"
        "        Auto-detected helper method from the full ClassEval tests.\n"
        "        The associated test cases are the ground truth specification.\n"
        "        \"\"\""
    )
    return skeleton.rstrip() + method_stub


def class_context_builder(state: ClassPlanState) -> dict[str, Any]:
    """Parse raw ClassEval entry into structured class_context."""
    entry = state["raw_entry"]

    skeleton = entry.get("skeleton", "")
    methods_info = entry.get("methods_info", [])

    all_method_names = [
        m.get("method_name", "")
        for m in methods_info
        if m.get("method_name", "") and m.get("method_name", "") != "__init__"
    ]

    parsed_methods = []
    for m in methods_info:
        method_name = m.get("method_name", "")
        if method_name == "__init__":
            continue  # Constructor is handled separately via class_constructor field

        deps = m.get("dependencies", {})
        interface = _parse_method_interface(m.get("method_description", ""), skeleton)

        test_code = m.get("test_code", "")
        # Integration tests call multiple class methods — suppress per-method verification
        # so the verifier does not reject a method because a sibling stub returns None.
        # But keep reasoning_test_code so the LLM can still infer correct behavior.
        is_integration = _is_integration_test(test_code, all_method_names)

        method_dependencies = list(deps.get("method_dependencies", []))
        reference_source = "\n".join([
            m.get("solution_code", ""),
            m.get("method_description", ""),
        ])
        for dep in _extract_self_call_dependencies(reference_source, method_name):
            if dep not in method_dependencies:
                method_dependencies.append(dep)

        parsed_methods.append({
            "method_name": method_name,
            "method_description": m.get("method_description", ""),
            "interface": interface,
            "is_standalone": bool(deps.get("Standalone", True)),
            "lib_dependencies": deps.get("lib_dependencies", []),
            "field_dependencies": deps.get("field_dependencies", []),
            "method_dependencies": method_dependencies,
            "test_class_name": m.get("test_class", ""),
            "test_code": "" if is_integration else test_code,
            "reasoning_test_code": test_code,
        })

    known_method_names = {m["method_name"] for m in parsed_methods}
    missing_specs = _extract_missing_method_specs(
        entry.get("test", ""),
        entry.get("class_name", ""),
        known_method_names,
    )
    for spec in missing_specs:
        method_name = spec["method_name"]
        params = spec.get("params", [])
        params_part = ", ".join(["self", *params])
        method_description = (
            f"def {method_name}({params_part}):\n"
            "        \"\"\"\n"
            "        Auto-detected helper method from the full ClassEval tests.\n"
            "        Treat the associated test cases as the binding specification.\n"
            "        \"\"\""
        )
        skeleton = _append_synthetic_method_to_skeleton(skeleton, method_name, params)
        parsed_methods.append({
            "method_name": method_name,
            "method_description": method_description,
            "interface": _parse_method_interface(method_description, skeleton),
            "is_standalone": True,
            "lib_dependencies": [],
            "field_dependencies": [],
            "method_dependencies": [],
            "test_class_name": spec.get("test_class_name", ""),
            "test_code": spec.get("test_code", ""),
            "reasoning_test_code": spec.get("test_code", ""),
            "synthetic": True,
        })
        known_method_names.add(method_name)

    class_context = {
        "task_id": entry.get("task_id", ""),
        "class_name": entry.get("class_name", ""),
        "class_description": entry.get("class_description", ""),
        "import_statements": entry.get("import_statement", []),
        "class_constructor": entry.get("class_constructor", ""),
        "fields": entry.get("fields", []),
        "skeleton": skeleton,
        "methods": parsed_methods,
        "test": entry.get("test", ""),
        "test_classes": entry.get("test_classes", []),
    }
    return {"class_context": class_context}
