"""Rule-based class assembler: inject generated method bodies into the class skeleton.

No LLM is needed — the ClassEval skeleton is machine-generated and consistent.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from langgraph.graph import END, StateGraph

from auditcoder_class.class_assembler_agent.state import ClassAssemblerState


def _find_method_body_range(lines: list[str], method_name: str) -> tuple[int, int] | None:
    """Return (body_start, body_end) line indices for method_name's body in the class.

    body_start is the first line after the def+docstring.
    body_end is exclusive (the line index of the next method def at the same indent, or EOF).
    Returns None if method not found.
    """
    method_def_re = re.compile(r'^    def\s+' + re.escape(method_name) + r'\s*\(')
    any_def_re = re.compile(r'^    def\s+\w')
    any_decorator_re = re.compile(r'^    @\w')

    method_start = None
    for i, line in enumerate(lines):
        if method_def_re.match(line):
            method_start = i
            break

    if method_start is None:
        return None

    # Advance past the def line
    i = method_start + 1

    # Skip the docstring (triple-quoted or single-line)
    if i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote_char = stripped[:3]
            if stripped.count(quote_char) >= 2 and len(stripped) > 3:
                # Single-line docstring: """ ... """
                i += 1
            else:
                # Multi-line docstring: advance until closing triple-quote
                i += 1
                while i < len(lines) and quote_char not in lines[i]:
                    i += 1
                i += 1  # skip closing line
        elif stripped.startswith('"') or stripped.startswith("'"):
            i += 1  # single-quoted one-liner docstring

    body_start = i

    # Find end: next decorator or def at class-method indent level, or EOF.
    # Must stop at the decorator (e.g. @staticmethod) rather than the def line so
    # that decorators are not consumed into the replaced body range.
    body_end = len(lines)
    for j in range(body_start, len(lines)):
        if any_decorator_re.match(lines[j]) or any_def_re.match(lines[j]):
            body_end = j
            break

    return body_start, body_end


def _replace_method_body(skeleton: str, method_name: str, new_body: str) -> str:
    """Replace the stub body of method_name in skeleton with new_body."""
    lines = skeleton.splitlines()
    range_ = _find_method_body_range(lines, method_name)
    if range_ is None:
        return skeleton

    body_start, body_end = range_

    # Normalise literal \n sequences that json_repair may leave when the LLM
    # double-escapes newlines in the JSON string value.
    # Only apply if the replacement produces valid Python syntax — a blanket
    # replace breaks code that legitimately contains '\n' string literals
    # (e.g. "'\n'.join(...)") by inserting a real newline inside the quote.
    _replaced = new_body.replace('\\n', '\n')
    try:
        ast.parse("class _C:\n    def _f(self):\n" + _replaced)
        new_body = _replaced
    except SyntaxError:
        pass  # keep new_body as-is; the \n are escape sequences, not line separators

    # Normalise indentation: ensure body lines are indented by 8 spaces
    new_body_lines = []
    for line in new_body.splitlines():
        # If the line is blank, keep it blank; otherwise ensure 8-space indent
        if line.strip() == "":
            new_body_lines.append("")
        elif not line.startswith("        "):
            new_body_lines.append("        " + line.lstrip())
        else:
            new_body_lines.append(line)

    # Fall back to a pass statement if body is empty
    if not new_body_lines or all(l.strip() == "" for l in new_body_lines):
        new_body_lines = ["        pass"]

    result_lines = lines[:body_start] + new_body_lines + lines[body_end:]
    return "\n".join(result_lines)


def build_partial_class(class_plan_ir: dict[str, Any], generated_methods: dict[str, str]) -> str:
    """Assemble a class with real bodies for generated methods and stubs for the rest.

    Used by the verifier during per-method dispatch.
    """
    skeleton = class_plan_ir.get("skeleton", "")
    for method_name, body in generated_methods.items():
        skeleton = _replace_method_body(skeleton, method_name, body)
    return skeleton


def check_structural_integrity(
    class_code: str, class_plan_ir: dict[str, Any]
) -> list[dict[str, Any]]:
    """Detect structural issues in the assembled class code.

    Returns a list of violation dicts with at least {"method_name", "issue"}.
    """
    violations: list[dict[str, Any]] = []
    lines = class_code.splitlines()

    try:
        compile(class_code, "<assembled_class>", "exec")
    except SyntaxError as e:
        method_name = ""
        if e.lineno is not None:
            for line in reversed(lines[: e.lineno]):
                match = re.match(r'\s*def\s+(\w+)\s*\(', line)
                if match:
                    method_name = match.group(1)
                    break
        violations.append({
            "method_name": method_name,
            "expected_type": "compilable_python",
            "issue": f"Compile error in assembled class: {e}",
        })

    for node in class_plan_ir.get("method_nodes", []):
        iface = node.get("interface", {})
        expected_type = iface.get("method_type", "instance")
        if expected_type not in ("staticmethod", "classmethod"):
            continue
        method_name = node.get("method_name", "")
        decorator = f"@{expected_type}"
        # Find the def line; check whether the preceding non-blank line is the decorator
        for i, line in enumerate(lines):
            if re.match(r'\s*def\s+' + re.escape(method_name) + r'\s*\(', line):
                # Walk back to find first non-blank line
                prev_idx = i - 1
                while prev_idx >= 0 and lines[prev_idx].strip() == "":
                    prev_idx -= 1
                if prev_idx < 0 or lines[prev_idx].strip() != decorator:
                    violations.append({
                        "method_name": method_name,
                        "expected_type": expected_type,
                        "issue": f"Missing {decorator} decorator before def {method_name}",
                    })
                break  # found the def line; no need to keep scanning

    return violations


def _class_assembler(state: ClassAssemblerState) -> dict[str, Any]:
    ir = state["class_plan_ir"]
    method_bodies = state["method_bodies"]

    code = build_partial_class(ir, method_bodies)

    # Prepend import statements if not already present in the skeleton
    imports = ir.get("import_statements", [])
    existing_first_line = code.splitlines()[0] if code.splitlines() else ""
    if imports and not existing_first_line.startswith("import") and not existing_first_line.startswith("from"):
        import_block = "\n".join(imports)
        code = import_block + "\n\n" + code

    structural_violations = check_structural_integrity(code, ir)

    return {"final_class_code": code, "structural_violations": structural_violations}


def build_class_assembler_graph() -> StateGraph:
    graph = StateGraph(ClassAssemblerState)
    graph.add_node("class_assembler", _class_assembler)
    graph.set_entry_point("class_assembler")
    graph.add_edge("class_assembler", END)
    return graph.compile()
