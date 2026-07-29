"""Node: contract_verifier — 检查代码是否满足接口契约。

纯程序化：AST 解析 + 签名匹配，不调用 LLM。
"""

from __future__ import annotations

import ast
from typing import Any

from auditcoder.verifier_agent.state import VerifierState


def _find_functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    """Return top-level function definitions keyed by name."""
    funcs: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = node
    return funcs


def _get_annotation_str(annotation: ast.expr | None) -> str:
    if annotation is None:
        return ""
    return ast.unparse(annotation)


def _check_signature(
    func: ast.FunctionDef,
    interface: dict[str, Any],
) -> list[str]:
    """Check function signature against FunctionInterface contract."""
    violations: list[str] = []
    params = interface.get("params", [])

    arg_names = [a.arg for a in func.args.args]
    if arg_names and arg_names[0] == "self":
        arg_names = arg_names[1:]

    expected_names = [p["name"] for p in params]

    for expected in expected_names:
        if expected not in arg_names:
            violations.append(f"缺少参数: {expected}")

    for actual in arg_names:
        if actual not in expected_names:
            violations.append(f"多余参数: {actual}")

    for p in params:
        idx_list = [i for i, a in enumerate(func.args.args) if a.arg == p["name"]]
        if not idx_list:
            continue
        idx = idx_list[0]
        arg_node = func.args.args[idx]
        actual_type = _get_annotation_str(arg_node.annotation)
        expected_type = p.get("type", "")
        if expected_type and actual_type and actual_type != expected_type:
            violations.append(
                f"参数 {p['name']} 类型不匹配: 期望 {expected_type}, 实际 {actual_type}"
            )

    ret_annotation = _get_annotation_str(func.returns)
    expected_ret = interface.get("return_type", "")
    if expected_ret and ret_annotation and ret_annotation != expected_ret:
        violations.append(
            f"返回类型不匹配: 期望 {expected_ret}, 实际 {ret_annotation}"
        )

    return violations


def _check_shared_state_mutations(tree: ast.Module) -> list[str]:
    """Detect global/nonlocal keyword usage and suspicious mutations."""
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            violations.append(f"使用了 global 关键字: {', '.join(node.names)}")
        elif isinstance(node, ast.Nonlocal):
            violations.append(f"使用了 nonlocal 关键字: {', '.join(node.names)}")
    return violations


def contract_verifier(state: VerifierState) -> dict:
    snippet = state["code_snippet"]
    contract = state["task_contract"]
    source_code = snippet["source_code"]
    interface = contract.get("interface", {})
    function_name = interface.get("function_name", "")

    violations: list[str] = []
    contract_ok = True

    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        violations.append(f"语法错误，无法解析 AST: {exc}")
        return {
            "contract_report": {
                "contract_ok": False,
                "violations": violations,
            }
        }

    funcs = _find_functions(tree)

    if function_name and function_name not in funcs:
        violations.append(f"未找到契约要求的函数: {function_name}")
    elif function_name:
        sig_violations = _check_signature(funcs[function_name], interface)
        violations.extend(sig_violations)

    state_violations = _check_shared_state_mutations(tree)
    violations.extend(state_violations)

    if violations:
        contract_ok = False

    return {
        "contract_report": {
            "contract_ok": contract_ok,
            "violations": violations,
        }
    }
