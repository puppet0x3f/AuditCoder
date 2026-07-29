"""Rule-based node: assemble ClassPlanIR from class_context + selections + order."""

from __future__ import annotations

from typing import Any

from auditcoder_class.class_plan_agent.state import ClassPlanState


def ir_assembler(state: ClassPlanState) -> dict[str, Any]:
    """Merge class_context, method_algo_selection, and execution_order into ClassPlanIR dict."""
    ctx = state["class_context"]
    algo_selection = state["method_algo_selection"]
    execution_order = state["execution_order"]

    # Build lookup: method_name → algo selection
    algo_map = {s["method_name"]: s for s in algo_selection}
    # Build lookup: method_name → raw method info
    method_map = {m["method_name"]: m for m in ctx["methods"]}
    # Build lookup: method_name → thinker reasoning
    thoughts_map = {
        t["method_name"]: t.get("reasoning", "")
        for t in state.get("method_thoughts", [])
    }

    method_nodes = []
    for name in execution_order:
        m = method_map.get(name)
        if not m:
            continue
        algo = algo_map.get(name, {"agent_type": "math", "hints": []})
        method_nodes.append({
            "method_name": name,
            "method_description": m["method_description"],
            "agent_type": algo["agent_type"],
            "is_standalone": m["is_standalone"],
            "lib_dependencies": m["lib_dependencies"],
            "field_dependencies": m["field_dependencies"],
            "method_dependencies": m["method_dependencies"],
            "interface": m["interface"],
            "hints": algo.get("hints", []),
            "reasoning": thoughts_map.get(name, ""),
            "test_class_name": m["test_class_name"],
            "test_code": m["test_code"],
            "reasoning_test_code": m.get("reasoning_test_code", ""),
        })

    class_plan_ir = {
        "task_id": ctx["task_id"],
        "class_name": ctx["class_name"],
        "class_description": ctx["class_description"],
        "import_statements": ctx["import_statements"],
        "class_constructor": ctx["class_constructor"],
        "fields": ctx["fields"],
        "skeleton": ctx["skeleton"],
        "method_nodes": method_nodes,
        "execution_order": execution_order,
        "test": ctx["test"],
        "test_classes": ctx["test_classes"],
    }
    return {"class_plan_ir": class_plan_ir}
