"""Rule-based node: topologically sort methods by their method_dependencies."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from auditcoder_class.class_plan_agent.state import ClassPlanState


def dependency_sorter(state: ClassPlanState) -> dict[str, Any]:
    """Return execution_order: method names sorted so dependencies come before dependents."""
    methods = state["class_context"]["methods"]
    method_names = {m["method_name"] for m in methods}

    in_degree: dict[str, int] = {m["method_name"]: 0 for m in methods}
    dependents: dict[str, list[str]] = defaultdict(list)

    for m in methods:
        name = m["method_name"]
        for dep in m["method_dependencies"]:
            if dep in method_names:
                in_degree[name] += 1
                dependents[dep].append(name)

    queue: deque[str] = deque(
        name for name, deg in in_degree.items() if deg == 0
    )
    # Preserve original order for same-level nodes
    order_index = {m["method_name"]: i for i, m in enumerate(methods)}
    queue = deque(sorted(queue, key=lambda n: order_index[n]))

    execution_order: list[str] = []
    while queue:
        node = queue.popleft()
        execution_order.append(node)
        for dependent in sorted(dependents[node], key=lambda n: order_index[n]):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    # Handle cycles: append remaining nodes in original order
    remaining = [m["method_name"] for m in methods if m["method_name"] not in execution_order]
    if remaining:
        execution_order.extend(remaining)

    return {"execution_order": execution_order}
