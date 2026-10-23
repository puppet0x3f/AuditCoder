"""Offline one-problem smoke test for the CATG-to-verifier integration.

The interval task is deterministic: no model is called. It exercises
pre-code graph compilation, leaf ownership registration, assembly ownership
registration, fixed development checks, and the final runner.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PACKAGE_ROOT = HERE.parent.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from auditcoder.final_verifier_agent.graph import build_final_verifier_graph  # noqa: E402
from auditcoder.schemas.contract_graph import (  # noqa: E402
    catg_summary,
    ensure_catg,
    register_assembly_units,
    register_code_owner,
    validate_contract_graph,
)


def _problem_ir() -> dict:
    checks = [
        {"input": "5\n1 3\n2 4\n5 6\n8 9\n8 10\n", "expected_output": "2"},
        {"input": "4\n1 1\n2 2\n3 3\n4 4\n", "expected_output": "1"},
        {"input": "1\n-5 -5\n", "expected_output": "1"},
    ]
    return {
        "problem_id": "catg_interval_smoke",
        "problem": {
            "title": "Count connected interval groups",
            "description": "Merge overlapping or touching closed intervals and count groups.",
            "constraints": {"time_limit": "2s", "memory_limit": "256MB"},
        },
        "root_id": "S0",
        "task_nodes": [
            {
                "id": "S0",
                "goal": "Read input, orchestrate sorting and greedy merging, and print the count.",
                "type": "orchestration",
                "is_leaf": False,
                "parent": None,
                "dependencies": ["S1", "S2"],
                "inputs": [],
                "outputs": [],
            },
            {
                "id": "S1",
                "goal": "Sort intervals by left and right endpoint.",
                "type": "sorting",
                "is_leaf": True,
                "agent_type": "sorting",
                "parent": "S0",
                "dependencies": [],
                "inputs": [{"name": "intervals", "type": "list[tuple[int, int]]", "source": None}],
                "outputs": [{"name": "sorted_intervals", "type": "list[tuple[int, int]]"}],
                "interface": {
                    "function_name": "sort_intervals",
                    "params": [{"name": "intervals", "type": "list[tuple[int, int]]"}],
                    "return_type": "list[tuple[int, int]]",
                    "preconditions": ["each interval has l <= r"],
                    "postconditions": ["output is sorted by (l, r)"],
                },
            },
            {
                "id": "S2",
                "goal": "Count groups in intervals sorted by left endpoint; touching intervals merge.",
                "type": "greedy",
                "is_leaf": True,
                "agent_type": "greedy",
                "parent": "S0",
                "dependencies": ["S1"],
                "inputs": [{"name": "sorted_intervals", "type": "list[tuple[int, int]]", "source": "S1.sorted_intervals"}],
                "outputs": [{"name": "group_count", "type": "int"}],
                "interface": {
                    "function_name": "count_merged_groups",
                    "params": [{"name": "sorted_intervals", "type": "list[tuple[int, int]]"}],
                    "return_type": "int",
                    "preconditions": ["input is sorted by (l, r)"],
                    "postconditions": ["touching intervals are in one group"],
                },
            },
        ],
        "execution_order": ["S1", "S2"],
        "validation": {"sample_test_cases": checks, "edge_cases": [], "invariants": ["1 <= answer <= n"]},
        "global_strategy": "Sort once, then greedily extend the current connected component.",
    }


def main() -> dict:
    ir = ensure_catg(_problem_ir())
    sort_code = """def sort_intervals(intervals):
    return sorted(intervals)
"""
    merge_code = """def count_merged_groups(sorted_intervals):
    if not sorted_intervals:
        return 0
    groups = 1
    right = sorted_intervals[0][1]
    for left, current_right in sorted_intervals[1:]:
        if left > right + 1:
            groups += 1
        right = max(right, current_right)
    return groups
"""
    ir = register_code_owner(ir, "S1", sort_code, verification_report={"verdict": "accept"})
    ir = register_code_owner(ir, "S2", merge_code, verification_report={"verdict": "accept"})
    final_code = sort_code + "\n" + merge_code + """

def solve():
    import sys
    values = list(map(int, sys.stdin.buffer.read().split()))
    n = values[0]
    intervals = [(values[i], values[i + 1]) for i in range(1, 2 * n + 1, 2)]
    ordered = sort_intervals(intervals)
    print(count_merged_groups(ordered))

if __name__ == "__main__":
    solve()
"""
    ir = register_assembly_units(ir, final_code)
    initial = {
        "problem_id": "catg_interval_smoke",
        "final_code": final_code,
        "task_plan_ir": ir,
        "code_snippets": {"S1": sort_code, "S2": merge_code},
        "verification_reports": {"S1": {"verdict": "accept"}, "S2": {"verdict": "accept"}},
        "max_repair_attempts": 1,
    }
    result = build_final_verifier_graph().invoke(initial)
    report = result["final_report"]
    summary = {
        "problem": "count connected interval groups",
        "all_passed": report["all_passed"],
        "verdict": report["verdict"],
        "catg": catg_summary(result["task_plan_ir"]),
        "catg_errors": validate_contract_graph(result["task_plan_ir"]),
        "registered_units": result.get("registered_units", {}).get("units", []),
        "development_check_count": len(result.get("development_checks", [])),
        "test_results": report["test_results"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    assert summary["all_passed"] is True
    assert not summary["catg_errors"]
    assert {item["name"] for item in summary["registered_units"]} >= {
        "sort_intervals",
        "count_merged_groups",
        "solve",
        "__main__",
    }
    return summary


if __name__ == "__main__":
    main()
