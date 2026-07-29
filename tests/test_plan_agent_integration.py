"""Integration tests for the Plan Agent — runs the full pipeline with LLM.

These tests require a valid LLM API Key configured in .env.
Run with:  pytest tests/test_plan_agent_integration.py -m integration
   or:     python tests/test_plan_agent_integration.py --demo

   python tests/test_plan_agent_integration.py --demo --problem lis
python tests/test_plan_agent_integration.py --demo --problem dp_rmq
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditcoder.plan_agent.graph import build_plan_agent_graph
from auditcoder.schemas.task_plan_ir import TaskPlanIR


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def validate_ir(ir_dict: dict) -> list[str]:
    """Validate a TaskPlanIR dict for structural correctness.

    Returns a list of issue strings (empty means valid).
    """
    issues: list[str] = []

    # 1. Pydantic validation
    try:
        TaskPlanIR.model_validate(ir_dict)
    except Exception as e:
        issues.append(f"Pydantic validation failed: {e}")
        return issues

    nodes = ir_dict.get("task_nodes", [])
    node_map = {n["id"]: n for n in nodes}

    # 2. root_id exists
    root_id = ir_dict.get("root_id", "")
    if root_id not in node_map:
        issues.append(f"root_id '{root_id}' not found in task_nodes")

    # 3. parent references valid
    for node in nodes:
        parent = node.get("parent")
        if parent is not None and parent not in node_map:
            issues.append(f"Node {node['id']}: parent '{parent}' not found")

    # 4. No orphan nodes (besides root)
    for node in nodes:
        if node["id"] != root_id and node.get("parent") is None:
            issues.append(f"Node {node['id']}: orphan (no parent, not root)")

    # 5. dependencies form a DAG (cycle detection via DFS)
    adj: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        for dep in node.get("dependencies", []):
            adj[node["id"]].append(dep)
            if dep not in node_map:
                issues.append(f"Node {node['id']}: dependency '{dep}' not found")

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n["id"]: WHITE for n in nodes}

    def has_cycle(nid: str) -> bool:
        color[nid] = GRAY
        for neighbor in adj.get(nid, []):
            if neighbor not in color:
                continue
            if color[neighbor] == GRAY:
                return True
            if color[neighbor] == WHITE and has_cycle(neighbor):
                return True
        color[nid] = BLACK
        return False

    for nid in color:
        if color[nid] == WHITE:
            if has_cycle(nid):
                issues.append("Dependency graph contains a cycle")
                break

    # 6. Leaf node completeness
    for node in nodes:
        if node.get("is_leaf"):
            if not node.get("agent_type"):
                issues.append(f"Leaf node {node['id']}: missing agent_type")
            if not node.get("interface"):
                issues.append(f"Leaf node {node['id']}: missing interface")

    # 7. DataSlot source references
    output_registry: dict[str, set[str]] = {}
    for node in nodes:
        output_registry[node["id"]] = {
            o["name"] for o in node.get("outputs", [])
        }

    for node in nodes:
        for inp in node.get("inputs", []):
            source = inp.get("source")
            if source is None:
                continue
            parts = source.split(".")
            if len(parts) != 2:
                issues.append(
                    f"Node {node['id']}: input '{inp['name']}' has "
                    f"invalid source format '{source}' (expected 'ID.name')"
                )
                continue
            ref_id, ref_name = parts
            if ref_id not in node_map:
                issues.append(
                    f"Node {node['id']}: source '{source}' references "
                    f"unknown task '{ref_id}'"
                )
            elif ref_name not in output_registry.get(ref_id, set()):
                issues.append(
                    f"Node {node['id']}: source '{source}' references "
                    f"unknown output '{ref_name}' on task '{ref_id}'"
                )

    # 8. execution_order is valid topo sort of leaf nodes
    exec_order = ir_dict.get("execution_order", [])
    leaf_ids = {n["id"] for n in nodes if n.get("is_leaf")}
    if leaf_ids and set(exec_order) != leaf_ids:
        issues.append(
            f"execution_order {exec_order} doesn't match "
            f"leaf node ids {leaf_ids}"
        )
    else:
        completed = set()
        for tid in exec_order:
            if tid in node_map:
                for dep in node_map[tid].get("dependencies", []):
                    if dep in leaf_ids and dep not in completed:
                        issues.append(
                            f"execution_order: {tid} before dependency {dep}"
                        )
            completed.add(tid)

    return issues


# ---------------------------------------------------------------------------
# Test problems
# ---------------------------------------------------------------------------

LIS_PROBLEM = """\
题目：最长递增子序列
给定一个整数数组 nums，找到其中最长严格递增子序列的长度。

输入格式：
第一行一个整数 n (1 <= n <= 100000)，表示数组长度。
第二行 n 个整数，表示数组 nums，其中 -10^9 <= nums[i] <= 10^9。

输出格式：
输出一个整数，表示最长严格递增子序列的长度。

样例输入：
8
10 9 2 5 3 7 101 18

样例输出：
4

样例解释：最长递增子序列为 [2, 3, 7, 101]，长度为 4。

时间限制：1s
内存限制：256MB
"""

DP_RMQ_PROBLEM = """\
题目：最优区间划分
给定长度为 n 的数组 a 和正整数 k。
将数组划分为若干连续子数组，每个子数组中不同元素的种类不超过 k。
每个子数组的代价为该子数组中的最大值。总代价为所有子数组代价之和。
求最小总代价。

输入格式：
第一行两个整数 n 和 k (1 <= k <= n <= 100000)。
第二行 n 个整数 a_1, a_2, ..., a_n (1 <= a_i <= 10^9)。

输出格式：
一个整数，表示最小总代价。

样例输入：
5 2
3 1 4 1 5

样例输出：
8

样例解释：划分为 [3,1] [4,1] [5]，代价 = 3 + 4 + 5 = 12；
或 [3,1,4,1] [5]（不同元素 3,1,4 超过 k=2，不合法）。
最优划分为 [3,1] [4,1] [5]，代价 = 3 + 4 + 5 = 12。

时间限制：2s
内存限制：256MB
"""


# ---------------------------------------------------------------------------
# Integration tests (marked so they can be skipped in CI without API key)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_full_pipeline_lis():
    """Run the full Plan Agent pipeline on the LIS problem."""
    app = build_plan_agent_graph()
    result = app.invoke({"problem_raw": LIS_PROBLEM, "revision_count": 0})

    assert "task_plan_ir" in result, "Pipeline should produce task_plan_ir"

    ir = result["task_plan_ir"]
    issues = validate_ir(ir)
    assert not issues, f"IR validation issues: {issues}"

    review = result.get("review_result", {})
    assert review.get("status") == "pass", f"Review failed: {review}"


@pytest.mark.integration
def test_full_pipeline_dp_rmq():
    """Run the full Plan Agent pipeline on the DP+RMQ problem.

    This problem should trigger multi-level decomposition with diverse agent types.
    """
    app = build_plan_agent_graph()
    result = app.invoke({"problem_raw": DP_RMQ_PROBLEM, "revision_count": 0})

    assert "task_plan_ir" in result, "Pipeline should produce task_plan_ir"

    ir = result["task_plan_ir"]
    issues = validate_ir(ir)
    assert not issues, f"IR validation issues: {issues}"

    nodes = ir["task_nodes"]
    leaf_nodes = [n for n in nodes if n.get("is_leaf")]
    agent_types = {n.get("agent_type") for n in leaf_nodes}

    assert len(nodes) >= 3, "Should have at least root + 2 task nodes"
    assert len(agent_types) >= 2, (
        f"Should have at least 2 different agent types, got: {agent_types}"
    )

    max_depth = 0
    for node in nodes:
        depth = node["id"].count("_")
        max_depth = max(max_depth, depth)
    assert max_depth >= 1, "Task tree should have depth >= 2 (at least S1_x nodes)"


# ---------------------------------------------------------------------------
# CLI demo runner
# ---------------------------------------------------------------------------

def demo_plan_agent_output(problem: str, label: str):
    """Run Plan Agent and print the TaskPlanIR output."""
    print("=" * 60)
    print(f"Plan Agent Demo — {label}")
    print("=" * 60)

    app = build_plan_agent_graph()
    result = app.invoke({"problem_raw": problem, "revision_count": 0})

    ir = result.get("task_plan_ir", {})
    print("\nTaskPlanIR output:\n")
    print(json.dumps(ir, ensure_ascii=False, indent=2))

    print("\n" + "-" * 40)
    issues = validate_ir(ir)
    if issues:
        print(f"Validation issues ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("Validation: PASSED")

    review = result.get("review_result", {})
    print(f"Review status: {review.get('status', 'unknown')}")
    if review.get("reason"):
        print(f"Review reason: {review['reason']}")
    print(f"Revision count: {result.get('revision_count', 0)}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plan Agent 集成测试与演示")
    parser.add_argument("--demo", action="store_true",
                        help="运行完整演示（需要 LLM API Key）")
    parser.add_argument("--problem", choices=["lis", "dp_rmq", "both"],
                        default="both", help="选择测试题目")
    args = parser.parse_args()

    if args.demo:
        if args.problem in ("lis", "both"):
            demo_plan_agent_output(LIS_PROBLEM, "最长递增子序列 (LIS)")
        if args.problem in ("dp_rmq", "both"):
            demo_plan_agent_output(DP_RMQ_PROBLEM, "最优区间划分 (DP+RMQ)")
    else:
        print("Use --demo to run the full pipeline. Running validate_ir smoke test...")
        minimal_ir = {
            "problem": {
                "title": "Test", "description": "d", "input_format": "i",
                "output_format": "o", "constraints": {}, "sample_cases": [],
            },
            "analysis": {
                "problem_category": "test", "key_observations": [],
                "complexity_target": {"time": "O(n)", "space": "O(n)"},
                "algorithm_selection": [], "reasoning_chain": [],
            },
            "global_strategy": "test",
            "complexity_budget": {"time": "O(n)", "space": "O(n)"},
            "task_nodes": [
                {
                    "id": "S0", "goal": "root", "type": "orchestration",
                    "is_leaf": False, "agent_type": None, "parent": None,
                    "dependencies": ["S1"], "inputs": [], "outputs": [],
                    "interface": None, "hints": [], "complexity_budget": "",
                    "context": "",
                },
                {
                    "id": "S1", "goal": "solve", "type": "dp",
                    "is_leaf": True, "agent_type": "dp", "parent": "S0",
                    "dependencies": [], "inputs": [], "outputs": [],
                    "interface": {
                        "function_name": "solve",
                        "params": [], "return_type": "int",
                    },
                    "hints": [], "complexity_budget": "O(n)", "context": "",
                },
            ],
            "root_id": "S0",
            "execution_order": ["S1"],
            "validation": {
                "sample_test_cases": [{"input": "1", "expected_output": "1"}],
                "edge_cases": [
                    {"description": "min", "input": "1", "expected_output": "1"},
                    {"description": "max", "input": "100000", "expected_output": "1"},
                ],
                "invariants": [],
            },
        }
        issues = validate_ir(minimal_ir)
        assert not issues, f"Smoke test failed: {issues}"
        print("validate_ir smoke test passed.")
