"""Integration tests for algo_agents — plan_agent → dispatch_algo_agents pipeline.

Usage:
  # 无 LLM 单元测试（任何时候都可以跑）
  pytest tests/test_algo_agents_integration.py

  # 完整集成测试（需要 LLM API Key）
  pytest tests/test_algo_agents_integration.py -m integration

  # CLI demo（打印完整输出）
  python tests/test_algo_agents_integration.py --demo
  python tests/test_algo_agents_integration.py --demo --problem greedy
  python tests/test_algo_agents_integration.py --demo --problem both
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditcoder.algo_agents import dispatch_algo_agents
from auditcoder.algo_agents.agents import (
    build_binary_search_agent_graph,
    build_dp_agent_graph,
    build_greedy_agent_graph,
)
from auditcoder.plan_agent.graph import build_plan_agent_graph

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

GREEDY_INTERVAL_PROBLEM = """\
题目：区间调度最大化
给定 n 个活动，每个活动有开始时间 s[i] 和结束时间 e[i]。
选出最多数量的互不重叠的活动（若一个活动在另一个活动开始时结束则不算重叠）。

输入格式：
第一行一个整数 n (1 <= n <= 100000)。
接下来 n 行，每行两个整数 s[i] e[i]，表示活动的开始和结束时间 (0 <= s[i] < e[i] <= 10^9)。

输出格式：
输出一个整数，表示最多可以选择的活动数量。

样例输入：
3
1 4
2 6
5 7

样例输出：
2

样例解释：选择活动 [1,4] 和 [5,7]，共 2 个。

时间限制：1s
内存限制：256MB
"""

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BASE_IR_SKELETON: dict[str, Any] = {
    "problem": {
        "title": "测试题目",
        "description": "这是一个用于测试的算法题目。",
        "input_format": "第一行一个整数 n。",
        "output_format": "输出一个整数。",
        "constraints": {
            "time_limit": "1s",
            "memory_limit": "256MB",
            "data_ranges": [{"var": "n", "min": 1, "max": 100000}],
        },
        "sample_cases": [{"input": "5\n1 2 3 4 5", "output": "5", "explanation": ""}],
    },
    "analysis": {
        "problem_category": "动态规划",
        "key_observations": ["存在最优子结构"],
        "complexity_target": {"time": "O(n log n)", "space": "O(n)"},
        "algorithm_selection": [],
        "reasoning_chain": [],
    },
    "global_strategy": "使用算法解决该问题",
    "complexity_budget": {"time": "O(n log n)", "space": "O(n)"},
    "root_id": "S0",
    "execution_order": ["S1"],
    "validation": {
        "sample_test_cases": [{"input": "5\n1 2 3 4 5", "expected_output": "5"}],
        "edge_cases": [{"description": "最小输入", "input": "1\n1", "expected_output": "1"}],
        "invariants": [],
    },
}

_BASE_S0_NODE: dict[str, Any] = {
    "id": "S0",
    "goal": "读入数据，调用算法求解，输出结果",
    "type": "orchestration",
    "is_leaf": False,
    "agent_type": None,
    "parent": None,
    "dependencies": ["S1"],
    "inputs": [{"name": "raw_input", "type": "str", "description": "原始输入", "source": None}],
    "outputs": [{"name": "result", "type": "int", "description": "最终结果"}],
    "interface": None,
    "hints": [],
    "complexity_budget": "O(n log n)",
    "context": "根节点",
}

_AGENT_TYPE_GOALS = {
    "dp": "给定整数数组 nums，用动态规划求最长严格递增子序列的长度",
    "greedy": "给定 n 个区间 [s, e]，贪心选出互不重叠的最多活动数",
    "binary_search": "在单调递增数组中二分查找第一个 >= target 的位置（bisect_left 语义）",
}

_AGENT_TYPE_INTERFACES = {
    "dp": {
        "function_name": "lis_length",
        "params": [{"name": "nums", "type": "list[int]", "description": "输入数组"}],
        "return_type": "int",
        "return_description": "最长严格递增子序列的长度",
        "preconditions": ["len(nums) >= 1"],
        "postconditions": ["返回值 >= 1"],
    },
    "greedy": {
        "function_name": "max_non_overlapping",
        "params": [
            {"name": "intervals", "type": "list[tuple[int, int]]", "description": "活动区间列表"}
        ],
        "return_type": "int",
        "return_description": "最多可选互不重叠活动数",
        "preconditions": ["每个区间满足 s < e"],
        "postconditions": ["返回值 >= 0"],
    },
    "binary_search": {
        "function_name": "bisect_left_impl",
        "params": [
            {"name": "arr", "type": "list[int]", "description": "单调递增数组"},
            {"name": "target", "type": "int", "description": "查找目标"},
        ],
        "return_type": "int",
        "return_description": "第一个 >= target 的索引，若全部小于则返回 len(arr)",
        "preconditions": ["arr 单调不减"],
        "postconditions": ["0 <= 返回值 <= len(arr)"],
    },
}


def _make_minimal_ir(agent_type: str, goal: str | None = None) -> dict[str, Any]:
    """Construct a minimal TaskPlanIR dict with a single leaf node of the given agent_type.

    Used by integration tests that bypass plan_agent and test algo agents directly.
    """
    effective_goal = goal or _AGENT_TYPE_GOALS.get(agent_type, f"用 {agent_type} 算法完成任务")
    interface = _AGENT_TYPE_INTERFACES.get(agent_type, {
        "function_name": "solve",
        "params": [{"name": "n", "type": "int", "description": "输入"}],
        "return_type": "int",
        "return_description": "结果",
        "preconditions": [],
        "postconditions": [],
    })

    leaf_node: dict[str, Any] = {
        "id": "S1",
        "goal": effective_goal,
        "type": agent_type,
        "is_leaf": True,
        "agent_type": agent_type,
        "parent": "S0",
        "dependencies": [],
        "inputs": [{"name": "nums", "type": "list[int]", "description": "输入数据", "source": None}],
        "outputs": [{"name": "result", "type": "int", "description": "计算结果"}],
        "interface": interface,
        "hints": list(_get_default_hints(agent_type)),
        "complexity_budget": "O(n log n)",
        "context": f"叶节点，由 {agent_type} agent 实现",
    }

    ir = dict(_BASE_IR_SKELETON)
    ir["task_nodes"] = [dict(_BASE_S0_NODE), leaf_node]
    return ir


def _get_default_hints(agent_type: str) -> list[str]:
    hints_map = {
        "dp": ["维护 tails 数组", "对每个元素用二分查找确定插入位置"],
        "greedy": ["按结束时间升序排序", "每次选结束时间最早的合法活动"],
        "binary_search": ["使用 left <= right 写法", "循环不变量：目标在 [left, right] 内"],
    }
    return hints_map.get(agent_type, [])


def validate_snippets(
    snippets: dict[str, str],
    ir: dict[str, Any],
    *,
    require_all_leaves: bool = True,
) -> list[str]:
    """Validate the code_snippets dict returned by dispatch_algo_agents.

    Returns a list of issue strings (empty means valid).
    """
    issues: list[str] = []

    node_map = {n["id"]: n for n in ir.get("task_nodes", [])}
    leaf_ids = {n["id"] for n in node_map.values() if n.get("is_leaf")}
    known_agent_types = {"dp", "greedy", "binary_search"}
    dispatchable_leaves = {
        nid for nid in leaf_ids
        if node_map[nid].get("agent_type") in known_agent_types
    }

    if require_all_leaves and dispatchable_leaves - snippets.keys():
        missing = dispatchable_leaves - snippets.keys()
        issues.append(f"Missing code snippets for leaf tasks: {missing}")

    for task_id, snippet in snippets.items():
        if task_id not in node_map:
            issues.append(f"Snippet for unknown task_id '{task_id}'")
            continue
        if not snippet or not snippet.strip():
            issues.append(f"Task {task_id}: empty code snippet")
            continue
        if "def " not in snippet:
            issues.append(
                f"Task {task_id}: snippet doesn't contain a function definition "
                f"(first 80 chars: {snippet[:80]!r})"
            )

    return issues


# ---------------------------------------------------------------------------
# Unit tests — no LLM required
# ---------------------------------------------------------------------------

def test_algo_agent_graphs_compile():
    """All three agent graphs should compile without error."""
    dp_graph = build_dp_agent_graph()
    greedy_graph = build_greedy_agent_graph()
    bs_graph = build_binary_search_agent_graph()

    for graph, name in [
        (dp_graph, "dp"),
        (greedy_graph, "greedy"),
        (bs_graph, "binary_search"),
    ]:
        assert graph is not None, f"{name} agent graph should not be None"
        nodes = graph.get_graph().nodes
        assert "code_generator" in nodes, (
            f"{name} agent graph should have a 'code_generator' node, got: {list(nodes)}"
        )


def test_dispatcher_skips_non_leaf():
    """Dispatcher should not invoke any agent for non-leaf nodes."""
    ir = _make_minimal_ir("dp")
    # Mark S1 as non-leaf
    for node in ir["task_nodes"]:
        if node["id"] == "S1":
            node["is_leaf"] = False

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"code_snippet": "def dummy(): pass"}

    with patch(
        "auditcoder.algo_agents.dispatcher.build_dp_agent_graph", return_value=mock_graph
    ), patch(
        "auditcoder.algo_agents.dispatcher.build_greedy_agent_graph", return_value=mock_graph
    ), patch(
        "auditcoder.algo_agents.dispatcher.build_binary_search_agent_graph", return_value=mock_graph
    ):
        code_snippets, verification_reports = dispatch_algo_agents(ir)

    assert code_snippets == {}, f"No snippets expected for non-leaf nodes, got: {code_snippets}"
    assert verification_reports == {}
    mock_graph.invoke.assert_not_called()


def test_dispatcher_skips_unknown_agent_type():
    """Dispatcher should silently skip leaf nodes with unregistered agent_type."""
    ir = _make_minimal_ir("dp")
    for node in ir["task_nodes"]:
        if node["id"] == "S1":
            node["agent_type"] = "unsupported_agent"

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"code_snippet": "def dummy(): pass"}

    with patch(
        "auditcoder.algo_agents.dispatcher.build_dp_agent_graph", return_value=mock_graph
    ), patch(
        "auditcoder.algo_agents.dispatcher.build_greedy_agent_graph", return_value=mock_graph
    ), patch(
        "auditcoder.algo_agents.dispatcher.build_binary_search_agent_graph", return_value=mock_graph
    ):
        code_snippets, verification_reports = dispatch_algo_agents(ir)

    assert code_snippets == {}, f"Unknown agent_type should be skipped, got: {code_snippets}"
    assert verification_reports == {}
    mock_graph.invoke.assert_not_called()


def test_dispatcher_respects_execution_order():
    """Dispatcher should call agents in execution_order sequence."""
    ir = _make_minimal_ir("dp")
    # Add a second leaf node to verify ordering
    ir["task_nodes"].append({
        "id": "S2",
        "goal": "第二个任务",
        "type": "binary_search",
        "is_leaf": True,
        "agent_type": "binary_search",
        "parent": "S0",
        "dependencies": [],
        "inputs": [],
        "outputs": [],
        "interface": {
            "function_name": "search",
            "params": [],
            "return_type": "int",
            "preconditions": [],
            "postconditions": [],
        },
        "hints": [],
        "complexity_budget": "O(log n)",
        "context": "",
    })
    ir["execution_order"] = ["S1", "S2"]

    call_order: list[str] = []

    def _make_mock(task_id: str):
        m = MagicMock()
        def _invoke(state, **_):
            call_order.append(state["task_node"]["id"])
            return {"code_snippet": f"def f_{task_id}(): pass"}
        m.invoke.side_effect = _invoke
        return m

    dp_mock = _make_mock("S1")
    bs_mock = _make_mock("S2")

    mock_verifier = MagicMock()
    mock_verifier.invoke.return_value = {
        "verification_report": {"verdict": "accept", "summary": ""}
    }

    with patch(
        "auditcoder.algo_agents.dispatcher.build_dp_agent_graph", return_value=dp_mock
    ), patch(
        "auditcoder.algo_agents.dispatcher.build_greedy_agent_graph", return_value=MagicMock()
    ), patch(
        "auditcoder.algo_agents.dispatcher.build_binary_search_agent_graph", return_value=bs_mock
    ), patch(
        "auditcoder.algo_agents.dispatcher.build_verifier_agent_graph", return_value=mock_verifier
    ):
        code_snippets, verification_reports = dispatch_algo_agents(ir)

    assert call_order == ["S1", "S2"], (
        f"Agents should be called in execution_order, got: {call_order}"
    )
    assert set(code_snippets.keys()) == {"S1", "S2"}
    assert set(verification_reports.keys()) == {"S1", "S2"}


def test_dispatcher_does_not_release_still_rejected_snippet():
    ir = _make_minimal_ir("dp")
    mock_agent = MagicMock()
    mock_agent.invoke.side_effect = [
        {"code_snippet": "def lis_length(nums): return 0"},
        {"code_snippet": "def lis_length(nums): return -1"},
        {"code_snippet": "def lis_length(nums): return -2"},
    ]
    mock_verifier = MagicMock()
    mock_verifier.invoke.side_effect = [
        {"verification_report": {"verdict": "reject", "summary": "bad"}},
        {"verification_report": {"verdict": "reject", "summary": "still bad"}},
        {"verification_report": {"verdict": "reject", "summary": "still bad"}},
    ]

    with patch(
        "auditcoder.algo_agents.dispatcher.build_dp_agent_graph",
        return_value=mock_agent,
    ), patch(
        "auditcoder.algo_agents.dispatcher.build_verifier_agent_graph",
        return_value=mock_verifier,
    ):
        snippets, reports = dispatch_algo_agents(ir)

    assert snippets == {}
    assert reports["S1"]["verdict"] == "reject"
    assert reports["S1"]["release_to_assembler"] is False
    assert reports["S1"]["generation_attempts"] == 3
    assert mock_agent.invoke.call_count == 3
    verifier_input = mock_verifier.invoke.call_args_list[0].args[0]
    assert verifier_input["task_contract"]["interface"]["function_name"] == "lis_length"


# ---------------------------------------------------------------------------
# Integration tests — require LLM API Key
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_algo_agent_dp_only():
    """DP agent should generate a Python function for a minimal DP task."""
    ir = _make_minimal_ir("dp")
    code_snippets, verification_reports = dispatch_algo_agents(ir)

    issues = validate_snippets(code_snippets, ir)
    assert not issues, f"Snippet validation issues: {issues}"
    assert "S1" in code_snippets, "S1 (dp leaf) should produce a code snippet"
    assert "S1" in verification_reports, "S1 should have a verification report"

    snippet = code_snippets["S1"]
    print(f"\n[dp agent output]\n{snippet}")
    print(f"[dp verifier verdict] {verification_reports['S1'].get('verdict', '?')}")


@pytest.mark.integration
def test_algo_agent_greedy_only():
    """Greedy agent should generate a Python function for a minimal greedy task."""
    ir = _make_minimal_ir("greedy")
    code_snippets, verification_reports = dispatch_algo_agents(ir)

    issues = validate_snippets(code_snippets, ir)
    assert not issues, f"Snippet validation issues: {issues}"
    assert "S1" in code_snippets, "S1 (greedy leaf) should produce a code snippet"

    snippet = code_snippets["S1"]
    print(f"\n[greedy agent output]\n{snippet}")
    print(f"[greedy verifier verdict] {verification_reports.get('S1', {}).get('verdict', '?')}")


@pytest.mark.integration
def test_algo_agent_binary_search_only():
    """Binary search agent should generate a Python function for a minimal BS task."""
    ir = _make_minimal_ir("binary_search")
    code_snippets, verification_reports = dispatch_algo_agents(ir)

    issues = validate_snippets(code_snippets, ir)
    assert not issues, f"Snippet validation issues: {issues}"
    assert "S1" in code_snippets, "S1 (binary_search leaf) should produce a code snippet"

    snippet = code_snippets["S1"]
    print(f"\n[binary_search agent output]\n{snippet}")
    print(f"[binary_search verifier verdict] {verification_reports.get('S1', {}).get('verdict', '?')}")


@pytest.mark.integration
def test_full_pipeline_lis_with_algo_agents():
    """End-to-end: plan_agent decomposes LIS → algo agents generate all leaf code."""
    # Step 1: plan_agent produces task_plan_ir
    plan_result = build_plan_agent_graph().invoke(
        {"problem_raw": LIS_PROBLEM, "revision_count": 0}
    )
    assert "task_plan_ir" in plan_result, "plan_agent should produce task_plan_ir"

    ir = plan_result["task_plan_ir"]
    leaf_nodes = [n for n in ir["task_nodes"] if n.get("is_leaf")]
    assert len(leaf_nodes) >= 1, "LIS problem should have at least one leaf task"

    # Step 2: algo agents generate code snippets for each leaf
    code_snippets, verification_reports = dispatch_algo_agents(ir)

    assert len(code_snippets) > 0, (
        "dispatch_algo_agents should return at least one code snippet"
    )

    issues = validate_snippets(code_snippets, ir, require_all_leaves=False)
    assert not issues, f"Snippet validation issues: {issues}"

    print(f"\n[Full pipeline — LIS] {len(code_snippets)} snippets generated:")
    for task_id, snippet in code_snippets.items():
        node = next(n for n in ir["task_nodes"] if n["id"] == task_id)
        verdict = verification_reports.get(task_id, {}).get("verdict", "?")
        print(f"\n  Task {task_id} [{node.get('agent_type')}]: {node.get('goal')} (verifier: {verdict})")
        print(f"  --- code ---\n{snippet}\n  ------------")


# ---------------------------------------------------------------------------
# CLI demo runner
# ---------------------------------------------------------------------------

def demo_full_pipeline(problem: str, label: str) -> None:
    """Run plan_agent + algo_agents and print every generated code snippet."""
    print("=" * 70)
    print(f"End-to-End Demo — {label}")
    print("=" * 70)

    print("\n[Step 1] Running plan_agent ...")
    plan_result = build_plan_agent_graph().invoke(
        {"problem_raw": problem, "revision_count": 0}
    )
    ir = plan_result.get("task_plan_ir", {})

    print(f"  Global strategy: {ir.get('global_strategy', '')}")
    print(f"  Execution order: {ir.get('execution_order', [])}")

    task_nodes = ir.get("task_nodes", [])
    leaf_nodes = [n for n in task_nodes if n.get("is_leaf")]
    print(f"\n  Task tree: {len(task_nodes)} nodes, {len(leaf_nodes)} leaves")
    for node in task_nodes:
        tag = "[LEAF]" if node.get("is_leaf") else "[INTERNAL]"
        agent = f" -> {node.get('agent_type')}" if node.get("agent_type") else ""
        print(f"    {node['id']} {tag}{agent}  {node.get('goal', '')}")

    print("\n[Step 2] Dispatching to algo agents (with per-task verification + retry) ...")
    code_snippets, verification_reports = dispatch_algo_agents(ir)
    print(f"  Generated {len(code_snippets)} code snippet(s)")
    for task_id, report in verification_reports.items():
        print(f"  Task {task_id}: verifier verdict = {report.get('verdict', '?')}")

    if not code_snippets:
        print("  (no snippets produced — check agent_type values in task_nodes)")
    else:
        for task_id, snippet in code_snippets.items():
            node = next((n for n in task_nodes if n["id"] == task_id), {})
            print(f"\n{'─' * 60}")
            print(f"  Task: {task_id}  [{node.get('agent_type')}]")
            print(f"  Goal: {node.get('goal', '')}")
            print(f"{'─' * 60}")
            print(snippet)

    issues = validate_snippets(code_snippets, ir, require_all_leaves=False)
    print(f"\n{'─' * 60}")
    if issues:
        print(f"Validation issues ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("Validation: PASSED")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Algo Agents 集成测试与演示")
    parser.add_argument(
        "--demo", action="store_true",
        help="运行完整端到端演示（需要 LLM API Key）",
    )
    parser.add_argument(
        "--problem", choices=["lis", "greedy", "both"],
        default="lis",
        help="选择测试题目 (lis=最长递增子序列, greedy=区间调度, both=两个都跑)",
    )
    args = parser.parse_args()

    if args.demo:
        if args.problem in ("lis", "both"):
            demo_full_pipeline(LIS_PROBLEM, "最长递增子序列 (LIS) — plan+dp+binary_search")
        if args.problem in ("greedy", "both"):
            demo_full_pipeline(GREEDY_INTERVAL_PROBLEM, "区间调度最大化 — plan+greedy")
    else:
        print("Use --demo to run the full pipeline.")
        print("Running smoke test (no LLM) ...")

        # Smoke: graph compilation
        build_dp_agent_graph()
        build_greedy_agent_graph()
        build_binary_search_agent_graph()
        print("  graph compilation: OK")

        # Smoke: minimal IR construction
        for at in ("dp", "greedy", "binary_search"):
            ir = _make_minimal_ir(at)
            assert ir["task_nodes"][1]["agent_type"] == at
        print("  _make_minimal_ir: OK")

        # Smoke: validate_snippets on empty
        issues = validate_snippets({}, _make_minimal_ir("dp"), require_all_leaves=False)
        assert issues == []
        print("  validate_snippets: OK")

        print("All smoke tests passed.")
