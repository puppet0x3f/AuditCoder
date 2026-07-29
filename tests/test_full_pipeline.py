"""Full pipeline integration tests — plan_agent → algo_agents → assembler_agent → final_verifier_agent.

Covers the end-to-end flow from a raw problem statement to a complete,
runnable Python solution with final verification.

Usage:
  # 无 LLM 单元测试（任何时候都可以跑）
  pytest tests/test_full_pipeline.py

  # 完整集成测试（需要 LLM API Key）
  pytest tests/test_full_pipeline.py -m integration

  # CLI demo（打印完整输出代码）
  python tests/test_full_pipeline.py --demo
  python tests/test_full_pipeline.py --demo --problem greedy
  python tests/test_full_pipeline.py --demo --problem knapsack
  python tests/test_full_pipeline.py --demo --problem all
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage

from auditcoder.algo_agents import dispatch_algo_agents
from auditcoder.assembler_agent import AssemblerState, build_assembler_agent_graph
from auditcoder.final_verifier_agent import build_final_verifier_graph
from auditcoder.plan_agent.graph import build_plan_agent_graph

# ---------------------------------------------------------------------------
# Re-export shared test fixtures from the existing integration test module
# ---------------------------------------------------------------------------

from tests.test_algo_agents_integration import (
    GREEDY_INTERVAL_PROBLEM,
    LIS_PROBLEM,
    _make_minimal_ir,
    validate_snippets,
)

# ---------------------------------------------------------------------------
# Additional test problem: 0/1 Knapsack (black-box end-to-end)
# ---------------------------------------------------------------------------

KNAPSACK_PROBLEM = """\
题目：0/1 背包问题
给定 n 个物品和一个容量为 W 的背包。每个物品有重量 w[i] 和价值 v[i]。
每个物品只能选一次。求在不超过背包容量的前提下，能取得的最大价值。

输入格式：
第一行两个整数 n W (1 <= n <= 1000, 1 <= W <= 10000)。
接下来 n 行，每行两个整数 w[i] v[i] (1 <= w[i] <= W, 1 <= v[i] <= 10000)。

输出格式：
输出一个整数，表示背包可装下的最大价值。

样例输入：
4 5
2 3
3 4
4 5
5 6

样例输出：
7

样例解释：选择物品1（重2，值3）和物品2（重3，值4），总重5，总价值7。

时间限制：1s
内存限制：256MB
"""

# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def validate_final_code(
    final_code: str,
    ir: dict[str, Any],
    *,
    require_main_block: bool = True,
) -> list[str]:
    """Validate the assembled final_code string.

    Returns a list of issue strings (empty list means valid).
    """
    issues: list[str] = []

    if not final_code or not final_code.strip():
        issues.append("final_code is empty")
        return issues

    if "def " not in final_code:
        issues.append("final_code contains no function definitions")

    if require_main_block and 'if __name__ == "__main__"' not in final_code:
        issues.append('final_code missing `if __name__ == "__main__":` block')

    try:
        compile(final_code, "<final_code>", "exec")
    except SyntaxError as exc:
        issues.append(f"final_code has SyntaxError: {exc}")

    node_map = {n["id"]: n for n in ir.get("task_nodes", [])}
    for task_id in ir.get("execution_order", []):
        node = node_map.get(task_id)
        if not node or not node.get("is_leaf"):
            continue
        interface = node.get("interface")
        if not interface:
            continue
        func_name = interface.get("function_name", "")
        if func_name and func_name not in final_code:
            issues.append(
                f"final_code missing function '{func_name}' (from task {task_id})"
            )

    return issues


# ---------------------------------------------------------------------------
# Unit tests — no LLM required
# ---------------------------------------------------------------------------


def test_assembler_graph_compiles():
    """Assembler agent graph should compile without error."""
    graph = build_assembler_agent_graph()
    assert graph is not None, "assembler graph should not be None"
    nodes = graph.get_graph().nodes
    assert "code_assembler" in nodes, (
        f"assembler graph should have a 'code_assembler' node, got: {list(nodes)}"
    )


def test_assembler_with_mock_llm():
    """Assembler should return final_code when LLM is mocked."""
    ir = _make_minimal_ir("dp")
    code_snippets = {
        "S1": (
            "def lis_length(nums: list[int]) -> int:\n"
            "    tails = []\n"
            "    import bisect\n"
            "    for x in nums:\n"
            "        pos = bisect.bisect_left(tails, x)\n"
            "        if pos == len(tails):\n"
            "            tails.append(x)\n"
            "        else:\n"
            "            tails[pos] = x\n"
            "    return len(tails)\n"
        )
    }

    expected_final = (
        "def lis_length(nums):\n"
        "    import bisect\n"
        "    tails = []\n"
        "    for x in nums:\n"
        "        pos = bisect.bisect_left(tails, x)\n"
        "        if pos == len(tails): tails.append(x)\n"
        "        else: tails[pos] = x\n"
        "    return len(tails)\n\n"
        "def solve():\n"
        "    n = int(input())\n"
        "    nums = list(map(int, input().split()))\n"
        "    print(lis_length(nums))\n\n"
        'if __name__ == "__main__":\n'
        "    solve()\n"
    )

    mock_response = AIMessage(content=f'{{"final_code": {repr(expected_final)}}}')

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    with patch("auditcoder.assembler_agent.agent.get_llm", return_value=mock_llm):
        graph = build_assembler_agent_graph()
        result = graph.invoke(
            {
                "task_plan_ir": ir,
                "code_snippets": code_snippets,
            }
        )

    assert "final_code" in result, "result should contain 'final_code'"
    assert isinstance(result["final_code"], str), "final_code should be a string"
    assert result["final_code"].strip(), "final_code should not be empty"
    assert "def " in result["final_code"], "final_code should contain a function definition"

    mock_llm.invoke.assert_called_once()


def test_assembler_passes_all_snippets_to_llm():
    """Assembler should include all snippet content in the LLM prompt."""
    ir = _make_minimal_ir("greedy")
    code_snippets = {
        "S1": "def max_non_overlapping(intervals): return 0"
    }

    captured_messages: list = []

    mock_response = AIMessage(
        content='{"final_code": "def max_non_overlapping(intervals): return 0\\n\\ndef solve():\\n    pass\\n\\nif __name__ == \\"__main__\\":\\n    solve()\\n"}'
    )

    mock_llm = MagicMock()

    def _capture_invoke(messages, **_):
        captured_messages.extend(messages)
        return mock_response

    mock_llm.invoke.side_effect = _capture_invoke

    with patch("auditcoder.assembler_agent.agent.get_llm", return_value=mock_llm):
        build_assembler_agent_graph().invoke(
            {"task_plan_ir": ir, "code_snippets": code_snippets}
        )

    assert captured_messages, "LLM should have been called with messages"
    full_prompt = " ".join(
        m.content for m in captured_messages if hasattr(m, "content")
    )
    assert "max_non_overlapping" in full_prompt, (
        "Prompt should include the snippet's function name"
    )
    assert "S1" in full_prompt, "Prompt should reference the task ID"


# ---------------------------------------------------------------------------
# Integration tests — require LLM API Key
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_pipeline_lis():
    """End-to-end: LIS problem → plan_agent → algo_agents → assembler → final_verifier."""
    plan_result = build_plan_agent_graph().invoke(
        {"problem_raw": LIS_PROBLEM, "revision_count": 0}
    )
    assert "task_plan_ir" in plan_result, "plan_agent should produce task_plan_ir"
    ir = plan_result["task_plan_ir"]

    code_snippets, verification_reports = dispatch_algo_agents(ir)
    assert len(code_snippets) > 0, "algo_agents should produce at least one snippet"

    snippet_issues = validate_snippets(code_snippets, ir, require_all_leaves=False)
    assert not snippet_issues, f"Snippet validation issues: {snippet_issues}"

    assembler_result = build_assembler_agent_graph().invoke(
        {"task_plan_ir": ir, "code_snippets": code_snippets}
    )
    assert "final_code" in assembler_result, "assembler should produce final_code"

    final_code = assembler_result["final_code"]
    issues = validate_final_code(final_code, ir)
    assert not issues, f"final_code validation issues: {issues}"

    final_result = build_final_verifier_graph().invoke({
        "final_code": final_code,
        "task_plan_ir": ir,
        "code_snippets": code_snippets,
        "verification_reports": verification_reports,
    })
    final_report = final_result.get("final_report", {})
    assert "verdict" in final_report, "final_report should contain 'verdict'"
    assert final_report["verdict"] in ("accept", "reject")

    print(f"\n{'=' * 70}")
    print("[Full pipeline — LIS] final_code:")
    print("=" * 70)
    print(final_code)
    print(f"\nFinal verdict: {final_report['verdict']}")
    if final_report.get("responsible_agent"):
        print(f"Responsible agent: {final_report['responsible_agent']}")
    print("=" * 70)


@pytest.mark.integration
def test_full_pipeline_greedy():
    """End-to-end: Greedy interval scheduling → plan_agent → algo_agents → assembler → final_verifier."""
    plan_result = build_plan_agent_graph().invoke(
        {"problem_raw": GREEDY_INTERVAL_PROBLEM, "revision_count": 0}
    )
    assert "task_plan_ir" in plan_result
    ir = plan_result["task_plan_ir"]

    code_snippets, verification_reports = dispatch_algo_agents(ir)
    assert len(code_snippets) > 0, "algo_agents should produce at least one snippet"

    assembler_result = build_assembler_agent_graph().invoke(
        {"task_plan_ir": ir, "code_snippets": code_snippets}
    )
    final_code = assembler_result.get("final_code", "")
    issues = validate_final_code(final_code, ir)
    assert not issues, f"final_code validation issues: {issues}"

    final_result = build_final_verifier_graph().invoke({
        "final_code": final_code,
        "task_plan_ir": ir,
        "code_snippets": code_snippets,
        "verification_reports": verification_reports,
    })
    final_report = final_result.get("final_report", {})
    assert "verdict" in final_report

    print(f"\n{'=' * 70}")
    print("[Full pipeline — Greedy Interval] final_code:")
    print("=" * 70)
    print(final_code)
    print(f"\nFinal verdict: {final_report['verdict']}")
    if final_report.get("responsible_agent"):
        print(f"Responsible agent: {final_report['responsible_agent']}")
    print("=" * 70)


@pytest.mark.integration
def test_full_pipeline_from_raw_problem():
    """Black-box end-to-end: raw problem string → complete runnable Python code.

    Uses a 0/1 Knapsack problem not seen in other tests to exercise the full
    pipeline purely from user input to final output.
    """
    plan_result = build_plan_agent_graph().invoke(
        {"problem_raw": KNAPSACK_PROBLEM, "revision_count": 0}
    )
    assert "task_plan_ir" in plan_result, "plan_agent should produce task_plan_ir"
    ir = plan_result["task_plan_ir"]

    leaf_nodes = [n for n in ir.get("task_nodes", []) if n.get("is_leaf")]
    assert len(leaf_nodes) >= 1, "Knapsack problem should have at least one leaf task"

    code_snippets, verification_reports = dispatch_algo_agents(ir)
    assert len(code_snippets) > 0, "algo_agents should produce at least one snippet"

    assembler_result = build_assembler_agent_graph().invoke(
        {"task_plan_ir": ir, "code_snippets": code_snippets}
    )
    final_code = assembler_result.get("final_code", "")

    assert final_code and final_code.strip(), "final_code should not be empty"
    try:
        compile(final_code, "<knapsack_final_code>", "exec")
    except SyntaxError as exc:
        pytest.fail(f"final_code has SyntaxError: {exc}")

    final_result = build_final_verifier_graph().invoke({
        "final_code": final_code,
        "task_plan_ir": ir,
        "code_snippets": code_snippets,
        "verification_reports": verification_reports,
    })
    final_report = final_result.get("final_report", {})
    assert "verdict" in final_report

    print(f"\n{'=' * 70}")
    print("[Full pipeline — 0/1 Knapsack] final_code:")
    print("=" * 70)
    print(final_code)
    print(f"\nFinal verdict: {final_report['verdict']}")
    if final_report.get("responsible_agent"):
        print(f"Responsible agent: {final_report['responsible_agent']}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI demo runner
# ---------------------------------------------------------------------------


def demo_full_pipeline(problem: str, label: str) -> None:
    """Run plan_agent + algo_agents + assembler_agent + final_verifier and print results."""
    print("\n" + "=" * 70)
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
    print(f"\n  Task tree: {len(task_nodes)} nodes, {len(leaf_nodes)} leaf(s)")
    for node in task_nodes:
        tag = "[LEAF]" if node.get("is_leaf") else "[INTERNAL]"
        agent = f" → {node.get('agent_type')}" if node.get("agent_type") else ""
        print(f"    {node['id']} {tag}{agent}  {node.get('goal', '')}")

    print("\n[Step 2] Dispatching to algo agents (with per-task verification + retry) ...")
    code_snippets, verification_reports = dispatch_algo_agents(ir)
    print(f"  Generated {len(code_snippets)} code snippet(s)")

    for task_id, report in verification_reports.items():
        verdict = report.get("verdict", "unknown")
        print(f"  Task {task_id}: verifier verdict = {verdict}")

    if not code_snippets:
        print("  (no snippets produced — check agent_type values in task_nodes)")
    else:
        for task_id, snippet in code_snippets.items():
            node = next((n for n in task_nodes if n["id"] == task_id), {})
            print(f"\n{'─' * 60}")
            print(f"  Task : {task_id}  [{node.get('agent_type')}]")
            print(f"  Goal : {node.get('goal', '')}")
            print(f"{'─' * 60}")
            print(snippet)

    print("\n[Step 3] Running assembler_agent ...")
    assembler_result = build_assembler_agent_graph().invoke(
        {"task_plan_ir": ir, "code_snippets": code_snippets}
    )
    final_code = assembler_result.get("final_code", "")

    print(f"\n{'═' * 70}")
    print("  FINAL CODE (complete runnable Python solution):")
    print(f"{'═' * 70}")
    print(final_code)

    issues = validate_final_code(final_code, ir)
    print(f"\n{'─' * 70}")
    if issues:
        print(f"Validation issues ({len(issues)}):")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("Validation: PASSED")

    print("\n[Step 4] Running final_verifier_agent ...")
    final_result = build_final_verifier_graph().invoke({
        "final_code": final_code,
        "task_plan_ir": ir,
        "code_snippets": code_snippets,
        "verification_reports": verification_reports,
    })
    final_report = final_result.get("final_report", {})
    print(f"  Final verdict  : {final_report.get('verdict', 'unknown')}")
    print(f"  Error type     : {final_report.get('error_type', 'none')}")
    if final_report.get("responsible_agent"):
        ra = final_report["responsible_agent"]
        print(f"  Responsible    : task={ra.get('task_id')}  agent={ra.get('agent_type')}  error={ra.get('error_type')}")
        print(f"  Detail         : {ra.get('detail', '')[:200]}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    _PROBLEMS = {
        "lis": (LIS_PROBLEM, "最长递增子序列 (LIS) — plan + dp/binary_search + assembler"),
        "greedy": (GREEDY_INTERVAL_PROBLEM, "区间调度最大化 — plan + greedy + assembler"),
        "knapsack": (KNAPSACK_PROBLEM, "0/1 背包问题 — plan + dp + assembler"),
    }

    parser = argparse.ArgumentParser(description="Full Pipeline 集成测试与演示")
    parser.add_argument(
        "--demo", action="store_true",
        help="运行完整端到端演示（需要 LLM API Key）",
    )
    parser.add_argument(
        "--problem",
        choices=[*_PROBLEMS.keys(), "all"],
        default="lis",
        help="选择测试题目（默认 lis）",
    )
    args = parser.parse_args()

    if args.demo:
        targets = list(_PROBLEMS.keys()) if args.problem == "all" else [args.problem]
        for key in targets:
            prob_text, prob_label = _PROBLEMS[key]
            demo_full_pipeline(prob_text, prob_label)
    else:
        print("Use --demo to run the full pipeline demo.")
        print("Running smoke tests (no LLM) ...")

        g = build_assembler_agent_graph()
        assert g is not None
        assert "code_assembler" in g.get_graph().nodes
        print("  assembler graph compilation : OK")

        ir_smoke = _make_minimal_ir("dp")
        assert ir_smoke["task_nodes"][1]["agent_type"] == "dp"
        print("  _make_minimal_ir            : OK")

        issues_smoke = validate_final_code("", ir_smoke)
        assert "final_code is empty" in issues_smoke[0]
        print("  validate_final_code (empty) : OK")

        dummy_code = (
            "def lis_length(nums): return 1\n"
            "def solve(): pass\n"
            'if __name__ == "__main__": solve()\n'
        )
        issues_smoke2 = validate_final_code(dummy_code, ir_smoke)
        assert not issues_smoke2, f"Unexpected issues: {issues_smoke2}"
        print("  validate_final_code (valid) : OK")

        print("All smoke tests passed.")
