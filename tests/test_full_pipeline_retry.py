"""Full pipeline + retry path integration tests.

Covers the complete flow:
  plan_agent → algo_agents → assembler_agent → final_verifier_agent (with retry)

Specifically exercises the retry path added in the Final Verifier Retry plan:
  error_locator → task_replanner → algo_retrier → code_reassembler → final_report_assembler

Usage:
  # 无 LLM 单元测试
  pytest tests/test_full_pipeline_retry.py

  # 完整集成测试（需要 LLM API Key）
  pytest tests/test_full_pipeline_retry.py -m integration

  # CLI demo
  python tests/test_full_pipeline_retry.py --demo
  python tests/test_full_pipeline_retry.py --demo --problem two_sum
  python tests/test_full_pipeline_retry.py --demo --problem all
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import AIMessage

from auditcoder.algo_agents import dispatch_algo_agents
from auditcoder.assembler_agent import build_assembler_agent_graph
from auditcoder.final_verifier_agent import build_final_verifier_graph
from auditcoder.final_verifier_agent.graph import _locator_router
from auditcoder.final_verifier_agent.nodes import (
    algo_retrier,
    code_reassembler,
    final_report_assembler,
    task_replanner,
)
from auditcoder.plan_agent.graph import build_plan_agent_graph

from tests.test_algo_agents_integration import (
    LIS_PROBLEM,
    _make_minimal_ir,
    validate_snippets,
)
from tests.test_full_pipeline import validate_final_code

# ---------------------------------------------------------------------------
# Problem fixtures
# ---------------------------------------------------------------------------

TWO_SUM_PROBLEM = """\
题目：有序数组两数之和
给定一个严格递增的整数数组 nums（下标从 1 开始）和一个目标值 target，\
找到两个数使得它们的和等于 target，返回它们的下标（较小的在前，用空格分隔）。\
保证恰好存在唯一一组解。

输入格式：
第一行两个整数 n target (2 <= n <= 100000, -10^9 <= target <= 10^9)。
第二行 n 个整数，严格递增，其中 -10^9 <= nums[i] <= 10^9。

输出格式：
输出两个整数 i j，满足 nums[i] + nums[j] == target，且 i < j（1-indexed）。

样例输入：
4 9
2 7 11 15

样例输出：
1 2

样例解释：nums[1] + nums[2] = 2 + 7 = 9。

时间限制：1s
内存限制：256MB
"""

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_wrong_answer_final_code() -> str:
    """Return a final_code that always prints -1 (guaranteed wrong answer)."""
    return (
        "def solve():\n"
        "    print(-1)\n\n"
        "if __name__ == '__main__':\n"
        "    solve()\n"
    )


def _make_full_verifier_input(
    final_code: str,
    ir: dict[str, Any],
    code_snippets: dict[str, str],
    verification_reports: dict[str, Any] | None = None,
    retried_tasks: list[str] | None = None,
) -> dict[str, Any]:
    """Build a FinalVerifierState-compatible input dict."""
    return {
        "final_code": final_code,
        "task_plan_ir": ir,
        "code_snippets": code_snippets,
        "verification_reports": verification_reports or {},
        "retried_tasks": retried_tasks or [],
    }


# ---------------------------------------------------------------------------
# Section 2 — Unit tests (no LLM required)
# ---------------------------------------------------------------------------


def test_final_verifier_graph_has_new_nodes():
    """Compiled graph should include all three retry nodes."""
    graph = build_final_verifier_graph()
    assert graph is not None
    nodes = graph.get_graph().nodes
    for expected in ("task_replanner", "algo_retrier", "code_reassembler"):
        assert expected in nodes, (
            f"Expected node '{expected}' in graph, found: {list(nodes)}"
        )


def test_locator_router_no_location():
    """_locator_router should skip to final_report_assembler when error_location is None."""
    state: dict[str, Any] = {
        "error_location": None,
        "retried_tasks": [],
    }
    assert _locator_router(state) == "final_report_assembler"


def test_locator_router_first_failure():
    """_locator_router should route to task_replanner on the first located failure."""
    state: dict[str, Any] = {
        "error_location": {"task_id": "S1", "agent_type": "dp"},
        "retried_tasks": [],
    }
    assert _locator_router(state) == "task_replanner"


def test_locator_router_already_retried():
    """_locator_router should skip retry when task_id was already retried."""
    state: dict[str, Any] = {
        "error_location": {"task_id": "S1", "agent_type": "dp"},
        "retried_tasks": ["S1"],
    }
    assert _locator_router(state) == "final_report_assembler"


def test_locator_router_missing_retried_tasks_field():
    """_locator_router should handle absent retried_tasks (treats as empty list)."""
    state: dict[str, Any] = {
        "error_location": {"task_id": "S1"},
    }
    assert _locator_router(state) == "task_replanner"


def test_task_replanner_with_mock_llm():
    """task_replanner should update task_plan_ir and populate retried_tasks via LLM."""
    ir = _make_minimal_ir("dp")
    original_node = next(n for n in ir["task_nodes"] if n["id"] == "S1")

    updated_node_json = {
        **original_node,
        "hints": ["use patience sorting", "maintain tails array"],
    }
    mock_response = AIMessage(content=json.dumps(updated_node_json))
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    state: dict[str, Any] = {
        "error_location": {
            "task_id": "S1",
            "agent_type": "dp",
            "error_type": "wrong_answer",
            "detail": "expected 4, got 3",
            "failing_case": {"case_index": 0, "expected": "4", "actual": "3"},
        },
        "task_plan_ir": ir,
        "retried_tasks": [],
    }

    with patch("auditcoder.final_verifier_agent.nodes.task_replanner.get_llm", return_value=mock_llm):
        result = task_replanner(state)

    assert "updated_task_node" in result
    assert "task_plan_ir" in result
    assert "retried_tasks" in result

    assert result["updated_task_node"].get("id") == "S1"
    assert "S1" in result["retried_tasks"]

    updated_ir = result["task_plan_ir"]
    replaced = next(n for n in updated_ir["task_nodes"] if n["id"] == "S1")
    assert replaced.get("hints") == ["use patience sorting", "maintain tails array"], (
        "task_plan_ir should contain the updated task node"
    )

    mock_llm.invoke.assert_called_once()


def test_task_replanner_idempotent_id():
    """task_replanner should preserve task id even if LLM omits it."""
    ir = _make_minimal_ir("greedy")

    mock_response = AIMessage(content='{"goal": "updated goal", "agent_type": "greedy"}')
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_response

    state: dict[str, Any] = {
        "error_location": {"task_id": "S1", "error_type": "wrong_answer", "detail": ""},
        "task_plan_ir": ir,
        "retried_tasks": [],
    }

    with patch("auditcoder.final_verifier_agent.nodes.task_replanner.get_llm", return_value=mock_llm):
        result = task_replanner(state)

    assert result["updated_task_node"]["id"] == "S1", (
        "task id must be preserved even when LLM response omits it"
    )


def test_algo_retrier_with_mock_agent():
    """algo_retrier should update code_snippets with the new snippet from the agent."""
    ir = _make_minimal_ir("dp")
    updated_node = next(n for n in ir["task_nodes"] if n["id"] == "S1")
    new_snippet = "def lis_length(nums: list[int]) -> int:\n    return len(nums)\n"

    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {"code_snippet": new_snippet}

    state: dict[str, Any] = {
        "updated_task_node": updated_node,
        "task_plan_ir": ir,
        "code_snippets": {"S1": "def lis_length(nums): return 0"},
    }

    with patch.dict(
        "auditcoder.final_verifier_agent.nodes.algo_retrier._AGENT_BUILDERS",
        {"dp": lambda: mock_agent},
    ):
        result = algo_retrier(state)

    assert "code_snippets" in result
    assert result["code_snippets"]["S1"] == new_snippet
    mock_agent.invoke.assert_called_once_with(
        {"task_node": updated_node, "task_plan_ir": ir}
    )


def test_algo_retrier_unknown_agent_type():
    """algo_retrier should return empty dict for unknown agent_type."""
    ir = _make_minimal_ir("dp")
    unknown_node = {**next(n for n in ir["task_nodes"] if n["id"] == "S1"), "agent_type": "unknown_xyz"}

    state: dict[str, Any] = {
        "updated_task_node": unknown_node,
        "task_plan_ir": ir,
        "code_snippets": {},
    }

    result = algo_retrier(state)
    assert result == {}, f"Expected empty dict for unknown agent_type, got: {result}"


def test_code_reassembler_with_mock_assembler():
    """code_reassembler should invoke assembler_agent and return updated final_code."""
    ir = _make_minimal_ir("dp")
    new_code = (
        "def lis_length(nums): return 1\n\n"
        "def solve(): print(lis_length([1,2,3]))\n\n"
        'if __name__ == "__main__": solve()\n'
    )

    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {"final_code": new_code}

    state: dict[str, Any] = {
        "task_plan_ir": ir,
        "code_snippets": {"S1": "def lis_length(nums): return 1"},
    }

    with patch(
        "auditcoder.final_verifier_agent.nodes.code_reassembler.build_assembler_agent_graph",
        return_value=mock_graph,
    ):
        result = code_reassembler(state)

    assert result == {"final_code": new_code}
    mock_graph.invoke.assert_called_once_with(
        {"task_plan_ir": ir, "code_snippets": state["code_snippets"]}
    )


def test_final_report_includes_retry_info():
    """final_report_assembler should write retry_info when retried_tasks is non-empty."""
    ir = _make_minimal_ir("dp")
    state: dict[str, Any] = {
        "execution_result": {
            "all_passed": False,
            "error_type": "wrong_answer",
            "time_ms": 50.0,
            "results": [],
            "stderr": "",
        },
        "error_location": {
            "task_id": "S1",
            "agent_type": "dp",
            "error_type": "wrong_answer",
            "detail": "expected 4, got -1",
            "method": "verification_report",
        },
        "verification_reports": {},
        "retried_tasks": ["S1"],
    }

    result = final_report_assembler(state)
    report = result["final_report"]

    assert "retry_info" in report, "retry_info should be present when retried_tasks is non-empty"
    assert report["retry_info"]["retry_attempted"] is True
    assert "S1" in report["retry_info"]["retried_task_ids"]


def test_final_report_no_retry_info_when_empty():
    """final_report_assembler should NOT add retry_info when retried_tasks is empty."""
    ir = _make_minimal_ir("dp")
    state: dict[str, Any] = {
        "execution_result": {
            "all_passed": True,
            "error_type": "none",
            "time_ms": 30.0,
            "results": [],
            "stderr": "",
        },
        "error_location": None,
        "verification_reports": {},
        "retried_tasks": [],
    }

    result = final_report_assembler(state)
    report = result["final_report"]

    assert "retry_info" not in report, (
        f"retry_info should be absent when retried_tasks is empty, got: {report.get('retry_info')}"
    )


# ---------------------------------------------------------------------------
# Section 3 — Integration tests (require LLM API Key)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_pipeline_two_sum():
    """End-to-end: two_sum problem → plan_agent → algo_agents → assembler → final_verifier."""
    plan_result = build_plan_agent_graph().invoke(
        {"problem_raw": TWO_SUM_PROBLEM, "revision_count": 0}
    )
    assert "task_plan_ir" in plan_result, "plan_agent should produce task_plan_ir"
    ir = plan_result["task_plan_ir"]

    leaf_nodes = [n for n in ir.get("task_nodes", []) if n.get("is_leaf")]
    assert len(leaf_nodes) >= 1, "two_sum problem should have at least one leaf task"

    code_snippets, verification_reports = dispatch_algo_agents(ir)
    assert len(code_snippets) > 0, "algo_agents should produce at least one code snippet"

    snippet_issues = validate_snippets(code_snippets, ir, require_all_leaves=False)
    assert not snippet_issues, f"Snippet validation issues: {snippet_issues}"

    assembler_result = build_assembler_agent_graph().invoke(
        {"task_plan_ir": ir, "code_snippets": code_snippets}
    )
    assert "final_code" in assembler_result
    final_code = assembler_result["final_code"]

    code_issues = validate_final_code(final_code, ir)
    assert not code_issues, f"final_code validation issues: {code_issues}"

    final_result = build_final_verifier_graph().invoke(
        _make_full_verifier_input(final_code, ir, code_snippets, verification_reports)
    )
    final_report = final_result.get("final_report", {})
    assert "verdict" in final_report
    assert final_report["verdict"] in ("accept", "reject")

    print(f"\n{'=' * 70}")
    print("[Full pipeline — Two Sum] final_code:")
    print("=" * 70)
    print(final_code)
    print(f"\nFinal verdict: {final_report['verdict']}")
    if final_report.get("responsible_agent"):
        print(f"Responsible agent: {final_report['responsible_agent']}")
    if final_report.get("retry_info"):
        print(f"Retry info: {final_report['retry_info']}")
    print("=" * 70)


@pytest.mark.integration
def test_retry_info_present_on_failure():
    """Injecting a bad final_code should trigger the retry path and produce retry_info.

    Strategy:
    - Use _make_minimal_ir("dp") for a controlled IR with known sample test case.
    - Pass a guaranteed-wrong final_code (always prints -1).
    - Pass verification_reports with S1 rejected so error_locator can pin the task.
    - Verify that final_report contains retry_info after task_replanner → algo_retrier
      → code_reassembler executes.
    """
    ir = _make_minimal_ir("dp")
    bad_final_code = _make_wrong_answer_final_code()

    verification_reports = {
        "S1": {"verdict": "reject", "agent_type": "dp", "summary": "wrong answer"}
    }
    code_snippets = {"S1": "def lis_length(nums: list[int]) -> int:\n    return -1\n"}

    final_result = build_final_verifier_graph().invoke(
        _make_full_verifier_input(
            bad_final_code, ir, code_snippets, verification_reports
        )
    )
    final_report = final_result.get("final_report", {})

    assert "verdict" in final_report, "final_report should contain verdict"
    assert final_report["verdict"] in ("accept", "reject")

    assert "retry_info" in final_report, (
        "Retry should have been triggered: final_report is missing retry_info. "
        f"Got keys: {list(final_report.keys())}"
    )
    retry_info = final_report["retry_info"]
    assert retry_info["retry_attempted"] is True
    assert "S1" in retry_info["retried_task_ids"]

    print(f"\n{'=' * 70}")
    print("[Retry path test] Final report:")
    print("=" * 70)
    print(f"  Verdict       : {final_report['verdict']}")
    print(f"  Retry info    : {retry_info}")
    if final_report.get("responsible_agent"):
        ra = final_report["responsible_agent"]
        print(f"  Responsible   : task={ra.get('task_id')}  agent={ra.get('agent_type')}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Section 4 — Demo function & CLI runner
# ---------------------------------------------------------------------------


def demo_full_pipeline_with_retry(problem: str, label: str) -> None:
    """Run the complete four-step pipeline and display per-step results including retry."""
    separator = "=" * 72
    thin = "-" * 72

    print(f"\n{separator}")
    print(f"  End-to-End Demo — {label}")
    print(separator)

    # ── Step 1: plan_agent ────────────────────────────────────────────────
    print("\n[Step 1] Running plan_agent ...")
    plan_result = build_plan_agent_graph().invoke(
        {"problem_raw": problem, "revision_count": 0}
    )
    ir = plan_result.get("task_plan_ir", {})

    print(f"  Global strategy : {ir.get('global_strategy', '')}")
    print(f"  Execution order : {ir.get('execution_order', [])}")

    task_nodes = ir.get("task_nodes", [])
    leaf_nodes = [n for n in task_nodes if n.get("is_leaf")]
    print(f"\n  Task tree: {len(task_nodes)} nodes, {len(leaf_nodes)} leaf(s)")
    for node in task_nodes:
        tag = "[LEAF]" if node.get("is_leaf") else "[INTERNAL]"
        agent = f" → {node.get('agent_type')}" if node.get("agent_type") else ""
        print(f"    {node['id']} {tag}{agent}  {node.get('goal', '')}")

    # ── Step 2: dispatch_algo_agents ─────────────────────────────────────
    print("\n[Step 2] Dispatching to algo agents (per-task verification + retry) ...")
    code_snippets, verification_reports = dispatch_algo_agents(ir)
    print(f"  Generated {len(code_snippets)} code snippet(s)")

    for task_id, report in verification_reports.items():
        print(f"  Task {task_id}: verifier verdict = {report.get('verdict', '?')}")

    if not code_snippets:
        print("  (no snippets produced — check agent_type values in task_nodes)")
    else:
        for task_id, snippet in code_snippets.items():
            node = next((n for n in task_nodes if n["id"] == task_id), {})
            print(f"\n{thin}")
            print(f"  Task : {task_id}  [{node.get('agent_type')}]")
            print(f"  Goal : {node.get('goal', '')}")
            print(thin)
            print(snippet)

    # ── Step 3: assembler_agent ───────────────────────────────────────────
    print("\n[Step 3] Running assembler_agent ...")
    assembler_result = build_assembler_agent_graph().invoke(
        {"task_plan_ir": ir, "code_snippets": code_snippets}
    )
    final_code = assembler_result.get("final_code", "")

    print(f"\n{'═' * 72}")
    print("  FINAL CODE (complete runnable Python solution):")
    print(f"{'═' * 72}")
    print(final_code)

    code_issues = validate_final_code(final_code, ir)
    print(thin)
    if code_issues:
        print(f"  Code validation issues ({len(code_issues)}):")
        for i, issue in enumerate(code_issues, 1):
            print(f"    {i}. {issue}")
    else:
        print("  Code validation: PASSED")

    # ── Step 4: final_verifier_agent (with retry) ─────────────────────────
    print("\n[Step 4] Running final_verifier_agent (retry path active) ...")
    final_result = build_final_verifier_graph().invoke(
        _make_full_verifier_input(final_code, ir, code_snippets, verification_reports)
    )
    final_report = final_result.get("final_report", {})

    print(f"\n  Final verdict   : {final_report.get('verdict', 'unknown')}")
    print(f"  Error type      : {final_report.get('error_type', 'none')}")

    if final_report.get("responsible_agent"):
        ra = final_report["responsible_agent"]
        print(
            f"  Responsible     : task={ra.get('task_id')}  "
            f"agent={ra.get('agent_type')}  error={ra.get('error_type')}"
        )
        detail = ra.get("detail", "")
        if detail:
            print(f"  Detail          : {detail[:200]}")

    retry_info = final_report.get("retry_info")
    if retry_info:
        print(f"\n  [RETRY TRIGGERED]")
        print(f"  Retried tasks   : {retry_info['retried_task_ids']}")
        print(f"  Retry attempted : {retry_info['retry_attempted']}")
        # Show the new final_code produced after retry
        new_final_code = final_result.get("final_code", "")
        if new_final_code and new_final_code != final_code:
            print(f"\n  Code after retry:")
            print(f"{'─' * 72}")
            print(new_final_code)
    else:
        print("  (no retry triggered — first attempt passed or no locatable error)")

    print(separator)


if __name__ == "__main__":
    import argparse

    _PROBLEMS: dict[str, tuple[str, str]] = {
        "lis": (
            LIS_PROBLEM,
            "最长递增子序列 (LIS) — plan + dp/binary_search + assembler + verifier",
        ),
        "two_sum": (
            TWO_SUM_PROBLEM,
            "有序数组两数之和 — plan + two_pointer + assembler + verifier",
        ),
    }

    parser = argparse.ArgumentParser(
        description="Full Pipeline + Retry 集成测试与演示"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
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
            demo_full_pipeline_with_retry(prob_text, prob_label)
    else:
        print("Use --demo to run the full pipeline demo.")
        print("Running smoke tests (no LLM) ...")

        # Graph compilation
        g = build_final_verifier_graph()
        assert g is not None
        nodes_in_graph = g.get_graph().nodes
        for node_name in ("task_replanner", "algo_retrier", "code_reassembler"):
            assert node_name in nodes_in_graph
        print("  new node registration   : OK")

        # _locator_router
        assert _locator_router({"error_location": None, "retried_tasks": []}) == "final_report_assembler"
        assert _locator_router({"error_location": {"task_id": "S1"}, "retried_tasks": []}) == "task_replanner"
        assert _locator_router({"error_location": {"task_id": "S1"}, "retried_tasks": ["S1"]}) == "final_report_assembler"
        print("  _locator_router logic   : OK")

        # final_report_assembler retry_info
        _state: dict[str, Any] = {
            "execution_result": {"all_passed": False, "error_type": "wrong_answer", "time_ms": 0, "results": [], "stderr": ""},
            "error_location": None,
            "verification_reports": {},
            "retried_tasks": ["S1"],
        }
        _report = final_report_assembler(_state)["final_report"]
        assert "retry_info" in _report
        assert _report["retry_info"]["retry_attempted"] is True
        print("  retry_info field        : OK")

        _state2: dict[str, Any] = {
            "execution_result": {"all_passed": True, "error_type": "none", "time_ms": 0, "results": [], "stderr": ""},
            "error_location": None,
            "verification_reports": {},
            "retried_tasks": [],
        }
        _report2 = final_report_assembler(_state2)["final_report"]
        assert "retry_info" not in _report2
        print("  no retry_info when empty: OK")

        print("All smoke tests passed.")
