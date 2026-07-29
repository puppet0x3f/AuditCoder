"""APPS 数据集批量测试脚本（消融实验：含根路径树重规划）

消融实验版本：错误定位后，从失败叶节点向上追溯父链直至根节点（含根），
对路径上**每个节点**（包括根节点）的所有叶后代逐一调用 task_replanner +
algo_retrier，保留原任务树结构，再调用 code_reassembler 重组代码。

与 run_apps_path_replan.py 的唯一核心差异：
  路径树重规划（排除根）：收集路径节点叶后代时**跳过根节点**，仅覆盖失败分支
                           （run_apps_path_replan.py）
  含根路径树重规划（本脚本）：**不跳过根节点**，从根节点收集全部叶后代 ← 本脚本

注意：由于根节点是所有节点的祖先，纳入根节点后路径子树叶节点集合等于整棵树的
全部叶节点。与全局重规划（run_apps_global_replan.py）的本质区别在于：本脚本
**保留原任务树的拓扑结构和节点分解方案**，只通过 task_replanner 对各节点的
hints/interface/context 做局部修正；全局重规划则通过 plan_agent 重建整棵树。

四点消融对比（重规划范围由小到大）：
  单节点重试：        仅对失败叶节点调用 task_replanner + algo_retrier
                      （final_verifier_agent 内置行为）
  路径树重规划：      失败节点祖先路径子树（排除根）内的叶节点
                      （run_apps_path_replan.py）
  含根路径树重规划：  祖先路径含根，覆盖整棵树叶节点，保留树结构逐叶修正 ← 本脚本
  全局重规划：        将所有错误拼入 problem_raw，重跑 plan_agent 生成全新任务树
                      （run_apps_global_replan.py）

流水线对比：
  原完整版:   plan_agent → algo_agents → assembler_agent → final_verifier_agent
              （重试：error_locator → task_replanner[单任务] → algo_retrier → code_reassembler）

  本次消融:   plan_agent → algo_agents → assembler_agent
              （重试：error_locator → _get_ancestor_path（含根）
                      → _get_path_subtree_leaves_full（不跳过根节点）
                      → task_replanner[逐叶] + algo_retrier[逐叶] → code_reassembler）

APPS 数据集字段说明：
  - id           : 题目编号（0-based，共 5000 条）
  - question     : 题目描述（自然语言）
  - solutions    : JSON 字符串，包含参考答案列表
  - input_output : JSON 字符串，格式 {"inputs": [...], "outputs": [...]}，已为标准 stdin/stdout
  - difficulty   : "introductory" | "interview" | "competition"
  - url          : 题目来源链接
  - starter_code : 起始代码模板（可能为空字符串）

Usage:
  # 从第 0 条开始读取 100 条（默认行为）
  python tests/run_apps_fullpath_replan.py --offset 0 --count 100

  # 从第 200 条开始读取 50 条
  python tests/run_apps_fullpath_replan.py --offset 200 --count 50

  # 按难度过滤
  python tests/run_apps_fullpath_replan.py --offset 0 --count 100 --difficulty introductory

  # 指定特定 id（逗号分隔）
  python tests/run_apps_fullpath_replan.py --ids 0,5,10

  # 每题最多运行 5 个测试用例（默认 10）
  python tests/run_apps_fullpath_replan.py --offset 0 --count 10 --max-cases 5

  # 自定义输出目录
  python tests/run_apps_fullpath_replan.py --offset 0 --count 10 --output result/apps_ablation_fullpath_replan

  # 对比实验（含根 vs 排除根）
  python tests/run_apps_path_replan.py     --ids 0,1,2   # 路径树排除根
  python tests/run_apps_fullpath_replan.py --ids 0,1,2   # 路径树含根（本脚本）

  # 安静模式
  python tests/run_apps_fullpath_replan.py --offset 0 --count 100 --quiet
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditcoder.algo_agents import dispatch_algo_agents
from auditcoder.assembler_agent import build_assembler_agent_graph
from auditcoder.final_verifier_agent.nodes.algo_retrier import algo_retrier
from auditcoder.final_verifier_agent.nodes.code_reassembler import code_reassembler
from auditcoder.final_verifier_agent.nodes.error_locator import error_locator
from auditcoder.final_verifier_agent.nodes.full_code_runner import full_code_runner
from auditcoder.final_verifier_agent.nodes.task_replanner import task_replanner
from auditcoder.plan_agent.graph import build_plan_agent_graph

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------

APPS_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "apps.jsonl"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "result" / "apps_ablation_fullpath_replan"
)

VALID_DIFFICULTIES = {"introductory", "interview", "competition"}

# ---------------------------------------------------------------------------
# APPS 数据分批加载
# ---------------------------------------------------------------------------


def load_apps_batch(
    path: Path,
    offset: int = 0,
    count: int = 100,
    difficulty: str | None = None,
) -> list[dict[str, Any]]:
    """从 JSONL 文件按行流式读取，跳过 offset 条后取 count 条。"""
    results: list[dict[str, Any]] = []
    skipped = 0

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            item = json.loads(raw_line)

            if difficulty and item.get("difficulty") != difficulty:
                continue

            if skipped < offset:
                skipped += 1
                continue

            results.append(item)
            if len(results) >= count:
                break

    return results


def load_apps_by_ids(
    path: Path,
    ids: list[int],
) -> list[dict[str, Any]]:
    """按 id 列表从 JSONL 中精确读取指定记录，流式扫描，无需全量加载。"""
    target_ids = set(ids)
    found: dict[int, dict[str, Any]] = {}

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            item = json.loads(raw_line)
            pid = item.get("id")
            if pid in target_ids:
                found[pid] = item
                if len(found) == len(target_ids):
                    break

    missing = [i for i in ids if i not in found]
    if missing:
        print(f"[警告] 以下 id 不存在于数据集中，已跳过：{missing}")

    return [found[i] for i in ids if i in found]


def parse_io_field(raw: str | dict | None) -> dict[str, Any]:
    """解析 input_output 字段（可能是 JSON 字符串或已解析的字典）。"""
    if not raw:
        return {"inputs": [], "outputs": []}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"inputs": [], "outputs": []}


# ---------------------------------------------------------------------------
# 问题格式化
# ---------------------------------------------------------------------------


def format_apps_problem(item: dict[str, Any], sample_count: int = 3) -> str:
    """将 APPS 条目格式化为 pipeline 可识别的题目描述。"""
    question: str = item.get("question", "").strip()
    starter_code: str = item.get("starter_code", "").strip()
    difficulty: str = item.get("difficulty", "unknown")

    io_data = parse_io_field(item.get("input_output"))
    inputs: list[str] = io_data.get("inputs", [])
    outputs: list[str] = io_data.get("outputs", [])

    lines: list[str] = []

    lines += [f"【题目难度】{difficulty}", ""]
    lines += ["【题目描述】", question, ""]

    if starter_code:
        lines += [
            "【起始代码模板】",
            "请在以下模板基础上实现完整逻辑（保留函数/类签名）：",
            "```python",
            starter_code,
            "```",
            "",
        ]

    lines += [
        "【输入输出格式】",
        "代码从标准输入（stdin）读取数据，将结果打印到标准输出（stdout）。",
        "",
    ]

    samples = list(zip(inputs[:sample_count], outputs[:sample_count]))
    if samples:
        lines.append("【样例】")
        for idx, (inp, out) in enumerate(samples, 1):
            lines.append(f"样例 {idx}：")
            lines.append("  输入：")
            for ln in inp.rstrip("\n").splitlines():
                lines.append(f"    {ln}")
            lines.append("  输出：")
            for ln in out.rstrip("\n").splitlines():
                lines.append(f"    {ln}")
            lines.append("")

    lines += [
        "【要求】",
        "  - 代码从标准输入（stdin）读取数据，将结果用 print() 输出到标准输出（stdout）。",
        "  - 代码必须通过全部样例，应简洁、正确、高效。",
        "  - 不要输出多余的调试信息。",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 测试执行
# ---------------------------------------------------------------------------


def run_apps_tests(
    final_code: str,
    io_data: dict[str, Any],
    max_cases: int = 10,
) -> dict[str, Any]:
    """用 subprocess 运行 final_code，逐条注入 stdin 并比对 stdout。"""
    inputs: list[str] = io_data.get("inputs", [])
    outputs: list[str] = io_data.get("outputs", [])

    pairs = list(zip(inputs, outputs))[:max_cases]

    if not pairs:
        return {
            "all_passed": False,
            "cases_run": 0,
            "cases_passed": 0,
            "compile_error": None,
            "cases": [],
        }

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(final_code)

        cases: list[dict[str, Any]] = []
        all_passed = True

        for idx, (inp, expected_out) in enumerate(pairs):
            expected = expected_out.rstrip("\n")
            try:
                proc = subprocess.run(
                    [sys.executable, tmp_path],
                    input=inp,
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
                actual = proc.stdout.rstrip("\n")
                passed = actual == expected
                error = proc.stderr.strip() if proc.returncode != 0 else None
            except subprocess.TimeoutExpired:
                actual = ""
                passed = False
                error = "timeout: 执行超过 10 秒"
            except Exception as exc:
                actual = ""
                passed = False
                error = f"{type(exc).__name__}: {exc}"

            if not passed:
                all_passed = False

            cases.append(
                {
                    "index": idx,
                    "input": inp,
                    "expected": expected,
                    "actual": actual,
                    "passed": passed,
                    "error": error,
                }
            )

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    cases_passed = sum(1 for c in cases if c["passed"])

    return {
        "all_passed": all_passed,
        "cases_run": len(cases),
        "cases_passed": cases_passed,
        "compile_error": None,
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# 路径树辅助函数（与 run_apps_path_replan.py 共享相同机制，仅去掉根节点排除）
# ---------------------------------------------------------------------------


def _get_ancestor_path(
    failing_id: str,
    node_map: dict[str, dict[str, Any]],
) -> list[str]:
    """从失败叶节点出发沿 parent 链向上追溯，返回完整祖先路径（含根）。

    返回列表从失败节点开始，以根节点结束：
        [failing_id, parent_id, ..., root_id]
    """
    path: list[str] = []
    current_id: str | None = failing_id

    while current_id is not None:
        path.append(current_id)
        node = node_map.get(current_id, {})
        current_id = node.get("parent")

    return path


def _collect_leaf_descendants(
    node_id: str,
    node_map: dict[str, dict[str, Any]],
) -> set[str]:
    """递归收集以 node_id 为根的子树中所有叶节点的 ID 集合。"""
    node = node_map.get(node_id, {})
    if node.get("is_leaf", False):
        return {node_id}

    leaves: set[str] = set()
    for child_id in node.get("dependencies", []):
        leaves |= _collect_leaf_descendants(child_id, node_map)
    return leaves


def _get_path_subtree_leaves_full(
    path_ids: list[str],
    node_map: dict[str, dict[str, Any]],
    execution_order: list[str],
) -> list[str]:
    """收集路径子树（含根节点）所有叶节点，按执行顺序排序。

    与 run_apps_path_replan.py 中 _get_path_subtree_leaves 的唯一差异：
    不跳过根节点（parent is None 的节点）。由于根节点是所有节点的祖先，
    纳入根节点后所收集的叶节点集合等于整棵树的全部叶节点。

    这使本脚本在"保留树结构、逐叶局部修正"的前提下覆盖最大范围，
    与 run_apps_global_replan.py（重建整棵树）形成对照。
    """
    all_leaves: set[str] = set()

    for nid in path_ids:
        # 不做根节点排除，直接收集所有路径节点的叶后代
        all_leaves |= _collect_leaf_descendants(nid, node_map)

    order_index = {tid: idx for idx, tid in enumerate(execution_order)}
    return sorted(all_leaves, key=lambda tid: order_index.get(tid, 9999))


# ---------------------------------------------------------------------------
# 叶节点 error_location 构造
# ---------------------------------------------------------------------------


def _make_leaf_error_location(
    leaf_id: str,
    is_failing_node: bool,
    actual_error_location: dict[str, Any],
) -> dict[str, Any]:
    """为含根路径子树中的单个叶节点构造 error_location。

    对失败节点本身，直接使用真实错误信息（仅覆盖 task_id 以确保指向正确）。
    对其余叶节点，注入关联上下文：告知 task_replanner 正在进行含根路径树联合重规划，
    传递原始错误供参考，以便其审视接口契约并加强鲁棒性。
    """
    if is_failing_node:
        return {**actual_error_location, "task_id": leaf_id}

    related: dict[str, Any] = {
        "task_id": leaf_id,
        "error_type": "fullpath_related",
        "detail": (
            f"当前正在对含根祖先路径子树进行联合重规划（含根节点）。"
            f"触发原因：节点 {actual_error_location.get('task_id', '?')} 失败。"
            f"原始错误：{actual_error_location.get('error_type', 'unknown')} — "
            f"{actual_error_location.get('detail', '')[:200]}"
        ),
    }
    failing_case = actual_error_location.get("failing_case")
    if failing_case:
        related["failing_case"] = failing_case
    return related


# ---------------------------------------------------------------------------
# 含根路径树重规划执行
# ---------------------------------------------------------------------------


def _run_fullpath_replan(
    ir: dict[str, Any],
    code_snippets: dict[str, str],
    verification_reports: dict[str, dict[str, Any]],
    error_location: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], str, list[str]]:
    """对含根路径子树内所有叶节点逐一执行 task_replanner + algo_retrier，最后重组代码。

    与 run_apps_path_replan.py 中 _run_path_replan 的唯一差异：
    使用 _get_path_subtree_leaves_full 取路径子树叶节点时**不跳过根节点**，
    从而覆盖整棵树的叶节点。路径遍历机制（_get_ancestor_path）与 path_replan 版本完全一致。

    流程：
      1. 从失败叶节点上溯完整祖先路径（含根节点）。
      2. 对路径上每个节点（含根）收集所有叶后代，按执行顺序排列。
      3. 对每个叶节点：
           a. 构造临时 FinalVerifierState（含正确的 error_location.task_id）。
           b. 调用 task_replanner → 更新节点规格并将新节点写回 ir。
           c. 调用 algo_retrier  → 重新生成该节点对应的代码片段。
      4. 调用 code_reassembler 用更新后的 ir + code_snippets 重组最终代码。

    Returns:
        (updated_ir, updated_code_snippets, final_code, subtree_leaf_ids)
    """
    failing_task_id = error_location.get("task_id", "")
    node_map: dict[str, dict[str, Any]] = {
        n["id"]: n for n in ir.get("task_nodes", [])
    }
    execution_order: list[str] = ir.get("execution_order", [])

    # ── 1. 计算含根路径子树叶节点 ────────────────────────────────────────
    ancestor_path = _get_ancestor_path(failing_task_id, node_map)
    subtree_leaf_ids = _get_path_subtree_leaves_full(ancestor_path, node_map, execution_order)

    # 极端兜底：树中无叶节点时退为仅重规划失败节点
    if not subtree_leaf_ids and failing_task_id:
        subtree_leaf_ids = [failing_task_id]

    current_ir = ir
    current_snippets = dict(code_snippets)

    # ── 2. 逐叶重规划 ────────────────────────────────────────────────────
    for leaf_id in subtree_leaf_ids:
        is_failing = leaf_id == failing_task_id
        leaf_error = _make_leaf_error_location(leaf_id, is_failing, error_location)

        temp_state: dict[str, Any] = {
            "task_plan_ir": current_ir,
            "code_snippets": current_snippets,
            "verification_reports": verification_reports,
            "error_location": leaf_error,
            "retried_tasks": [],
            "updated_task_node": None,
        }

        # a. 重规划节点规格
        replan_result = task_replanner(temp_state)
        temp_state.update(replan_result)
        current_ir = temp_state.get("task_plan_ir", current_ir)

        # b. 重跑对应算法 Agent
        retry_result = algo_retrier(temp_state)
        if retry_result:
            updated_snippets = retry_result.get("code_snippets")
            if updated_snippets:
                current_snippets = updated_snippets

    # ── 3. 重组最终代码 ──────────────────────────────────────────────────
    reassemble_state: dict[str, Any] = {
        "task_plan_ir": current_ir,
        "code_snippets": current_snippets,
    }
    reassemble_result = code_reassembler(reassemble_state)
    final_code: str = reassemble_result.get("final_code", "")

    return current_ir, current_snippets, final_code, subtree_leaf_ids


# ---------------------------------------------------------------------------
# 单轮流水线
# ---------------------------------------------------------------------------


def _run_single_pass(
    problem_text: str,
) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, Any]], str]:
    """运行单轮 plan_agent → algo_agents → assembler_agent。

    Returns:
        (ir, code_snippets, verification_reports, final_code)
    """
    plan_result = build_plan_agent_graph().invoke(
        {"problem_raw": problem_text, "revision_count": 0}
    )
    ir: dict[str, Any] = plan_result.get("task_plan_ir", {})

    code_snippets, verification_reports = dispatch_algo_agents(ir)

    assembler_result = build_assembler_agent_graph().invoke(
        {"task_plan_ir": ir, "code_snippets": code_snippets}
    )
    final_code: str = assembler_result.get("final_code", "")

    return ir, code_snippets, verification_reports, final_code


def run_pipeline(problem_text: str) -> dict[str, Any]:
    """运行含根路径树重规划消融版 pipeline 并返回各阶段结果。

    流程：
      第一轮: plan_agent → algo_agents → assembler_agent
      检查:   full_code_runner + error_locator
      若失败: _get_ancestor_path（含根）→ _get_path_subtree_leaves_full（不跳过根节点）
              → task_replanner[逐叶] + algo_retrier[逐叶] → code_reassembler
      最终:   再次运行 full_code_runner，构建 final_report
    """
    # ── 第一轮 ────────────────────────────────────────────────────────────
    ir, code_snippets, verification_reports, final_code = _run_single_pass(problem_text)

    # ── 用 final_verifier_agent 的节点函数检查第一轮结果 ──────────────────
    fv_state: dict[str, Any] = {
        "final_code": final_code,
        "task_plan_ir": ir,
        "code_snippets": code_snippets,
        "verification_reports": verification_reports,
    }
    runner_result = full_code_runner(fv_state)
    fv_state.update(runner_result)

    execution_result: dict[str, Any] = fv_state.get("execution_result", {})
    fullpath_replan_attempted = False
    error_location_out: dict[str, Any] | None = None
    all_leaf_ids: list[str] = []

    if not execution_result.get("all_passed", False):
        # ── 错误定位 ──────────────────────────────────────────────────────
        locator_result = error_locator(fv_state)
        error_location_out = locator_result.get("error_location")

        if error_location_out:
            # ── 全节点路径重规划：逐叶调用 task_replanner + algo_retrier ──
            fullpath_replan_attempted = True
            ir, code_snippets, final_code, all_leaf_ids = _run_fullpath_replan(
                ir, code_snippets, verification_reports, error_location_out
            )

            # 对重规划后的代码重新运行检查
            fv_state_retry: dict[str, Any] = {
                "final_code": final_code,
                "task_plan_ir": ir,
                "code_snippets": code_snippets,
                "verification_reports": verification_reports,
            }
            runner_result_retry = full_code_runner(fv_state_retry)
            execution_result = runner_result_retry.get("execution_result", execution_result)

    # ── 构建 final_report ─────────────────────────────────────────────────
    all_passed = execution_result.get("all_passed", False)
    final_report: dict[str, Any] = {
        "verdict": "accept" if all_passed else "reject",
        "all_passed": all_passed,
        "error_type": execution_result.get("error_type", "none"),
        "time_ms": execution_result.get("time_ms", 0.0),
        "error_location": error_location_out,
        "retry_info": {
            "fullpath_replan_attempted": fullpath_replan_attempted,
            "all_leaf_ids": all_leaf_ids,
            "retry_strategy": "fullpath_replan",
        },
    }

    return {
        "ir": ir,
        "code_snippets": code_snippets,
        "verification_reports": verification_reports,
        "final_code": final_code,
        "final_report": final_report,
        "execution_result": execution_result,
    }


# ---------------------------------------------------------------------------
# JSON 结果构建
# ---------------------------------------------------------------------------


def build_apps_result_json(
    item: dict[str, Any],
    pipeline_result: dict[str, Any],
    apps_test_results: dict[str, Any],
) -> dict[str, Any]:
    """构建最终保存的 JSON 结构（含根路径树重规划消融版）。"""
    ir = pipeline_result["ir"]
    code_snippets = pipeline_result["code_snippets"]
    verification_reports = pipeline_result["verification_reports"]

    algo_outputs: dict[str, Any] = {}
    for task_id, snippet in code_snippets.items():
        algo_outputs[task_id] = {
            "snippet": snippet,
            "verification_report": verification_reports.get(task_id, {}),
        }

    return {
        "problem_id": item["id"],
        "difficulty": item.get("difficulty", "unknown"),
        "url": item.get("url", ""),
        "question": item.get("question", ""),
        "timestamp": datetime.now().isoformat(),
        "ablation": "fullpath_replan_on_failure",
        # 1. 测试结果
        "apps_test_results": apps_test_results,
        # 2. 完整代码
        "final_code": pipeline_result["final_code"],
        # 3. 各 agent 输出
        "agent_outputs": {
            "plan_agent": {
                "task_plan_ir": ir,
            },
            "algo_agents": algo_outputs,
            "assembler_agent": {
                "final_code": pipeline_result["final_code"],
            },
            "final_verifier_agent": {
                "final_report": pipeline_result["final_report"],
                "execution_result": pipeline_result["execution_result"],
            },
        },
    }


# ---------------------------------------------------------------------------
# 单题运行
# ---------------------------------------------------------------------------


def run_single_apps_problem(
    item: dict[str, Any],
    output_dir: Path,
    *,
    max_cases: int = 10,
    verbose: bool = True,
) -> bool:
    """对单道 APPS 题目运行含根路径树重规划消融版 pipeline 并将结果写入 JSON 文件。

    返回 True 表示执行成功（即使测试未全过），False 表示 pipeline 本身出现异常。
    """
    problem_id: int = item["id"]
    output_file = output_dir / f"apps_{problem_id}.json"

    if verbose:
        print(f"\n{'=' * 64}")
        difficulty = item.get("difficulty", "?")
        short_q = item.get("question", "")[:80]
        suffix = "..." if len(item.get("question", "")) > 80 else ""
        print(f"[Problem {problem_id} | {difficulty}] {short_q}{suffix}")
        print("=" * 64)

    try:
        io_data = parse_io_field(item.get("input_output"))
        total_cases = len(io_data.get("inputs", []))

        problem_text = format_apps_problem(item)
        pipeline_result = run_pipeline(problem_text)
        apps_test_results = run_apps_tests(
            pipeline_result["final_code"], io_data, max_cases=max_cases
        )

        result = build_apps_result_json(item, pipeline_result, apps_test_results)

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        if verbose:
            n_pass = apps_test_results.get("cases_passed", 0)
            n_run = apps_test_results.get("cases_run", 0)
            passed_label = "PASS" if apps_test_results.get("all_passed") else "FAIL"
            verdict = pipeline_result["final_report"].get("verdict", "?")
            retry_info = pipeline_result["final_report"].get("retry_info", {})
            leaf_ids = retry_info.get("all_leaf_ids", [])
            if retry_info.get("fullpath_replan_attempted"):
                replan_label = f"（已含根路径树重规划，涉及 {len(leaf_ids)} 个叶节点：{leaf_ids}）"
            else:
                replan_label = ""
            print(f"  APPS 测试  : {n_pass}/{n_run} 通过（共 {total_cases} 个用例）[{passed_label}]")
            print(f"  流水线裁决 : {verdict}{replan_label}")
            print(f"  保存至     : {output_file}")

        return True

    except Exception as exc:
        error_result: dict[str, Any] = {
            "problem_id": problem_id,
            "difficulty": item.get("difficulty", "unknown"),
            "url": item.get("url", ""),
            "question": item.get("question", ""),
            "timestamp": datetime.now().isoformat(),
            "ablation": "fullpath_replan_on_failure",
            "apps_test_results": None,
            "final_code": None,
            "agent_outputs": None,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(error_result, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"  [错误] Pipeline 异常：{exc}")
            print(f"  错误详情已保存至：{output_file}")

        return False


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "APPS 数据集批量测试 — 消融实验"
            "（含根路径树重规划：错误定位后沿含根祖先路径收集全部叶节点逐一调用 task_replanner）"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--offset",
        type=int,
        metavar="N",
        default=None,
        help="跳过前 N 条记录后开始读取（默认 0），需配合 --count 使用",
    )
    group.add_argument(
        "--ids",
        type=str,
        metavar="ID,ID,...",
        help="指定 problem id 列表，逗号分隔，如 0,5,10",
    )

    parser.add_argument(
        "--count",
        type=int,
        metavar="N",
        default=100,
        help="本次最多读取多少条题目（默认 100，与 --offset 配合使用）",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=list(VALID_DIFFICULTIES),
        default=None,
        metavar="LEVEL",
        help="按难度过滤：introductory | interview | competition（默认不过滤）",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        metavar="N",
        default=10,
        help="每道题最多运行的测试用例数（默认 10，避免数百个用例导致超时）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        metavar="DIR",
        help=f"结果输出目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(APPS_DATA_PATH),
        metavar="PATH",
        help=f"APPS 数据集路径（默认：{APPS_DATA_PATH}）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="安静模式，减少终端输出",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[错误] 数据集文件不存在：{data_path}", file=sys.stderr)
        sys.exit(1)

    if args.ids:
        id_list = [int(x.strip()) for x in args.ids.split(",")]
        selected = load_apps_by_ids(data_path, id_list)
    else:
        offset = args.offset if args.offset is not None else 0
        selected = load_apps_batch(
            data_path,
            offset=offset,
            count=args.count,
            difficulty=args.difficulty,
        )

    if not selected:
        print(
            "[错误] 未选取到任何题目，请检查 --offset / --count / --ids / --difficulty 参数。",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output)
    verbose = not args.quiet
    max_cases = args.max_cases

    difficulty_label = f"（难度：{args.difficulty}）" if args.difficulty else ""
    print("[消融实验：含根路径树重规划（错误定位后沿含根祖先路径收集叶节点逐一调用 task_replanner）]")
    print(f"共选取 {len(selected)} 题{difficulty_label}  |  输出目录：{output_dir}")
    print(f"每题最多运行 {max_cases} 个测试用例")

    success_count = 0
    for item in selected:
        ok = run_single_apps_problem(
            item, output_dir, max_cases=max_cases, verbose=verbose
        )
        if ok:
            success_count += 1

    print(f"\n{'=' * 64}")
    print(f"完成：{success_count}/{len(selected)} 题成功（pipeline 无异常）")
    print(f"结果保存于：{output_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
