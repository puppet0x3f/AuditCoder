"""APPS 数据集批量测试脚本（消融实验：全局重规划）

消融实验版本：修改错误重试策略。错误定位后不再只修正单个失败任务，
而是将全部错误信息拼入问题描述，交给 plan_agent 重新生成整棵任务树，
再重走 algo_agents → assembler_agent 完整流程。

流水线对比：
  原完整版:   plan_agent → algo_agents → assembler_agent → final_verifier_agent
              （重试：error_locator → task_replanner[单任务] → algo_retrier → code_reassembler）

  本次消融:   plan_agent → algo_agents → assembler_agent
              （重试：error_locator[收集全部错误] → plan_agent[全局重规划] → algo_agents → assembler_agent）

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
  python tests/run_apps_global_replan.py --offset 0 --count 100

  # 从第 200 条开始读取 50 条
  python tests/run_apps_global_replan.py --offset 200 --count 50

  # 按难度过滤
  python tests/run_apps_global_replan.py --offset 0 --count 100 --difficulty introductory

  # 指定特定 id（逗号分隔）
  python tests/run_apps_global_replan.py --ids 0,5,10

  # 每题最多运行 5 个测试用例（默认 10）
  python tests/run_apps_global_replan.py --offset 0 --count 10 --max-cases 5

  # 自定义输出目录
  python tests/run_apps_global_replan.py --offset 0 --count 10 --output result/apps_ablation_global_replan

  # 安静模式
  python tests/run_apps_global_replan.py --offset 0 --count 100 --quiet
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
from auditcoder.final_verifier_agent.nodes.error_locator import error_locator
from auditcoder.final_verifier_agent.nodes.full_code_runner import full_code_runner
from auditcoder.plan_agent.graph import build_partial_replan_graph, build_plan_agent_graph

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------

APPS_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "apps.jsonl"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "result" / "apps_ablation_global_replan"
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
# 错误反馈构建（拼入 problem_raw 供 plan_agent 全局感知）
# ---------------------------------------------------------------------------


def _build_error_feedback(
    error_location: dict[str, Any],
    task_plan_ir: dict[str, Any],
    verification_reports: dict[str, dict[str, Any]],
) -> str:
    """将完整任务规划 IR、各任务验证报告及错误定位信息格式化为可拼入 problem_raw 的文本段落。

    plan_agent 的 problem_parser → algorithm_selector 链会读取这段信息，
    在重新规划时全局感知上次的完整任务分解方案、每个任务的验证结果以及最终执行错误，
    从而做出更有针对性的全局调整。
    """
    lines = [
        "",
        "【上次尝试的完整规划与错误信息（请在重新规划时参考并避免以下问题）】",
    ]

    # ── 1. 上次的全局策略 ────────────────────────────────────────────────
    global_strategy = task_plan_ir.get("global_strategy", "")
    if global_strategy:
        lines += ["", "▶ 上次全局策略：", global_strategy]

    # ── 2. 上次的完整任务树 ──────────────────────────────────────────────
    task_nodes: list[dict] = task_plan_ir.get("task_nodes", [])
    execution_order: list[str] = task_plan_ir.get("execution_order", [])
    if task_nodes:
        lines += ["", "▶ 上次任务分解（执行顺序：{}）：".format(
            " → ".join(execution_order) if execution_order else "无"
        )]
        node_map = {n["id"]: n for n in task_nodes}
        for tid in execution_order:
            node = node_map.get(tid)
            if not node or not node.get("is_leaf"):
                continue
            iface = node.get("interface") or {}
            fn_name = iface.get("function_name", "?")
            agent_type = node.get("agent_type", "?")
            complexity = node.get("complexity_budget", "?")
            goal = node.get("goal", "")
            lines.append(
                f"  - 任务 {tid}（{agent_type}，函数 {fn_name}，复杂度 {complexity}）：{goal}"
            )

    # ── 3. 各任务的验证报告 ──────────────────────────────────────────────
    if verification_reports:
        lines += ["", "▶ 各任务验证结果："]
        for tid, report in verification_reports.items():
            verdict = report.get("verdict", "unknown")
            summary = report.get("summary", "")
            lines.append(f"  - 任务 {tid}：{verdict}  {summary[:200]}")

    # ── 4. 最终执行错误定位 ──────────────────────────────────────────────
    lines += ["", "▶ 最终执行错误："]
    lines.append(f"  错误类型：{error_location.get('error_type', 'unknown')}")

    detail = error_location.get("detail", "")
    if detail:
        lines.append(f"  错误详情：{detail[:400]}")

    failing_case = error_location.get("failing_case")
    if failing_case:
        lines.append(
            f"  失败用例：输入 {str(failing_case.get('input', ''))[:100]}，"
            f"期望输出 {str(failing_case.get('expected', ''))[:100]}，"
            f"实际输出 {str(failing_case.get('actual', ''))[:100]}"
        )

    task_id = error_location.get("task_id", "")
    agent_type = error_location.get("agent_type", "")
    method = error_location.get("method", "")
    fn = error_location.get("function_name", "")

    locator_parts = []
    if fn:
        locator_parts.append(f"函数 {fn}")
    if task_id:
        locator_parts.append(f"任务 {task_id}")
    if agent_type:
        locator_parts.append(f"算法类型 {agent_type}")
    if method:
        locator_parts.append(f"定位方法 {method}")
    if locator_parts:
        lines.append(f"  错误定位：{', '.join(locator_parts)}")

    lines += [
        "",
        "请根据上述完整规划方案和错误信息，重新分解任务、选择更合适的算法，"
        "并在 hints 中补充边界处理和正确性说明，避免重复上次的错误。",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 单轮流水线（plan_agent → algo_agents → assembler_agent）
# ---------------------------------------------------------------------------


def _run_single_pass(
    problem_text: str,
) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, Any]], str, dict[str, Any]]:
    """运行单轮 plan_agent → algo_agents → assembler_agent。

    Returns:
        (ir, code_snippets, verification_reports, final_code, plan_state)
        plan_state 保留 problem_structured / deep_thinking / constraint_analysis，供 partial replan 复用。
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

    return ir, code_snippets, verification_reports, final_code, plan_result


def _run_partial_replan(
    prior_plan_state: dict[str, Any],
    failure_context: str,
) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, Any]], str]:
    """Partial replan: 复用 problem_structured（题目解析不变），
    从 deep_thinker 重新开始，注入失败上下文让其自行诊断是算法错误还是实现错误，
    并据此决定是保留原算法方向还是切换新算法。

    Returns:
        (ir, code_snippets, verification_reports, final_code)
    """
    partial_state: dict[str, Any] = {
        "problem_structured": prior_plan_state.get("problem_structured", {}),
        "failure_context": failure_context,
        "revision_count": 0,
    }

    plan_result = build_partial_replan_graph().invoke(partial_state)
    ir: dict[str, Any] = plan_result.get("task_plan_ir", {})

    code_snippets, verification_reports = dispatch_algo_agents(ir)

    assembler_result = build_assembler_agent_graph().invoke(
        {"task_plan_ir": ir, "code_snippets": code_snippets}
    )
    final_code: str = assembler_result.get("final_code", "")

    return ir, code_snippets, verification_reports, final_code


# ---------------------------------------------------------------------------
# Pipeline 执行（改进版：partial replan + 多次重试）
# ---------------------------------------------------------------------------

MAX_REPLAN_RETRIES = 2


def run_pipeline(problem_text: str) -> dict[str, Any]:
    """运行改进版 global replan pipeline。

    改进点（相比原版）：
    1. Partial replan：retry 时复用 problem_structured，从 deep_thinker 重新开始，
       注入失败信息让其自行诊断根因（算法错误 vs 实现错误），避免盲目换算法。
    2. 多次重试：最多 MAX_REPLAN_RETRIES 次，而非只重试一次。
    """
    # ── 第一轮 ────────────────────────────────────────────────────────────
    ir, code_snippets, verification_reports, final_code, plan_state = _run_single_pass(
        problem_text
    )

    fv_state: dict[str, Any] = {
        "final_code": final_code,
        "task_plan_ir": ir,
        "code_snippets": code_snippets,
        "verification_reports": verification_reports,
    }
    runner_result = full_code_runner(fv_state)
    fv_state.update(runner_result)

    execution_result: dict[str, Any] = fv_state.get("execution_result", {})
    replan_count = 0
    error_location: dict[str, Any] | None = None

    # ── 重试循环（partial replan，最多 MAX_REPLAN_RETRIES 次）────────────
    while not execution_result.get("all_passed", False) and replan_count < MAX_REPLAN_RETRIES:
        locator_result = error_locator(fv_state)
        error_location = locator_result.get("error_location")
        if not error_location:
            break

        failure_context = _build_error_feedback(error_location, ir, verification_reports)

        replan_count += 1
        ir, code_snippets, verification_reports, final_code = _run_partial_replan(
            plan_state, failure_context
        )

        fv_state = {
            "final_code": final_code,
            "task_plan_ir": ir,
            "code_snippets": code_snippets,
            "verification_reports": verification_reports,
        }
        runner_result = full_code_runner(fv_state)
        fv_state.update(runner_result)
        execution_result = fv_state.get("execution_result", execution_result)

    # ── 构建 final_report ─────────────────────────────────────────────────
    all_passed = execution_result.get("all_passed", False)
    final_report: dict[str, Any] = {
        "verdict": "accept" if all_passed else "reject",
        "all_passed": all_passed,
        "error_type": execution_result.get("error_type", "none"),
        "time_ms": execution_result.get("time_ms", 0.0),
        "error_location": error_location,
        "retry_info": {
            "global_replan_attempted": replan_count > 0,
            "replan_count": replan_count,
            "retry_strategy": "partial_replan_with_failed_algo_exclusion",
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
    """构建最终保存的 JSON 结构（全局重规划消融版）。"""
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
        "ablation": "global_replan_on_failure",
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
    """对单道 APPS 题目运行全局重规划消融版 pipeline 并将结果写入 JSON 文件。

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
            replan_label = "（已全局重规划）" if retry_info.get("global_replan_attempted") else ""
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
            "ablation": "global_replan_on_failure",
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
        description="APPS 数据集批量测试 — 消融实验（全局重规划：错误定位后重跑 plan_agent）",
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
    print(f"[消融实验：全局重规划（错误定位后重跑 plan_agent）]")
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
