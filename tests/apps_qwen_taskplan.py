"""APPS 数据集批量测试脚本（消融实验：plan_agent + 单智能体直接生成完整程序）

消融实验版本：仅保留 plan_agent 生成任务树，再由单一 Qwen 智能体一次调用
直接输出完整可运行程序。去掉 algo_agents（多专用智能体）、assembler_agent、
final_verifier_agent。

流水线对比：
  完整版:   plan_agent → algo_agents → assembler_agent → final_verifier_agent
  消融版1:  plan_agent → algo_agents → assembler_agent              （run_apps_no_fv.py）
  本次消融: plan_agent → 单智能体（一次调用，直接输出完整程序）

APPS 数据集字段说明：
  - id           : 题目编号（0-based，共 5000 条）
  - question     : 题目描述（自然语言）
  - solutions    : JSON 字符串，包含参考答案列表
  - input_output : JSON 字符串，格式 {"inputs": [...], "outputs": [...]}
  - difficulty   : "introductory" | "interview" | "competition"
  - url          : 题目来源链接
  - starter_code : 起始代码模板（可能为空字符串）

Usage:
  # 从第 0 条开始读取 100 条（默认行为）
  python tests/run_apps_no_algo_no_fv.py --offset 0 --count 100

  # 从第 200 条开始读取 50 条
  python tests/run_apps_no_algo_no_fv.py --offset 200 --count 50

  # 按难度过滤
  python tests/run_apps_no_algo_no_fv.py --offset 0 --count 100 --difficulty introductory

  # 指定特定 id（逗号分隔）
  python tests/run_apps_no_algo_no_fv.py --ids 0,5,10

  # 每题最多运行 5 个测试用例（默认 10）
  python tests/run_apps_no_algo_no_fv.py --offset 0 --count 10 --max-cases 5

  # 自定义输出目录
  python tests/run_apps_no_algo_no_fv.py --offset 0 --count 10 --output result/apps_ablation_no_algo_no_fv_qwen

  # 安静模式
  python tests/run_apps_no_algo_no_fv.py --offset 0 --count 100 --quiet
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

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
os.environ["AUDITCODER_LLM_PROVIDER"] = "qwen"

from auditcoder.common.llm import get_llm
from auditcoder.common.utils import extract_json
from auditcoder.plan_agent.graph import build_plan_agent_graph

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------

APPS_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "apps_sampled_400.jsonl"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "result" / "apps_ablation_no_algo_no_fv_qwen"
)

VALID_DIFFICULTIES = {"introductory", "interview", "competition"}

# ---------------------------------------------------------------------------
# 单智能体 Prompt
# ---------------------------------------------------------------------------

SINGLE_AGENT_SYSTEM = """\
You are a senior competitive programming engineer. Given a structured problem analysis \
(task plan IR) produced by a planning agent, your job is to write a complete, directly \
runnable Python solution in one shot.

## Input

You will receive:
1. Problem information: title, description, input/output format, constraints, sample cases
2. Global solving strategy
3. A task decomposition tree: ordered leaf tasks with goals, function interfaces, \
   complexity budgets, and implementation hints
4. Execution order of tasks

## Requirements

1. Write a complete Python source file that:
   - Implements every leaf task as a standalone function (respecting the given interfaces)
   - Implements a top-level `solve()` function that calls the leaf functions in the \
     specified execution order and passes data between them correctly
   - Includes an `if __name__ == "__main__":` block that reads from stdin, calls \
     `solve()`, and prints the result to stdout
2. Follow the execution order: tasks listed earlier may be dependencies of later ones
3. Respect complexity budgets where possible
4. Code must be correct for the sample test cases
5. Do NOT import libraries beyond the Python standard library unless already implied \
   by the task hints
6. Do not output any debug information

## Output Format

Output a JSON object with exactly one field:
  final_code: the complete Python source code as a string

Output strictly valid JSON, no extra text."""


SINGLE_AGENT_USER = """\
## Problem Information

Title: {title}
Description:
{description}

Input format: {input_format}
Output format: {output_format}
Constraints: {constraints}

Sample cases:
{sample_cases}

## Global Solving Strategy

{global_strategy}

## Execution Order

{execution_order}

## Leaf Task Nodes

{task_nodes_text}

Please write a complete, runnable Python solution and output JSON: {{"final_code": "..."}}"""

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
    """按 id 列表从 JSONL 中精确读取指定记录。"""
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
# 单智能体代码生成
# ---------------------------------------------------------------------------


def _format_interface(interface: dict | None) -> str:
    if not interface:
        return "（接口未定义）"
    params = interface.get("params", [])
    param_str = ", ".join(
        f"{p.get('name')}: {p.get('type', 'Any')}" for p in params
    )
    return (
        f"def {interface.get('function_name', '?')}({param_str}) "
        f"-> {interface.get('return_type', 'Any')}"
    )


def _format_slots(slots: list[dict]) -> str:
    if not slots:
        return "（无）"
    return "; ".join(
        f"{s.get('name')}({s.get('type', '?')})"
        + (f" [来自 {s['source']}]" if s.get("source") else " [外部输入]")
        for s in slots
    )


def _format_constraints(constraints: dict | str) -> str:
    if not constraints:
        return "（无）"
    if isinstance(constraints, str):
        return constraints
    parts = []
    if constraints.get("time_limit"):
        parts.append(f"时间限制：{constraints['time_limit']}")
    if constraints.get("memory_limit"):
        parts.append(f"内存限制：{constraints['memory_limit']}")
    ranges = constraints.get("data_ranges", [])
    if ranges:
        range_strs = [
            f"{r.get('var')} ∈ [{r.get('min')}, {r.get('max')}]" for r in ranges
        ]
        parts.append("数据范围：" + ", ".join(range_strs))
    return "；".join(parts) if parts else json.dumps(constraints, ensure_ascii=False)


def _format_sample_cases(sample_cases: list[dict]) -> str:
    if not sample_cases:
        return "（无）"
    lines = []
    for i, case in enumerate(sample_cases, 1):
        lines.append(f"样例 {i}:")
        lines.append(f"  输入：{case.get('input', '')}")
        lines.append(
            f"  输出：{case.get('expected_output', case.get('output', ''))}"
        )
        if case.get("explanation"):
            lines.append(f"  解释：{case['explanation']}")
    return "\n".join(lines)


def _build_task_nodes_text(
    task_nodes: list[dict],
    execution_order: list[str],
) -> str:
    node_map: dict[str, dict] = {n["id"]: n for n in task_nodes}
    blocks = []
    for task_id in execution_order:
        node = node_map.get(task_id)
        if not node or not node.get("is_leaf"):
            continue
        hints = node.get("hints", [])
        hints_text = "\n".join(f"  - {h}" for h in hints) if hints else "  （无）"
        block = "\n".join([
            f"### Task {task_id}",
            f"Goal: {node.get('goal', '')}",
            f"Algorithm type: {node.get('type', '')}",
            f"Complexity budget: {node.get('complexity_budget', '')}",
            f"Function interface: {_format_interface(node.get('interface'))}",
            f"Inputs:  {_format_slots(node.get('inputs', []))}",
            f"Outputs: {_format_slots(node.get('outputs', []))}",
            f"Hints:\n{hints_text}",
        ])
        blocks.append(block)
    return "\n\n".join(blocks) if blocks else "（无叶节点）"


def generate_final_code_single_agent(task_plan_ir: dict[str, Any]) -> str:
    """使用单一 Qwen 智能体，一次 LLM 调用直接生成完整可运行程序。

    Args:
        task_plan_ir: plan_agent 输出的完整任务规划 IR。

    Returns:
        完整的 Python 源代码字符串。
    """
    problem: dict[str, Any] = task_plan_ir.get("problem", {})
    task_nodes: list[dict] = task_plan_ir.get("task_nodes", [])
    execution_order: list[str] = task_plan_ir.get("execution_order", [])

    constraints = problem.get("constraints", {})
    constraints_text = _format_constraints(constraints)
    sample_cases_text = _format_sample_cases(problem.get("sample_cases", []))
    task_nodes_text = _build_task_nodes_text(task_nodes, execution_order)

    user_content = SINGLE_AGENT_USER.format(
        title=problem.get("title", ""),
        description=problem.get("description", ""),
        input_format=problem.get("input_format", ""),
        output_format=problem.get("output_format", ""),
        constraints=constraints_text,
        sample_cases=sample_cases_text,
        global_strategy=task_plan_ir.get("global_strategy", ""),
        execution_order=" → ".join(execution_order) if execution_order else "（无）",
        task_nodes_text=task_nodes_text,
    )

    llm = get_llm(temperature=0.2)
    messages = [
        SystemMessage(content=SINGLE_AGENT_SYSTEM),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(messages)
    parsed = extract_json(response)

    if isinstance(parsed, dict):
        return parsed.get("final_code", "")
    return ""


# ---------------------------------------------------------------------------
# Pipeline 执行（消融版：plan_agent + 单智能体）
# ---------------------------------------------------------------------------


def run_pipeline(problem_text: str) -> dict[str, Any]:
    """运行消融版 pipeline（plan_agent → 单智能体直接生成完整程序）。"""
    plan_result = build_plan_agent_graph().invoke(
        {"problem_raw": problem_text, "revision_count": 0}
    )
    ir: dict[str, Any] = plan_result.get("task_plan_ir", {})

    final_code: str = generate_final_code_single_agent(ir)

    return {
        "ir": ir,
        "final_code": final_code,
    }


# ---------------------------------------------------------------------------
# JSON 结果构建
# ---------------------------------------------------------------------------


def build_apps_result_json(
    item: dict[str, Any],
    pipeline_result: dict[str, Any],
    apps_test_results: dict[str, Any],
) -> dict[str, Any]:
    """构建最终保存的 JSON 结构（消融版：只含 plan_agent + single_agent）。"""
    ir = pipeline_result["ir"]

    return {
        "problem_id": item["id"],
        "difficulty": item.get("difficulty", "unknown"),
        "url": item.get("url", ""),
        "question": item.get("question", ""),
        "timestamp": datetime.now().isoformat(),
        "ablation": "plan_agent_plus_single_agent_only",
        # 1. 测试结果
        "apps_test_results": apps_test_results,
        # 2. 完整代码
        "final_code": pipeline_result["final_code"],
        # 3. 各 agent 输出
        "agent_outputs": {
            "plan_agent": {
                "task_plan_ir": ir,
            },
            "single_agent": {
                "final_code": pipeline_result["final_code"],
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
    """对单道 APPS 题目运行消融版 pipeline 并将结果写入 JSON 文件。

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
            print(f"  APPS 测试  : {n_pass}/{n_run} 通过（共 {total_cases} 个用例）[{passed_label}]")
            print(f"  保存至     : {output_file}")

        return True

    except Exception as exc:
        error_result: dict[str, Any] = {
            "problem_id": problem_id,
            "difficulty": item.get("difficulty", "unknown"),
            "url": item.get("url", ""),
            "question": item.get("question", ""),
            "timestamp": datetime.now().isoformat(),
            "ablation": "plan_agent_plus_single_agent_only",
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
        description="APPS 数据集批量测试 — 消融实验（plan_agent + 单智能体直接生成完整程序）",
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
        help="每道题最多运行的测试用例数（默认 10）",
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
    print(f"[消融实验：plan_agent + Qwen 单智能体直接生成完整程序]")
    print(f"共选取 {len(selected)} 题{difficulty_label}  |  输出目录：{output_dir}")
    print(f"每题最多运行 {max_cases} 个测试用例")
    print(f"模型：Qwen / {os.getenv('QWEN_MODEL', 'qwen-plus')}")

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
