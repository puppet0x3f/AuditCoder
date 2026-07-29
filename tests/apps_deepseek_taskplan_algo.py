"""APPS 数据集批量测试脚本（消融实验：无 final_verifier_agent）

消融实验版本：去掉错误定位与重试模块（final_verifier_agent），
仅保留 plan_agent → algo_agents → assembler_agent 三阶段流水线。

完整流水线（run_apps.py）：
  plan_agent → algo_agents → assembler_agent → final_verifier_agent

本消融实验流水线：
  plan_agent → algo_agents → assembler_agent

APPS 数据集字段说明：
  - id           : 题目编号（0-based，共 5000 条）
  - question     : 题目描述（自然语言）
  - solutions    : JSON 字符串，包含参考答案列表
  - input_output : JSON 字符串，格式 {"inputs": [...], "outputs": [...]}，已为标准 stdin/stdout
  - difficulty   : "introductory" | "interview" | "competition"
  - url          : 题目来源链接
  - starter_code : 起始代码模板（可能为空字符串）

JSON 结果文件结构：
  1. apps_test_results  — stdin/stdout 测试结果
  2. final_code         — 生成的完整代码
  3. agent_outputs      — 各 agent 的详细输出（无 final_verifier_agent 字段）

Usage:
  # 从第 0 条开始读取 100 条（默认行为）
  python tests/run_apps_no_fv.py --offset 0 --count 100

  # 从第 200 条开始读取 50 条
  python tests/run_apps_no_fv.py --offset 200 --count 50

  # 按难度过滤：只读取 introductory 难度的前 100 条
  python tests/run_apps_no_fv.py --offset 0 --count 100 --difficulty introductory

  # 指定特定 id（逗号分隔）
  python tests/run_apps_no_fv.py --ids 0,5,10

  # 每题最多运行 5 个测试用例（默认 10）
  python tests/run_apps_no_fv.py --offset 0 --count 10 --max-cases 5

  # 自定义输出目录
  python tests/run_apps_no_fv.py --offset 0 --count 10 --output result/apps_ablation_no_fv_deepseek

  # 安静模式
  python tests/run_apps_no_fv.py --offset 0 --count 100 --quiet
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

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
os.environ["AUDITCODER_LLM_PROVIDER"] = "deepseek"

from auditcoder.algo_agents import dispatch_algo_agents
from auditcoder.assembler_agent import build_assembler_agent_graph
from auditcoder.plan_agent.graph import build_plan_agent_graph

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------

APPS_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "apps_sampled_400.jsonl"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "result" / "apps_ablation_no_fv_deepseek"
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
    """从 JSONL 文件按行流式读取，跳过 offset 条后取 count 条。

    当 difficulty 不为 None 时，只统计符合该难度的记录（offset/count 均在过滤后计算）。
    不会将整个文件加载进内存，适合大数据集分批处理。

    Args:
        path       : apps.jsonl 文件路径。
        offset     : 跳过的起始条数（过滤后计算）。
        count      : 本次最多读取条数。
        difficulty : 难度过滤，None 表示不过滤。

    Returns:
        符合条件的记录列表（长度 ≤ count）。
    """
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
    """按 id 列表从 JSONL 中精确读取指定记录，流式扫描，无需全量加载。

    Args:
        path : apps.jsonl 文件路径。
        ids  : 需要读取的 id 列表。

    Returns:
        按 ids 顺序返回的记录列表（不存在的 id 被跳过并打印警告）。
    """
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
    """解析 input_output 字段（可能是 JSON 字符串或已解析的字典）。

    Returns:
        {"inputs": [...], "outputs": [...]} 或 {"inputs": [], "outputs": []}
    """
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
    """将 APPS 条目格式化为 pipeline 可识别的题目描述。

    prompt 结构：
      1. 题目描述（原文）
      2. 起始代码（若有）
      3. 输入输出格式说明 + 前 sample_count 个样例

    Args:
        item         : APPS 数据集中的一条记录。
        sample_count : 写入 prompt 的样例 I/O 数量上限。
    """
    question: str = item.get("question", "").strip()
    starter_code: str = item.get("starter_code", "").strip()
    difficulty: str = item.get("difficulty", "unknown")

    io_data = parse_io_field(item.get("input_output"))
    inputs: list[str] = io_data.get("inputs", [])
    outputs: list[str] = io_data.get("outputs", [])

    lines: list[str] = []

    # ── 1. 难度标注 ────────────────────────────────────────────────────────
    lines += [f"【题目难度】{difficulty}", ""]

    # ── 2. 题目描述 ────────────────────────────────────────────────────────
    lines += ["【题目描述】", question, ""]

    # ── 3. 起始代码（若非空）──────────────────────────────────────────────
    if starter_code:
        lines += [
            "【起始代码模板】",
            "请在以下模板基础上实现完整逻辑（保留函数/类签名）：",
            "```python",
            starter_code,
            "```",
            "",
        ]

    # ── 4. 输入输出格式说明 + 样例 ─────────────────────────────────────────
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

    # ── 5. 要求 ────────────────────────────────────────────────────────────
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
    """用 subprocess 运行 final_code，逐条注入 stdin 并比对 stdout。

    APPS 的 input_output 字段已包含标准 stdin/stdout 格式，无需转换。
    为避免超时，最多运行 max_cases 条测试用例。

    Args:
        final_code : pipeline 生成的完整 Python 代码。
        io_data    : {"inputs": [...], "outputs": [...]}
        max_cases  : 最多运行的测试用例数量。

    Returns:
        {
          "all_passed"   : bool,
          "cases_run"    : int,
          "cases_passed" : int,
          "compile_error": str | None,
          "cases"        : [
            {
              "index"    : 测试用例序号（0-based）,
              "input"    : 注入的 stdin 文本,
              "expected" : 期望 stdout,
              "actual"   : 实际 stdout,
              "passed"   : bool,
              "error"    : stderr 或超时信息（通过时为 None）,
            },
            ...
          ]
        }
    """
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
# Pipeline 执行（消融版：去掉 final_verifier_agent）
# ---------------------------------------------------------------------------


def run_pipeline(problem_text: str) -> dict[str, Any]:
    """运行消融版 AuditCoder pipeline（无 final_verifier_agent）并返回各阶段结果。

    流程：plan_agent → algo_agents → assembler_agent
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

    return {
        "ir": ir,
        "code_snippets": code_snippets,
        "verification_reports": verification_reports,
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
    """构建最终保存的 JSON 结构（消融版，无 final_verifier_agent 字段）。

    字段顺序：
      基础信息 → apps_test_results → final_code → agent_outputs
    """
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
        "ablation": "no_final_verifier_agent",
        # 1. 测试结果
        "apps_test_results": apps_test_results,
        # 2. 完整代码
        "final_code": pipeline_result["final_code"],
        # 3. 各 agent 输出（不含 final_verifier_agent）
        "agent_outputs": {
            "plan_agent": {
                "task_plan_ir": ir,
            },
            "algo_agents": algo_outputs,
            "assembler_agent": {
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
            "ablation": "no_final_verifier_agent",
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
        description="APPS 数据集批量测试 — 消融实验（无 final_verifier_agent）",
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

    # 加载题目
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
    print(f"[消融实验：无 final_verifier_agent | DeepSeek]")
    print(f"共选取 {len(selected)} 题{difficulty_label}  |  输出目录：{output_dir}")
    print(f"每题最多运行 {max_cases} 个测试用例")
    print(f"模型：DeepSeek / {os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')}")

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
