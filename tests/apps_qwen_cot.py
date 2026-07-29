"""APPS 数据集 Qwen CoT（Chain-of-Thought）求解脚本

两阶段流程：
  阶段一（思考）：将题目发送给 Qwen，要求给出详细的解题思路
                  （算法选择、数据结构、边界情况、复杂度分析等），不输出代码。
  阶段二（编码）：将题目 + 阶段一的思考结果一起发送给 Qwen，
                  要求严格按照给定思路生成可执行的 Python 代码。

与直接求解版的区别：
  - 多出一次 API 调用（思考阶段）
  - 生成代码时携带完整的推理过程，减少算法选错的概率

APPS 数据集字段说明：
  - id           : 题目编号（0-based，共 5000 条）
  - question     : 题目描述（自然语言）
  - solutions    : JSON 字符串，包含参考答案列表
  - input_output : JSON 字符串，格式 {"inputs": [...], "outputs": [...]}，已为标准 stdin/stdout
  - difficulty   : "introductory" | "interview" | "competition"
  - url          : 题目来源链接
  - starter_code : 起始代码模板（可能为空字符串）

JSON 结果文件结构：
  - problem_id        : 题目编号
  - difficulty        : 难度
  - url               : 来源链接
  - question          : 题目描述
  - timestamp         : 运行时间
  - apps_test_results : stdin/stdout 测试结果（最终一次）
  - final_code        : 模型生成并提取出的可执行代码（最终一次）
  - thinking_response : 阶段一思考结果（最终一次）
  - code_response     : 阶段二代码生成结果（最终一次）
  - retry_count       : 实际重试次数（0 = 首次即通过或未重试）
  - attempts          : 所有尝试的完整记录列表（含首次）

Usage:
  # 从第 0 条开始读取 10 条
  python tests/apps_qwen_cot.py --offset 0 --count 10

  # 指定特定 id（逗号分隔）
  python tests/apps_qwen_cot.py --ids 0,5,10

  # 按难度过滤
  python tests/apps_qwen_cot.py --offset 0 --count 50 --difficulty introductory

  # 自定义输出目录
  python tests/apps_qwen_cot.py --offset 0 --count 10 --output result/apps_qwen_cot

  # 安静模式
  python tests/apps_qwen_cot.py --offset 0 --count 100 --quiet

  # 最多重试 3 次（默认 2 次）
  python tests/apps_qwen_cot.py --offset 0 --count 10 --max-retries 3

  # 禁用重试
  python tests/apps_qwen_cot.py --offset 0 --count 10 --max-retries 0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------

APPS_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "apps.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "result" / "apps_qwen_cot"

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
    """将 APPS 条目格式化为发送给模型的 prompt。"""
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
# CoT 两阶段 Qwen 调用
# ---------------------------------------------------------------------------

_THINKING_SYSTEM_PROMPT = """\
你是一名资深算法竞赛选手，请对给定的编程题进行深入分析，输出详细的解题思路。

要求：
1. 分析题意，明确输入输出规格及约束条件。
2. 识别核心算法类型（如动态规划、BFS/DFS、贪心、二分、数学等），并说明选择原因。
3. 设计数据结构与算法流程，写出关键步骤的伪代码或逻辑描述。
4. 分析边界情况（空输入、极大值、负数等）及处理方法。
5. 估算时间复杂度与空间复杂度。
6. 【重要】不要输出任何 Python 代码，只输出文字分析。
"""

_CODING_SYSTEM_PROMPT = """\
你是一名资深算法竞赛选手，请严格按照给定的解题思路，实现完整的 Python 代码。

要求：
1. 代码从标准输入（stdin）读取数据，用 print() 将结果输出到标准输出（stdout）。
2. 只输出一个完整的 Python 代码块（用 ```python ... ``` 包裹），不输出任何解释文字。
3. 代码必须能直接运行，不能依赖非标准库。
4. 严格遵循给定思路，不要随意更换算法策略。
5. 不要输出多余的调试信息或注释。
"""


_RETRY_THINKING_SYSTEM_PROMPT = """\
你是一名资深算法竞赛选手。你之前给出的解题思路和代码未能通过全部测试用例。
请根据失败的测试用例，重新分析并修正解题思路。

要求：
1. 仔细分析失败用例，找出代码逻辑的错误根源（边界处理、算法错误、溢出等）。
2. 如有必要，重新选择更合适的算法；如算法方向正确则指出实现层面的修正点。
3. 给出完整修正后的解题思路，不得省略关键步骤。
4. 【重要】不要输出任何 Python 代码，只输出文字分析。
"""


def _build_llm() -> Any:
    """构造 ChatOpenAI 实例（Qwen / DashScope 兼容 OpenAI 接口）。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("QWEN_MODEL", "qwen-plus"),
        base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        api_key=os.getenv("QWEN_API_KEY"),
        temperature=0.0,
    )


def call_qwen_thinking(problem_text: str) -> str:
    """阶段一：让 Qwen 对题目进行 CoT 思考，返回纯文字分析结果，不含代码。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _build_llm()
    messages = [
        SystemMessage(content=_THINKING_SYSTEM_PROMPT),
        HumanMessage(content=problem_text),
    ]
    response = llm.invoke(messages)
    return response.content


def call_qwen_coding(problem_text: str, thinking: str) -> str:
    """阶段二：将题目 + 思考结果一起发给 Qwen，生成最终 Python 代码。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _build_llm()

    user_content = (
        f"{problem_text}\n\n"
        "---\n"
        "【解题思路（请严格遵循）】\n"
        f"{thinking}\n"
        "---\n\n"
        "请根据以上解题思路，给出完整的 Python 代码（用 ```python ... ``` 包裹）。"
    )

    messages = [
        SystemMessage(content=_CODING_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(messages)
    return response.content


def call_qwen_thinking_retry(
    problem_text: str,
    prev_thinking: str,
    prev_code: str,
    error_summary: str,
) -> str:
    """Retry 阶段一：携带失败信息，要求 Qwen 重新修正解题思路。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _build_llm()
    user_content = (
        f"{problem_text}\n\n"
        "---\n"
        "【上一次解题思路】\n"
        f"{prev_thinking}\n"
        "---\n\n"
        "【上一次生成的代码】\n"
        "```python\n"
        f"{prev_code}\n"
        "```\n"
        "---\n\n"
        "【测试失败信息】\n"
        f"{error_summary}\n"
        "---\n\n"
        "请根据以上失败信息，重新分析并给出修正后的解题思路（不要输出代码）。"
    )
    messages = [
        SystemMessage(content=_RETRY_THINKING_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(messages)
    return response.content


def _build_error_summary(test_results: dict[str, Any], max_failures: int = 3) -> str:
    """从测试结果中提取失败用例的错误摘要，用于 retry 提示词。"""
    cases = test_results.get("cases", [])
    failed = [c for c in cases if not c["passed"]]
    if not failed:
        return "无失败用例信息。"

    total_failed = len(failed)
    shown = failed[:max_failures]
    lines = [f"共 {total_failed} 个用例失败（展示前 {len(shown)} 个）：\n"]
    for c in shown:
        lines.append(f"用例 {c['index'] + 1}：")
        lines.append(f"  输入    : {repr(c['input'][:300])}")
        lines.append(f"  期望输出: {repr(c['expected'][:300])}")
        lines.append(f"  实际输出: {repr(c['actual'][:300])}")
        if c.get("error"):
            lines.append(f"  错误信息: {c['error'][:400]}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 代码提取
# ---------------------------------------------------------------------------


def extract_code(response: str) -> str:
    """从模型回复中提取可执行的 Python 代码。

    优先级：
      1. ```python ... ``` 代码块
      2. ``` ... ``` 任意代码块
      3. 原始文本（保底）
    """
    python_block = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if python_block:
        return python_block.group(1).strip()

    generic_block = re.search(r"```\s*(.*?)```", response, re.DOTALL)
    if generic_block:
        return generic_block.group(1).strip()

    return response.strip()


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
# JSON 结果构建
# ---------------------------------------------------------------------------


def build_result_json(
    item: dict[str, Any],
    final_code: str,
    apps_test_results: dict[str, Any],
    thinking_response: str,
    code_response: str,
    *,
    retry_count: int = 0,
    attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """构建保存的 JSON 结构。"""
    return {
        "problem_id": item["id"],
        "difficulty": item.get("difficulty", "unknown"),
        "url": item.get("url", ""),
        "question": item.get("question", ""),
        "timestamp": datetime.now().isoformat(),
        "apps_test_results": apps_test_results,
        "final_code": final_code,
        "thinking_response": thinking_response,
        "code_response": code_response,
        "retry_count": retry_count,
        "attempts": attempts or [],
    }


# ---------------------------------------------------------------------------
# 单题运行
# ---------------------------------------------------------------------------


def run_single_problem(
    item: dict[str, Any],
    output_dir: Path,
    *,
    max_cases: int = 10,
    max_retries: int = 2,
    verbose: bool = True,
) -> bool:
    """对单道 APPS 题目执行 CoT 两阶段求解，将结果写入 JSON 文件。

    返回 True 表示执行成功（即使测试未全过），False 表示出现异常。
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

        # 阶段一：CoT 思考
        if verbose:
            print("  [阶段一] 正在进行 CoT 思考分析...")
        thinking_response = call_qwen_thinking(problem_text)

        if verbose:
            preview = thinking_response[:200].replace("\n", " ")
            print(f"  思考摘要  : {preview}...")

        # 阶段二：依据思考结果生成代码
        if verbose:
            print("  [阶段二] 正在依据思考结果生成代码...")
        code_response = call_qwen_coding(problem_text, thinking_response)

        final_code = extract_code(code_response)
        apps_test_results = run_apps_tests(final_code, io_data, max_cases=max_cases)

        attempts: list[dict[str, Any]] = [
            {
                "attempt": 1,
                "thinking_response": thinking_response,
                "code_response": code_response,
                "final_code": final_code,
                "apps_test_results": apps_test_results,
            }
        ]

        # Retry：测试未全通过时，收集错误信息重新生成思路和代码
        retry_count = 0
        while not apps_test_results["all_passed"] and retry_count < max_retries:
            retry_count += 1
            n_pass = apps_test_results.get("cases_passed", 0)
            n_run = apps_test_results.get("cases_run", 0)
            if verbose:
                print(
                    f"  [Retry {retry_count}/{max_retries}] "
                    f"测试未全通过（{n_pass}/{n_run}），收集错误信息重新生成..."
                )

            error_summary = _build_error_summary(apps_test_results)

            if verbose:
                print(f"  [Retry {retry_count}] 阶段一：重新进行 CoT 思考...")
            thinking_response = call_qwen_thinking_retry(
                problem_text, thinking_response, final_code, error_summary
            )

            if verbose:
                preview = thinking_response[:200].replace("\n", " ")
                print(f"  修正思路  : {preview}...")
                print(f"  [Retry {retry_count}] 阶段二：依据修正思路重新生成代码...")
            code_response = call_qwen_coding(problem_text, thinking_response)

            final_code = extract_code(code_response)
            apps_test_results = run_apps_tests(final_code, io_data, max_cases=max_cases)

            attempts.append(
                {
                    "attempt": retry_count + 1,
                    "thinking_response": thinking_response,
                    "code_response": code_response,
                    "final_code": final_code,
                    "apps_test_results": apps_test_results,
                }
            )

        result = build_result_json(
            item,
            final_code,
            apps_test_results,
            thinking_response,
            code_response,
            retry_count=retry_count,
            attempts=attempts,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        if verbose:
            n_pass = apps_test_results.get("cases_passed", 0)
            n_run = apps_test_results.get("cases_run", 0)
            passed_label = "PASS" if apps_test_results.get("all_passed") else "FAIL"
            retry_label = f"（重试 {retry_count} 次）" if retry_count else ""
            print(
                f"  APPS 测试  : {n_pass}/{n_run} 通过"
                f"（共 {total_cases} 个用例）[{passed_label}]{retry_label}"
            )
            print(f"  保存至     : {output_file}")

        return True

    except Exception as exc:
        error_result: dict[str, Any] = {
            "problem_id": problem_id,
            "difficulty": item.get("difficulty", "unknown"),
            "url": item.get("url", ""),
            "question": item.get("question", ""),
            "timestamp": datetime.now().isoformat(),
            "apps_test_results": None,
            "final_code": None,
            "thinking_response": None,
            "code_response": None,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(error_result, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"  [错误] 调用异常：{exc}")
            print(f"  错误详情已保存至：{output_file}")

        return False


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="APPS 数据集测试 - Qwen CoT 两阶段求解（先思考后编码）",
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
        "--max-retries",
        type=int,
        metavar="N",
        default=2,
        help="测试未全通过时的最大重试次数（默认 2，设为 0 禁用重试）",
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
    print(f"共选取 {len(selected)} 题{difficulty_label}  |  输出目录：{output_dir}")
    print(f"每题最多运行 {max_cases} 个测试用例")
    print(f"模型：{os.getenv('QWEN_MODEL', 'qwen-plus')}")
    max_retries = args.max_retries
    retry_label = f"最多重试 {max_retries} 次" if max_retries > 0 else "不重试"
    print(f"模式：CoT 两阶段（思考 → 编码），首次 2 次 API，{retry_label}（每次再 +2 次）")

    success_count = 0
    for item in selected:
        ok = run_single_problem(
            item, output_dir, max_cases=max_cases, max_retries=max_retries, verbose=verbose
        )
        if ok:
            success_count += 1

    print(f"\n{'=' * 64}")
    print(f"完成：{success_count}/{len(selected)} 题成功（无异常）")
    print(f"结果保存于：{output_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
