"""ClassEval 数据集 CoT（Chain-of-Thought）对比测试脚本

两阶段流程：
  阶段一（思考）：将类描述和所有方法接口发送给模型，要求给出详细的实现思路
                  （每个方法的算法选择、数据结构、边界情况处理等），不输出代码。
  阶段二（编码）：将类描述 + 阶段一的思考结果一起发送给模型，
                  要求按照给定思路实现完整的 Python 类。

与 AuditCoder pipeline（run_classeval.py）的区别：
  - 不使用任务分解和专用方法 Agent，而是用单次 CoT 调用生成完整类
  - 重试时根据失败方法名和测试输出重新思考并修正代码

JSON 结果文件结构：
  - task_id                 : ClassEval 题目 ID
  - class_name              : 类名
  - timestamp               : 运行时间
  - classeval_test_results  : 测试结果（all_passed, pass_count, method_pass_rates 等）
  - final_class_code        : 最终生成的类代码
  - thinking_response       : 阶段一思考结果（最终一次）
  - code_response           : 阶段二编码结果（最终一次）
  - retry_count             : 实际重试次数（0 = 首次即通过或未重试）
  - attempts                : 所有尝试的完整记录列表（含首次）

Usage:
  # 运行前 5 道题（index 0-4）
  python auditcoder_class/tests/run_classeval_cot.py --start 0 --count 5

  # 运行特定 ID
  python auditcoder_class/tests/run_classeval_cot.py --ids ClassEval_0,ClassEval_5

  # 自定义输出目录
  python auditcoder_class/tests/run_classeval_cot.py --start 0 --count 3 --output data/my_cot_results

  # 最多重试 3 次（默认 2 次）
  python auditcoder_class/tests/run_classeval_cot.py --start 0 --count 10 --max-retries 3

  # 禁用重试
  python auditcoder_class/tests/run_classeval_cot.py --start 0 --count 10 --max-retries 0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

CLASSEVAL_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "ClassEval_data.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "classeval_cot_results"

MAX_RETRIES_DEFAULT = 2

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------


def load_classeval(path: Path) -> dict[str, dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {item["task_id"]: item for item in data}


def select_problems(
    problems: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    all_ids = sorted(problems.keys())

    if args.ids:
        selected_ids = [s.strip() for s in args.ids.split(",")]
        return [problems[i] for i in selected_ids if i in problems]

    start_idx = args.start or 0
    end_idx = args.end if args.end is not None else len(all_ids)
    selected = all_ids[start_idx:end_idx]

    if args.count:
        selected = selected[: args.count]

    return [problems[i] for i in selected]


# ---------------------------------------------------------------------------
# 问题格式化
# ---------------------------------------------------------------------------


def _format_entry(entry: dict[str, Any]) -> str:
    """将 ClassEval 条目格式化为发送给模型的 prompt（思考和编码阶段共用）。"""
    lines: list[str] = []

    class_name = entry.get("class_name", "")
    class_description = entry.get("class_description", "").strip()
    class_constructor = entry.get("class_constructor", "").strip()
    fields = entry.get("fields", [])
    imports = entry.get("import_statement", [])
    methods_info = entry.get("methods_info", [])

    lines += [f"【类名】{class_name}", ""]

    if imports:
        lines += ["【依赖导入】"]
        for imp in imports:
            lines.append(f"  {imp}")
        lines.append("")

    if class_description:
        lines += ["【类描述】", class_description, ""]

    if class_constructor:
        lines += ["【构造方法（已提供，直接使用）】", "```python", class_constructor, "```", ""]

    if fields:
        lines += ["【实例字段】"]
        for field in fields:
            lines.append(f"  - {field}")
        lines.append("")

    lines += ["【需要实现的方法】"]
    for m in methods_info:
        method_name = m.get("method_name", "")
        if method_name == "__init__":
            continue
        method_desc = m.get("method_description", "").strip()
        lines.append(f"  ▶ {method_name}")
        if method_desc:
            for line in method_desc.splitlines():
                lines.append(f"    {line}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

_THINKING_SYSTEM_PROMPT = """\
你是一名资深 Python 工程师，请对给定的类实现任务进行深入分析，输出详细的实现思路。

要求：
1. 理解类的整体设计意图和各方法的职责。
2. 对每个方法，分析其输入输出规格、核心逻辑、所需数据结构和算法。
3. 识别方法间的依赖关系（哪些方法依赖其他方法的状态或返回值）。
4. 分析边界情况（空输入、类型异常、越界等）及处理方法。
5. 若方法有状态修改，说明对实例字段的影响。
6. 【重要】不要输出任何 Python 代码，只输出文字分析。
"""

_CODING_SYSTEM_PROMPT = """\
你是一名资深 Python 工程师，请严格按照给定的实现思路，实现完整的 Python 类。

要求：
1. 只输出完整的 Python 类代码块（用 ```python ... ``` 包裹），不输出任何解释文字。
2. 代码必须能直接运行，类定义外不要添加任何语句。
3. 严格遵循给定的实现思路，不要随意更改算法策略。
4. 包含类定义所需的所有方法（若已提供 __init__，直接照抄）。
5. 不要输出多余的调试信息或注释。
"""

_RETRY_THINKING_SYSTEM_PROMPT = """\
你是一名资深 Python 工程师。你之前给出的类实现未能通过全部测试用例。
请根据失败的方法和测试输出，重新分析并修正实现思路。

要求：
1. 仔细分析失败的方法，找出逻辑错误的根源（边界处理、算法错误、状态管理等）。
2. 如有必要，重新设计失败方法的实现思路；如思路正确则指出实现层面的修正点。
3. 给出完整修正后的实现思路，所有方法（含已通过的）都要描述，避免遗漏。
4. 【重要】不要输出任何 Python 代码，只输出文字分析。
"""

# ---------------------------------------------------------------------------
# LLM 调用
# ---------------------------------------------------------------------------


_API_RETRY_ATTEMPTS = 3
_API_RETRY_BASE_DELAY = 5.0  # seconds


def _build_llm() -> Any:
    import os
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("QWEN_MODEL", "qwen-plus"),
        base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/"),
        api_key=os.getenv("QWEN_API_KEY"),
        temperature=0.0,
        extra_body={"enable_thinking": False},
    )


def _invoke_with_retry(messages: list, label: str = "") -> str:
    """调用 LLM，遇到连接错误时最多重试 _API_RETRY_ATTEMPTS 次（指数退避）。"""
    llm = _build_llm()
    last_exc: Exception | None = None
    for attempt in range(1, _API_RETRY_ATTEMPTS + 1):
        try:
            return llm.invoke(messages).content
        except Exception as e:
            err_str = str(type(e).__name__)
            is_conn_err = any(
                kw in err_str or kw in str(e)
                for kw in ("ConnectionError", "APIConnectionError", "RemoteProtocolError",
                           "ConnectError", "ReadTimeout", "TimeoutError")
            )
            if not is_conn_err or attempt == _API_RETRY_ATTEMPTS:
                raise
            delay = _API_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"  [连接错误{' (' + label + ')' if label else ''}，{delay:.0f}s 后重试 {attempt}/{_API_RETRY_ATTEMPTS - 1}] {e}")
            time.sleep(delay)
            last_exc = e
    raise last_exc  # type: ignore[misc]


def call_thinking(problem_text: str) -> str:
    """阶段一：CoT 思考分析，不含代码。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=_THINKING_SYSTEM_PROMPT),
        HumanMessage(content=problem_text),
    ]
    return _invoke_with_retry(messages, label="thinking")


def call_coding(problem_text: str, thinking: str) -> str:
    """阶段二：依据思考结果生成完整 Python 类。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    user_content = (
        f"{problem_text}\n\n"
        "---\n"
        "【实现思路（请严格遵循）】\n"
        f"{thinking}\n"
        "---\n\n"
        "请根据以上实现思路，给出完整的 Python 类代码（用 ```python ... ``` 包裹）。"
    )
    messages = [
        SystemMessage(content=_CODING_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    return _invoke_with_retry(messages, label="coding")


def call_thinking_retry(
    problem_text: str,
    prev_thinking: str,
    prev_code: str,
    error_summary: str,
) -> str:
    """Retry 阶段一：携带失败信息，要求模型重新修正实现思路。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    user_content = (
        f"{problem_text}\n\n"
        "---\n"
        "【上一次实现思路】\n"
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
        "请根据以上失败信息，重新分析并给出修正后的实现思路（不要输出代码）。"
    )
    messages = [
        SystemMessage(content=_RETRY_THINKING_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    return _invoke_with_retry(messages, label="retry-thinking")


# ---------------------------------------------------------------------------
# 代码提取
# ---------------------------------------------------------------------------


def extract_class_code(response: str) -> str:
    """从模型回复中提取 Python 类代码块。"""
    python_block = re.search(r"```python\s*(.*?)```", response, re.DOTALL)
    if python_block:
        return python_block.group(1).strip()

    generic_block = re.search(r"```\s*(.*?)```", response, re.DOTALL)
    if generic_block:
        return generic_block.group(1).strip()

    return response.strip()


# ---------------------------------------------------------------------------
# 错误摘要构建（用于 retry prompt）
# ---------------------------------------------------------------------------


def _build_error_summary(eval_result: dict[str, Any]) -> str:
    """根据 classeval 评估结果构建 retry 提示词中的错误摘要。"""
    method_pass_rates: dict[str, bool] = eval_result.get("method_pass_rates", {})
    failed = [m for m, ok in method_pass_rates.items() if not ok]
    passed = [m for m, ok in method_pass_rates.items() if ok]

    lines: list[str] = []
    lines.append(
        f"共 {len(method_pass_rates)} 个方法，"
        f"{len(passed)} 个通过，{len(failed)} 个失败。"
    )
    if failed:
        lines.append(f"失败方法：{', '.join(failed)}")
    if passed:
        lines.append(f"通过方法：{', '.join(passed)}")

    stdout = eval_result.get("stdout", "")
    if stdout:
        relevant = [
            line for line in stdout.splitlines()
            if any(k in line for k in ("FAIL:", "ERROR:", "AssertionError", "Error", "Traceback"))
        ]
        if relevant:
            lines.append("\n测试输出（关键行）：")
            lines.extend(relevant[:20])

    stderr = eval_result.get("stderr", "").strip()
    if stderr:
        lines.append("\n标准错误：")
        lines.append(stderr[:500])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ClassEval 评估（与 run_classeval.py 保持一致）
# ---------------------------------------------------------------------------


def run_classeval_evaluation(
    final_class_code: str,
    entry: dict[str, Any],
) -> dict[str, Any]:
    """运行 ClassEval 完整测试套件及逐方法测试，返回评估结果。"""
    from auditcoder.verifier_agent.sandbox.executor import SandboxExecutor

    executor = SandboxExecutor(default_timeout_s=30.0)
    imports = entry.get("import_statement", [])
    test_code = entry.get("test", "")
    test_classes = entry.get("test_classes", [])
    methods_info = entry.get("methods_info", [])

    if not test_code:
        return {"all_passed": True, "test_count": 0, "pass_count": 0, "method_pass_rates": {}}

    script_parts = []
    if imports:
        script_parts.append("\n".join(imports))
    script_parts.append(final_class_code)
    script_parts.append("import unittest")
    script_parts.append(test_code)
    script_parts.append(f"""
if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls_name in [{', '.join(repr(c) for c in test_classes)}]:
        try:
            cls = globals()[cls_name]
            suite.addTests(loader.loadTestsFromTestCase(cls))
        except KeyError:
            pass
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    import sys
    sys.exit(0 if result.wasSuccessful() else 1)
""")
    script = "\n\n".join(script_parts)
    result = executor.run(script)
    all_passed = result.exit_code == 0 and not result.timed_out

    ran_match = re.search(r'Ran (\d+) test', result.stdout)
    test_count = int(ran_match.group(1)) if ran_match else 0

    method_pass_rates: dict[str, bool] = {}
    tc_to_method = {m.get("test_class", ""): m["method_name"] for m in methods_info}

    for tc_name in test_classes:
        method_name = tc_to_method.get(tc_name, tc_name)
        per_parts = []
        if imports:
            per_parts.append("\n".join(imports))
        per_parts.append(final_class_code)
        per_parts.append("import unittest")
        per_parts.append(test_code)
        per_parts.append(f"""
if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase({tc_name})
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    import sys
    sys.exit(0 if result.wasSuccessful() else 1)
""")
        per_script = "\n\n".join(per_parts)
        per_result = executor.run(per_script)
        method_pass_rates[method_name] = per_result.exit_code == 0 and not per_result.timed_out

    pass_count = sum(1 for v in method_pass_rates.values() if v)

    return {
        "all_passed": all_passed,
        "test_count": test_count,
        "pass_count": pass_count,
        "method_pass_rates": method_pass_rates,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


# ---------------------------------------------------------------------------
# 单题运行
# ---------------------------------------------------------------------------


def run_single_problem(
    entry: dict[str, Any],
    args: argparse.Namespace,
    max_retries: int = MAX_RETRIES_DEFAULT,
) -> dict[str, Any]:
    task_id = entry["task_id"]
    if not args.quiet:
        print(f"\n{'='*60}")
        print(f"[{task_id}] {entry.get('class_name', '')}")
        print(f"{'='*60}")

    result: dict[str, Any] = {"task_id": task_id, "class_name": entry.get("class_name", "")}

    try:
        problem_text = _format_entry(entry)

        # 阶段一：CoT 思考
        if not args.quiet:
            print("  [阶段一] CoT 思考分析...")
        thinking_response = call_thinking(problem_text)
        if not args.quiet:
            preview = thinking_response[:200].replace("\n", " ")
            print(f"  思考摘要  : {preview}...")

        # 阶段二：依据思考生成类代码
        if not args.quiet:
            print("  [阶段二] 依据思考生成类代码...")
        code_response = call_coding(problem_text, thinking_response)
        final_class_code = extract_class_code(code_response)

        eval_result = run_classeval_evaluation(final_class_code, entry)

        attempts: list[dict[str, Any]] = [
            {
                "attempt": 1,
                "thinking_response": thinking_response,
                "code_response": code_response,
                "final_class_code": final_class_code,
                "classeval_test_results": eval_result,
            }
        ]

        # Retry：测试未全通过时，携带失败信息重新思考并生成代码
        retry_count = 0
        while not eval_result.get("all_passed", False) and retry_count < max_retries:
            retry_count += 1
            n_pass = eval_result.get("pass_count", 0)
            n_methods = len(eval_result.get("method_pass_rates", {}))
            if not args.quiet:
                print(
                    f"  [Retry {retry_count}/{max_retries}] "
                    f"测试未全通过（{n_pass}/{n_methods} 方法），重新思考..."
                )

            error_summary = _build_error_summary(eval_result)

            thinking_response = call_thinking_retry(
                problem_text, thinking_response, final_class_code, error_summary
            )
            if not args.quiet:
                preview = thinking_response[:200].replace("\n", " ")
                print(f"  修正思路  : {preview}...")
                print(f"  [Retry {retry_count}] 重新生成类代码...")

            code_response = call_coding(problem_text, thinking_response)
            final_class_code = extract_class_code(code_response)
            eval_result = run_classeval_evaluation(final_class_code, entry)

            attempts.append(
                {
                    "attempt": retry_count + 1,
                    "thinking_response": thinking_response,
                    "code_response": code_response,
                    "final_class_code": final_class_code,
                    "classeval_test_results": eval_result,
                }
            )

        result["classeval_test_results"] = eval_result
        result["final_class_code"] = final_class_code
        result["thinking_response"] = thinking_response
        result["code_response"] = code_response
        result["retry_count"] = retry_count
        result["attempts"] = attempts
        result["timestamp"] = datetime.now().isoformat()

        passed = eval_result.get("all_passed", False)
        n_methods = len(eval_result.get("method_pass_rates", {}))
        n_pass = eval_result.get("pass_count", 0)
        status = "PASS" if passed else "FAIL"
        retry_label = f"（重试 {retry_count} 次）" if retry_count else ""
        if not args.quiet:
            print(f"Result: {status}  ({n_pass}/{n_methods} methods passed){retry_label}")

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        result["classeval_test_results"] = {"all_passed": False}
        if not args.quiet:
            print(f"ERROR: {e}")

    return result


def save_result(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    task_id = result["task_id"]
    safe_id = task_id.replace("/", "_")
    out_path = output_dir / f"{safe_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ClassEval 数据集对比测试 — CoT 两阶段（先思考后编码）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--start", type=int, default=None, help="起始 index（0-based）")
    parser.add_argument("--end", type=int, default=None, help="终止 index（exclusive）")
    parser.add_argument("--count", type=int, default=None, help="最多运行多少题")
    parser.add_argument("--ids", type=str, default=None, help="指定 task_id，逗号分隔")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES_DEFAULT,
        help=f"最大重试次数（默认 {MAX_RETRIES_DEFAULT}，设为 0 禁用重试）",
    )
    parser.add_argument("--quiet", action="store_true", help="安静模式")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR

    problems = load_classeval(CLASSEVAL_DATA_PATH)
    selected = select_problems(problems, args)

    if not selected:
        print("No problems selected.")
        return

    import os
    model_name = os.getenv("QWEN_MODEL", "qwen-plus")
    retry_label = f"最多重试 {args.max_retries} 次" if args.max_retries > 0 else "不重试"
    print(f"[CoT 两阶段（思考 → 编码）]")
    print(f"模型：{model_name}  |  {retry_label}")
    print(f"运行 {len(selected)} 题  →  {output_dir}")

    passed_count = 0
    for entry in selected:
        result = run_single_problem(entry, args, max_retries=args.max_retries)
        save_result(result, output_dir)
        if result.get("classeval_test_results", {}).get("all_passed", False):
            passed_count += 1

    print(f"\nDone. {passed_count}/{len(selected)} problems passed all tests.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
