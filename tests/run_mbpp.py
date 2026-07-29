"""MBPP 数据集批量测试脚本

通过完整 AuditCoder pipeline（plan_agent → algo_agents → assembler_agent → final_verifier_agent）
对 MBPP 数据集中的题目进行测试，并将结果保存为每题独立的 JSON 文件。

JSON 文件结构（按此顺序）：
  1. mbpp_test_results  — MBPP 断言测试结果
  2. final_code         — 生成的完整代码
  3. agent_outputs      — 各 agent 的详细输出

Usage:
  # 从 task_id=1 开始运行 5 题
  python tests/run_mbpp.py --start 1 --count 5

  # 运行 task_id 1~50
  python tests/run_mbpp.py --start 1 --end 50

  # 指定特定 task_id（逗号分隔）
  python tests/run_mbpp.py --ids 1,3,5,100

  # 自定义输出目录
  python tests/run_mbpp.py --start 1 --count 3 --output data/my_results

  # 安静模式（减少终端输出）
  python tests/run_mbpp.py --start 1 --count 10 --quiet
"""

from __future__ import annotations

import argparse
import ast
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

from auditcoder.algo_agents import dispatch_algo_agents
from auditcoder.assembler_agent import build_assembler_agent_graph
from auditcoder.final_verifier_agent import build_final_verifier_graph
from auditcoder.plan_agent.graph import build_plan_agent_graph

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------

MBPP_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "mbpp.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "results"


# ---------------------------------------------------------------------------
# MBPP 数据加载
# ---------------------------------------------------------------------------


def load_mbpp(path: Path) -> dict[int, dict[str, Any]]:
    """加载 MBPP 数据集，以 task_id 为键建立索引。"""
    problems: dict[int, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                problems[item["task_id"]] = item
    return problems


def select_problems(
    problems: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """根据 CLI 参数筛选题目列表。"""
    all_ids = sorted(problems.keys())

    if args.ids:
        selected_ids = [int(x.strip()) for x in args.ids.split(",")]
        missing = [i for i in selected_ids if i not in problems]
        if missing:
            print(f"[警告] 以下 task_id 不存在于数据集中，已跳过：{missing}")
        return [problems[i] for i in selected_ids if i in problems]

    start = args.start if args.start is not None else min(all_ids)
    end = args.end if args.end is not None else max(all_ids)

    selected = [problems[i] for i in all_ids if start <= i <= end]

    if args.count is not None:
        selected = selected[: args.count]

    return selected


# ---------------------------------------------------------------------------
# 问题格式化 — 工具函数
# ---------------------------------------------------------------------------


def extract_required_functions(test_list: list[str]) -> list[dict[str, str]]:
    """从 test_list 断言中提取被调用的函数名及其第一次调用时的完整参数示例。

    示例：
      "assert min_cost([[1,2],[3,4]], 2, 2) == 8"
      → {"name": "min_cost", "call_example": "min_cost([[1,2],[3,4]], 2, 2)"}
    """
    seen: set[str] = set()
    results: list[dict[str, str]] = []

    for assertion in test_list:
        m = re.search(r"assert\s+(\w+)\s*\(", assertion)
        if not m:
            continue
        func_name = m.group(1)
        if func_name in seen:
            continue
        seen.add(func_name)

        call_part = re.split(r"\s*==\s*", assertion.lstrip("assert").strip(), maxsplit=1)[0].strip()
        results.append({"name": func_name, "call_example": call_part})

    return results


def parse_function_signature(code: str, func_name: str) -> list[str]:
    """从 MBPP code 字段中找到 func_name 的定义，返回参数名列表（有序）。

    使用 ast 解析，安全处理 SyntaxError（如 Windows 换行或缩进问题）。
    若解析失败或找不到目标函数，返回空列表。

    示例（task 1）：
      code 含 "def min_cost(cost, m, n): ..."
      → ["cost", "m", "n"]
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # 部分 MBPP 代码有 Windows 换行（\r\n），统一替换后重试
        try:
            tree = ast.parse(code.replace("\r\n", "\n").replace("\r", "\n"))
        except SyntaxError:
            return []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return [arg.arg for arg in node.args.args]

    return []


def _infer_type_str(value: Any) -> str:
    """将一个 Python 值推断为类型注解字符串。

    规则：
      int/float/str/bool → 对应类型名
      list → 检查首元素，递归推断为 list[X] 或 list[list[X]]
      tuple → "tuple"
      其他 → "Any"
    """
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, list):
        if not value:
            return "list"
        elem_type = _infer_type_str(value[0])
        return f"list[{elem_type}]"
    return "Any"


def annotate_test_cases(
    test_list: list[str],
    func_name: str,
    param_names: list[str],
) -> dict[str, Any]:
    """解析 test_list，生成带命名参数的标注结构与函数签名。

    返回 dict，包含：
      - "annotated_calls"  : list[str]  每条断言的命名参数展示
      - "param_types"      : dict[str, str]  各参数推断类型
      - "return_type"      : str  返回值推断类型
      - "param_examples"   : dict[str, str]  各参数的示例值（取第一条断言）
      - "return_example"   : str  返回值示例（取第一条断言）
      - "typed_signature"  : str  带类型的函数签名字符串

    若 param_names 为空或某条断言解析失败，对应条目退化为原始断言文本。
    """
    annotated_calls: list[str] = []
    param_types: dict[str, str] = {}
    return_type: str = "Any"
    param_examples: dict[str, str] = {}
    return_example: str = ""

    for assertion in test_list:
        # 用 ast 解析断言：形如 assert func(...) == expected
        try:
            tree = ast.parse(assertion.strip())  # exec 模式，得到 Module
            if not tree.body or not isinstance(tree.body[0], ast.Assert):
                annotated_calls.append(assertion)
                continue

            assert_node: ast.Assert = tree.body[0]
            expr = assert_node.test  # 取断言表达式：Compare 节点

            if not (
                isinstance(expr, ast.Compare)
                and len(expr.ops) == 1
                and isinstance(expr.ops[0], ast.Eq)
                and isinstance(expr.left, ast.Call)
            ):
                annotated_calls.append(assertion)
                continue

            call_node: ast.Call = expr.left
            expected_node = expr.comparators[0]

            # 确认是目标函数的调用（排除嵌套调用干扰）
            called_name = ""
            if isinstance(call_node.func, ast.Name):
                called_name = call_node.func.id
            if called_name != func_name:
                annotated_calls.append(assertion)
                continue

            # 只处理纯位置参数（无 keyword args）且数量匹配
            if call_node.keywords or len(call_node.args) != len(param_names):
                annotated_calls.append(assertion)
                continue

            # 将每个位置参数反编译为源码字符串，同时 eval 出值用于类型推断
            named_parts: list[str] = []
            for pname, arg_node in zip(param_names, call_node.args):
                arg_src = ast.unparse(arg_node)
                named_parts.append(f"{pname}={arg_src}")

                # 类型推断（仅对第一条断言推断，后续沿用）
                if pname not in param_types:
                    try:
                        val = ast.literal_eval(arg_node)
                        param_types[pname] = _infer_type_str(val)
                        param_examples[pname] = arg_src
                    except (ValueError, TypeError):
                        param_types[pname] = "Any"
                        param_examples[pname] = arg_src

            # 返回值
            ret_src = ast.unparse(expected_node)
            if not return_example:
                try:
                    ret_val = ast.literal_eval(expected_node)
                    return_type = _infer_type_str(ret_val)
                    return_example = ret_src
                except (ValueError, TypeError):
                    return_type = "Any"
                    return_example = ret_src

            annotated_call = f"{func_name}({', '.join(named_parts)}) → {ret_src}"
            annotated_calls.append(annotated_call)

        except SyntaxError:
            annotated_calls.append(assertion)

    # 构建带类型注解的函数签名字符串
    sig_parts = [
        f"{pname}: {param_types.get(pname, 'Any')}" for pname in param_names
    ]
    typed_signature = f"def {func_name}({', '.join(sig_parts)}) -> {return_type}"

    return {
        "annotated_calls": annotated_calls,
        "param_types": param_types,
        "return_type": return_type,
        "param_examples": param_examples,
        "return_example": return_example,
        "typed_signature": typed_signature,
    }


# ---------------------------------------------------------------------------
# ACM I/O 转换工具
# ---------------------------------------------------------------------------


def format_value_as_acm_lines(value: Any) -> list[str]:
    """将 Python 值转换为 ACM stdin 的文本行列表（含集合尺寸前缀）。

    转换规则：
      bool/int/float/str  → 单行字符串
      list[scalar]        → 第1行：元素个数 N；第2行：空格分隔元素
      list[list[scalar]]  → 第1行："R C"；后 R 行：各行空格分隔
      tuple               → 同 list[scalar]（先行长度，后行元素）
      其他                 → repr 单行
    """
    if isinstance(value, bool):
        return [str(value)]
    if isinstance(value, (int, float, str)):
        return [str(value)]
    if isinstance(value, (list, tuple)):
        seq = list(value)
        if not seq:
            return ["0"]
        # 2D 矩阵：首元素也是 list/tuple
        if isinstance(seq[0], (list, tuple)):
            R = len(seq)
            C = len(seq[0])
            lines = [f"{R} {C}"]
            for row in seq:
                lines.append(" ".join(str(x) for x in row))
            return lines
        # 1D 序列
        return [str(len(seq)), " ".join(str(x) for x in seq)]
    return [repr(value)]


def _acm_read_instruction(ptype: str) -> str:
    """将类型字符串映射为 ACM 读取方式的文字说明。"""
    if ptype.startswith("list[list["):
        return '第1行 "R C" 表示矩阵行列数，后 R 行每行 C 个空格分隔的元素'
    if ptype.startswith("list["):
        return "第1行：元素个数 N；第2行：N 个空格分隔的元素"
    if ptype == "tuple":
        return "第1行：元素个数 N；第2行：N 个空格分隔的元素"
    if ptype == "bool":
        return "单行 True 或 False"
    return "单行值"


def convert_assertions_to_acm_tests(
    test_list: list[str],
    func_name: str,
    param_names: list[str],
) -> list[dict[str, Any]]:
    """将 MBPP 断言列表转换为 ACM stdin/stdout 测试用例列表。

    每条断言 "assert func(arg0, arg1, ...) == expected" 被解析后：
      - 各位置参数通过 format_value_as_acm_lines() 展开为多行文本（含尺寸前缀）
      - 所有参数的行依次拼接为 acm_input
      - 期望输出为 str(expected_value)（与 print(result) 默认行为一致）

    返回：
      [
        {
          "original_assertion": "assert ...",
          "acm_input": "3 3\\n1 2 3\\n...\\n2\\n2",
          "acm_output": "8",
        },
        ...
      ]

    若某条断言解析失败或参数数量不匹配，该条目标记为 conversion_failed。
    """
    results: list[dict[str, Any]] = []

    for assertion in test_list:
        entry: dict[str, Any] = {
            "original_assertion": assertion,
            "acm_input": "",
            "acm_output": "",
            "conversion_failed": False,
        }
        try:
            tree = ast.parse(assertion.strip())
            if not tree.body or not isinstance(tree.body[0], ast.Assert):
                entry["conversion_failed"] = True
                results.append(entry)
                continue

            expr = tree.body[0].test
            if not (
                isinstance(expr, ast.Compare)
                and len(expr.ops) == 1
                and isinstance(expr.ops[0], ast.Eq)
                and isinstance(expr.left, ast.Call)
            ):
                entry["conversion_failed"] = True
                results.append(entry)
                continue

            call_node: ast.Call = expr.left
            expected_node = expr.comparators[0]

            # 确认目标函数名
            called_name = call_node.func.id if isinstance(call_node.func, ast.Name) else ""
            if called_name != func_name:
                entry["conversion_failed"] = True
                results.append(entry)
                continue

            # 参数数量必须与签名匹配
            if call_node.keywords or len(call_node.args) != len(param_names):
                entry["conversion_failed"] = True
                results.append(entry)
                continue

            # 将每个参数值转换为 ACM 行，依次拼接
            all_lines: list[str] = []
            for arg_node in call_node.args:
                val = ast.literal_eval(arg_node)
                all_lines.extend(format_value_as_acm_lines(val))

            entry["acm_input"] = "\n".join(all_lines)

            # 期望输出：直接用 Python str() 与 print() 默认行为一致
            expected_val = ast.literal_eval(expected_node)
            entry["acm_output"] = str(expected_val)

        except Exception:
            entry["conversion_failed"] = True

        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# 问题格式化 — 主函数
# ---------------------------------------------------------------------------


def format_mbpp_problem(item: dict[str, Any]) -> str:
    """将 MBPP 条目格式化为 pipeline 可识别的题目描述。

    prompt 结构：
      1. 【强制约束】区块 — 函数名不可更改
      2. 【函数接口】区块 — 带类型签名 + 参数说明
      3. 【输入输出格式】区块 — ACM stdin/stdout 格式说明 + 样例 I/O
      4. 题目描述（原文）
      5. 要求
    """
    text = item.get("text", "")
    test_list: list[str] = item.get("test_list", [])
    code_ref: str = item.get("code", "")
    setup = item.get("test_setup_code", "").strip()

    required_funcs = extract_required_functions(test_list)

    lines: list[str] = []

    # ── 1. 强制约束区块 ────────────────────────────────────────────────
    if required_funcs:
        lines += [
            "【强制约束 — 函数名不可更改】",
            "测试脚本将直接通过以下函数名调用你的代码，函数名必须与下方完全一致（大小写、拼写均不得修改）：",
        ]
        for f in required_funcs:
            lines.append(f"  ▸ 函数名：{f['name']}")
        lines += [
            "若生成的代码中不存在上述函数名，测试将直接失败。",
            "即使你将功能拆分为多个内部辅助函数，最终也必须暴露上述名称的公开入口函数。",
            "",
        ]

    # ── 2 & 3. 函数接口 + ACM I/O 区块 ──────────────────────────────
    for f in required_funcs:
        func_name = f["name"]
        param_names = parse_function_signature(code_ref, func_name)

        if not param_names:
            continue

        annotation = annotate_test_cases(test_list, func_name, param_names)
        acm_tests = convert_assertions_to_acm_tests(test_list, func_name, param_names)

        # ── 2. 函数接口（签名 + 参数说明）────────────────────────────
        lines += [
            f"【必须实现的函数接口 — {func_name}】",
            f"函数签名：{annotation['typed_signature']}",
            "",
            "参数说明（从测试用例推断）：",
        ]
        for pname in param_names:
            ptype = annotation["param_types"].get(pname, "Any")
            pexample = annotation["param_examples"].get(pname, "")
            lines.append(f"  - {pname:<12} {ptype:<24}  示例值: {pexample}")
        lines.append(
            f"  - {'返回值':<10} {annotation['return_type']:<24}  示例值: {annotation['return_example']}"
        )
        lines.append("")

        # ── 3. ACM 输入输出格式 ────────────────────────────────────
        param_order = " → ".join(param_names)
        lines += [
            "【输入输出格式（标准输入 → 标准输出）】",
            f"代码须从标准输入（stdin）按如下顺序读取参数：{param_order}",
            "将最终结果打印到标准输出（stdout）一行，直接 print(result) 即可。",
            "",
            "各参数的读取方式：",
        ]
        for pname in param_names:
            ptype = annotation["param_types"].get(pname, "Any")
            instr = _acm_read_instruction(ptype)
            lines.append(f"  - {pname} ({ptype}) : {instr}")
        lines.append("")

        # 样例 I/O（取所有可成功转换的条目）
        valid_tests = [t for t in acm_tests if not t.get("conversion_failed")]
        for idx, t in enumerate(valid_tests, 1):
            lines.append(f"样例 {idx}：")
            lines.append("  输入：")
            for il in t["acm_input"].splitlines():
                lines.append(f"    {il}")
            lines.append(f"  输出：{t['acm_output']}")
            lines.append("")

    # ── 4. 题目描述 ────────────────────────────────────────────────────
    lines += [f"题目：{text}", ""]

    if setup:
        lines += ["初始化代码（测试前会先执行）：", "```python", setup, "```", ""]

    # 降级：无法解析接口时，保留原始断言
    if not required_funcs or not parse_function_signature(
        code_ref, required_funcs[0]["name"]
    ):
        if test_list:
            lines += ["测试用例（需通过以下所有断言）："]
            for assertion in test_list:
                lines.append(f"  {assertion}")
            lines.append("")

    lines += [
        "要求：",
        "  - 实现上述函数，函数名与【强制约束】中完全一致。",
        "  - 代码从标准输入按上述格式读取，将结果打印到标准输出（print(result)）。",
        "  - 代码必须通过全部样例，应简洁、正确、高效。",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MBPP 测试执行（subprocess + ACM I/O）
# ---------------------------------------------------------------------------


def run_mbpp_tests(
    final_code: str,
    acm_tests: list[dict[str, Any]],
) -> dict[str, Any]:
    """用 subprocess 运行 final_code，逐条注入 ACM stdin 并比对 stdout。

    将 final_code 写入临时文件后，对每条 ACM 测试用例执行：
      python <tmp.py> < acm_input  →  比对 stdout.strip() == acm_output.strip()

    Args:
        final_code: pipeline 生成的完整 Python 代码（含 solve() + main block）。
        acm_tests : convert_assertions_to_acm_tests() 返回的测试列表。

    Returns:
        {
          "all_passed": bool,
          "compile_error": str | None,
          "cases": [
            {
              "assertion"  : 原始断言字符串,
              "acm_input"  : 注入的 stdin 文本,
              "expected"   : 期望 stdout 输出,
              "actual"     : 实际 stdout 输出,
              "passed"     : bool,
              "error"      : stderr 或超时信息（通过时为 None）,
            },
            ...
          ]
        }
    """
    if not acm_tests:
        return {"all_passed": False, "compile_error": None, "cases": []}

    # 将 final_code 写入临时文件（删除策略：手动 unlink，兼容 Windows）
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(final_code)

        cases: list[dict[str, Any]] = []
        all_passed = True

        for t in acm_tests:
            # 转换失败的条目直接标记失败，不执行
            if t.get("conversion_failed"):
                cases.append(
                    {
                        "assertion": t["original_assertion"],
                        "acm_input": "",
                        "expected": "",
                        "actual": "",
                        "passed": False,
                        "error": "conversion_failed: 无法将断言转换为 ACM 格式",
                    }
                )
                all_passed = False
                continue

            try:
                proc = subprocess.run(
                    [sys.executable, tmp_path],
                    input=t["acm_input"],
                    text=True,
                    capture_output=True,
                    timeout=10,
                )
                actual = proc.stdout.strip()
                expected = t["acm_output"].strip()
                passed = actual == expected
                error = proc.stderr.strip() if proc.returncode != 0 else None
            except subprocess.TimeoutExpired:
                actual = ""
                expected = t["acm_output"].strip()
                passed = False
                error = "timeout: 执行超过 10 秒"
            except Exception as exc:
                actual = ""
                expected = t["acm_output"].strip()
                passed = False
                error = f"{type(exc).__name__}: {exc}"

            if not passed:
                all_passed = False

            cases.append(
                {
                    "assertion": t["original_assertion"],
                    "acm_input": t["acm_input"],
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

    return {"all_passed": all_passed, "compile_error": None, "cases": cases}


# ---------------------------------------------------------------------------
# Pipeline 执行
# ---------------------------------------------------------------------------


def run_pipeline(problem_text: str) -> dict[str, Any]:
    """运行完整 AuditCoder pipeline 并返回各阶段结果。"""
    # Step 1: plan_agent — 题目分析与任务规划
    plan_result = build_plan_agent_graph().invoke(
        {"problem_raw": problem_text, "revision_count": 0}
    )
    ir: dict[str, Any] = plan_result.get("task_plan_ir", {})

    # Step 2: algo_agents — 各子任务代码生成与片段级验证
    code_snippets, verification_reports = dispatch_algo_agents(ir)

    # Step 3: assembler_agent — 拼装完整可运行代码
    assembler_result = build_assembler_agent_graph().invoke(
        {"task_plan_ir": ir, "code_snippets": code_snippets}
    )
    final_code: str = assembler_result.get("final_code", "")

    # Step 4: final_verifier_agent — 整体运行验证（含可选 retry）
    final_result = build_final_verifier_graph().invoke(
        {
            "final_code": final_code,
            "task_plan_ir": ir,
            "code_snippets": code_snippets,
            "verification_reports": verification_reports,
        }
    )

    return {
        "ir": ir,
        "code_snippets": code_snippets,
        "verification_reports": verification_reports,
        "final_code": final_code,
        "final_report": final_result.get("final_report", {}),
        "execution_result": final_result.get("execution_result", {}),
    }


# ---------------------------------------------------------------------------
# JSON 结果构建
# ---------------------------------------------------------------------------


def build_result_json(
    item: dict[str, Any],
    pipeline_result: dict[str, Any],
    mbpp_test_results: dict[str, Any],
) -> dict[str, Any]:
    """构建最终保存的 JSON 结构。

    字段顺序（按需求）：
      基础信息 → mbpp_test_results → final_code → agent_outputs
    """
    ir = pipeline_result["ir"]
    code_snippets = pipeline_result["code_snippets"]
    verification_reports = pipeline_result["verification_reports"]

    # 各 algo_agent 的片段与验证报告
    algo_outputs: dict[str, Any] = {}
    for task_id, snippet in code_snippets.items():
        algo_outputs[task_id] = {
            "snippet": snippet,
            "verification_report": verification_reports.get(task_id, {}),
        }

    # 使用有序字典保证 JSON 字段顺序
    result: dict[str, Any] = {
        "task_id": item["task_id"],
        "problem_text": item.get("text", ""),
        "test_list": item.get("test_list", []),
        "timestamp": datetime.now().isoformat(),
        # 1. 代码测试结果
        "mbpp_test_results": mbpp_test_results,
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

    return result


# ---------------------------------------------------------------------------
# 单题运行
# ---------------------------------------------------------------------------


def run_single_problem(
    item: dict[str, Any],
    output_dir: Path,
    *,
    verbose: bool = True,
) -> bool:
    """对单道 MBPP 题目运行完整 pipeline 并将结果写入 JSON 文件。

    返回 True 表示执行成功（即使 MBPP 测试未全过），False 表示 pipeline 本身出现异常。
    """
    task_id: int = item["task_id"]
    output_file = output_dir / f"mbpp_{task_id}.json"

    if verbose:
        print(f"\n{'=' * 64}")
        short_text = item.get("text", "")[:80]
        print(f"[Task {task_id}] {short_text}{'...' if len(item.get('text', '')) > 80 else ''}")
        print("=" * 64)

    try:
        # 在运行 pipeline 前，先从 item 提取参数名并生成 ACM 测试用例，
        # 以便 run_mbpp_tests() 使用 subprocess 进行 stdin/stdout 比对。
        test_list = item.get("test_list", [])
        code_ref = item.get("code", "")
        req_funcs = extract_required_functions(test_list)
        if req_funcs:
            _fn = req_funcs[0]["name"]
            _pnames = parse_function_signature(code_ref, _fn)
            acm_tests = convert_assertions_to_acm_tests(test_list, _fn, _pnames)
        else:
            acm_tests = []

        problem_text = format_mbpp_problem(item)
        pipeline_result = run_pipeline(problem_text)
        mbpp_test_results = run_mbpp_tests(pipeline_result["final_code"], acm_tests)

        result = build_result_json(item, pipeline_result, mbpp_test_results)

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        if verbose:
            cases = mbpp_test_results.get("cases", [])
            n_pass = sum(1 for c in cases if c["passed"])
            n_total = len(cases)
            passed_label = "PASS" if mbpp_test_results.get("all_passed") else "FAIL"
            verdict = pipeline_result["final_report"].get("verdict", "?")
            print(f"  MBPP 测试  : {n_pass}/{n_total} 通过 [{passed_label}]")
            print(f"  流水线裁决 : {verdict}")
            if mbpp_test_results.get("compile_error"):
                print(f"  编译错误   : {mbpp_test_results['compile_error']}")
            print(f"  保存至     : {output_file}")

        return True

    except Exception as exc:
        error_result: dict[str, Any] = {
            "task_id": task_id,
            "problem_text": item.get("text", ""),
            "test_list": item.get("test_list", []),
            "timestamp": datetime.now().isoformat(),
            "mbpp_test_results": None,
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
        description="MBPP 数据集批量测试 — 通过 AuditCoder pipeline 生成代码并验证",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--start",
        type=int,
        metavar="N",
        help="起始 task_id（含），需配合 --end 或 --count 使用",
    )
    group.add_argument(
        "--ids",
        type=str,
        metavar="ID,ID,...",
        help="指定 task_id 列表，逗号分隔，如 1,3,5,100",
    )

    parser.add_argument(
        "--end",
        type=int,
        metavar="N",
        help="终止 task_id（含，默认为数据集最大值）",
    )
    parser.add_argument(
        "--count",
        type=int,
        metavar="N",
        help="最多运行多少题（从 --start 开始计数）",
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
        default=str(MBPP_DATA_PATH),
        metavar="PATH",
        help=f"MBPP 数据集路径（默认：{MBPP_DATA_PATH}）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="安静模式，减少终端输出",
    )

    args = parser.parse_args()

    # 加载数据集
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[错误] 数据集文件不存在：{data_path}", file=sys.stderr)
        sys.exit(1)

    problems = load_mbpp(data_path)
    selected = select_problems(problems, args)

    if not selected:
        print("[错误] 未选取到任何题目，请检查 --start / --end / --ids 参数。", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    verbose = not args.quiet

    print(f"共选取 {len(selected)} 题  |  输出目录：{output_dir}")

    success_count = 0
    for item in selected:
        ok = run_single_problem(item, output_dir, verbose=verbose)
        if ok:
            success_count += 1

    print(f"\n{'=' * 64}")
    print(f"完成：{success_count}/{len(selected)} 题成功（pipeline 无异常）")
    print(f"结果保存于：{output_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
