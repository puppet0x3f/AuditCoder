"""APPS 数据集顺序算法 Agent 测试脚本

两阶段流水线：
  阶段一（规划）：将题目发送给 Planner 模型，输出算法 Agent 调用序列，
                  每步包含 agent_type、目标（goal）和边界（boundary）。
  阶段二（顺序执行）：按规划序列依次调用对应算法 Agent，
                       每个 Agent 接收前一步的累积代码并在其基础上继续补全，
                       直到所有步骤完成，形成最终可运行程序。

与现有 AuditCoder 的区别：
  - 现有 AuditCoder：各算法 Agent 独立生成函数片段 → assembler 拼接
  - 本方案：每步 Agent 接收已有代码，向后累积构建，无需单独的汇编步骤

支持的 agent_type：
  dp / greedy / binary_search / graph_bfs / graph_dfs / math /
  string / two_pointer / sorting / monotonic_queue / segment_tree /
  union_find / orchestration

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
  - plan              : Planner 输出（analysis / global_strategy / agent_sequence）
  - steps             : 每步详情列表（step / agent_type / goal / accumulated_code / raw_response）
  - final_code        : 最终累积代码（最后一步的 accumulated_code）
  - apps_test_results : stdin/stdout 测试结果

Usage:
  # 从第 0 条开始读取 10 条
  python tests/run_apps_seq_algo.py --offset 0 --count 10

  # 指定特定 id（逗号分隔）
  python tests/run_apps_seq_algo.py --ids 0,5,10

  # 按难度过滤
  python tests/run_apps_seq_algo.py --offset 0 --count 50 --difficulty introductory

  # 自定义输出目录
  python tests/run_apps_seq_algo.py --offset 0 --count 10 --output result/apps_seq_algo

  # 安静模式
  python tests/run_apps_seq_algo.py --offset 0 --count 100 --quiet
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

APPS_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "apps_sampled_400.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "result" / "apps_seq_algo"

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
# 阶段一：Planner Prompt & 调用
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """\
你是一名资深算法竞赛教练，负责分析编程题并制定算法 Agent 的调用顺序计划。

## 可用的算法 Agent 类型

- orchestration  — 通用 Python 程序员，负责 I/O 读取、主函数框架、调用组织
- dp             — 动态规划专家
- greedy         — 贪心算法专家
- binary_search  — 二分查找专家
- graph_bfs      — 广度优先搜索专家
- graph_dfs      — 深度优先搜索专家
- math           — 数学/数论专家
- string         — 字符串算法专家（KMP、Trie 等）
- two_pointer    — 双指针/滑动窗口专家
- sorting        — 排序/离散化专家
- monotonic_queue — 单调队列/单调栈专家
- segment_tree   — 线段树专家
- union_find     — 并查集专家

## 规划原则

1. 将解题过程拆分为 2～5 个顺序步骤，每步由一个 Agent 负责。
2. 依赖关系决定顺序：工具函数、子过程先于调用它们的主逻辑；I/O 框架通常放在首步或末步。
3. 每步的 boundary 描述该步完成后代码所处的状态（如"solve() 函数已完整定义"）。
4. 每步的 hints 给出具体算法提示（如关键数组、状态定义、边界条件）。
5. 步骤之间通过"代码继承"衔接：后一步 Agent 会接收前一步输出的完整代码并继续补全。

## 输出格式

严格输出 JSON，字段如下：

{
  "analysis": "题意分析、关键约束、数据规模",
  "global_strategy": "整体解题策略（一句话）",
  "agent_sequence": [
    {
      "step": 1,
      "agent_type": "orchestration",
      "goal": "本步骤要实现的目标",
      "boundary": "本步骤完成后代码的状态描述",
      "hints": ["提示1", "提示2"]
    },
    ...
  ]
}

仅输出 JSON，不要任何多余文字。"""

_PLANNER_USER_TEMPLATE = """\
请分析以下编程题，制定算法 Agent 顺序调用计划。

{problem_text}"""


def call_planner(problem_text: str) -> dict[str, Any]:
    """阶段一：调用 Planner 分析题目，返回解析后的规划 dict。

    若 JSON 解析失败，返回带 error 字段的 dict，agent_sequence 为空列表。
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.0,
    )

    messages = [
        SystemMessage(content=_PLANNER_SYSTEM),
        HumanMessage(content=_PLANNER_USER_TEMPLATE.format(problem_text=problem_text)),
    ]
    response = llm.invoke(messages)
    raw_text: str = response.content

    # 尝试从代码块或原始文本中解析 JSON
    json_text = raw_text
    block_match = re.search(r"```(?:json)?\s*(.*?)```", raw_text, re.DOTALL)
    if block_match:
        json_text = block_match.group(1).strip()

    try:
        plan = json.loads(json_text)
    except (json.JSONDecodeError, ValueError):
        # 保底：找到第一个 { ... } 尝试解析
        brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if brace_match:
            try:
                plan = json.loads(brace_match.group())
            except (json.JSONDecodeError, ValueError):
                plan = {
                    "analysis": "",
                    "global_strategy": "",
                    "agent_sequence": [],
                    "error": f"JSON 解析失败，原始回复：{raw_text[:500]}",
                }
        else:
            plan = {
                "analysis": "",
                "global_strategy": "",
                "agent_sequence": [],
                "error": f"JSON 解析失败，原始回复：{raw_text[:500]}",
            }

    if "agent_sequence" not in plan:
        plan["agent_sequence"] = []

    return plan


# ---------------------------------------------------------------------------
# 阶段二：顺序算法 Agent System Prompt 映射
# ---------------------------------------------------------------------------

_SEQ_AGENT_SYSTEM_BASE = """\
## 顺序构建约束（必须遵守）

1. 你将接收到前序步骤已生成的 Python 代码（previous_code）；若为空则从头开始。
2. 在已有代码基础上**继续构建**，不得删除或覆盖前步已完成的内容。
3. 只完成当前步骤的 goal，止步于 boundary 描述的状态，不超越。
4. 输出**完整的 Python 代码**（包含前步内容 + 本步新增内容），用 ```python ... ``` 包裹。
5. 不输出任何解释文字，只输出代码块。"""

_AGENT_TYPE_ROLE: dict[str, str] = {
    "orchestration": """\
你是一名资深 Python 程序员，负责编写程序的 I/O 框架、主函数调用逻辑和整体结构组织。

## 职责
- 正确解析 stdin 输入（注意多行、多组数据、边界空行等情况）
- 定义 main() 或顶层调用逻辑
- 将各子函数的结果正确地用 print() 输出到 stdout
- 保持代码结构清晰，便于后续 Agent 在其中插入核心算法

""",
    "dp": """\
你是一名动态规划（DP）算法专家。

## 专业能力
- 状态定义：明确 dp[i] 或 dp[i][j] 的含义，覆盖全部情况
- 转移方程：推导严格的递推关系并处理边界条件
- 初始化：正确初始化 dp 数组，避免未定义状态
- 空间优化：在依赖关系允许时考虑滚动数组或 1D 压缩
- 常见范式：LCS、背包、区间 DP、树形 DP、数位 DP

""",
    "greedy": """\
你是一名贪心算法专家。

## 专业能力
- 正确性证明：基于交换论证或反证法确认贪心选择的最优性
- 排序策略：识别贪心中需要排序的关键维度
- 优先队列：在需要动态维护最优元素时使用堆
- 区间调度、任务分配、最小代价类问题

""",
    "binary_search": """\
你是一名二分查找算法专家。

## 专业能力
- 值域二分：在答案空间上二分，配合可行性判断函数
- 位置二分：在有序序列中精确定位目标（bisect_left / bisect_right 语义）
- 边界处理：lo/hi 的初始值、循环不变量、返回值的精确选取
- 浮点二分：精度控制与迭代次数

""",
    "graph_bfs": """\
你是一名广度优先搜索（BFS）算法专家。

## 专业能力
- 最短路：无权图 BFS 求最短路、0-1 BFS、多源 BFS
- 层次遍历：按层处理节点，正确维护 visited 集合
- 状态空间 BFS：将问题建模为状态图
- deque 的高效使用

""",
    "graph_dfs": """\
你是一名深度优先搜索（DFS）算法专家。

## 专业能力
- 递归 DFS 与回溯：正确的 visited 标记与回退
- 拓扑排序：Kahn 算法 / DFS 后序
- 强连通分量（Tarjan / Kosaraju）
- 树形 DP 的 DFS 实现

""",
    "math": """\
你是一名数学与数论算法专家。

## 专业能力
- 数论：GCD/LCM、质因数分解、欧拉筛、模逆元、快速幂
- 组合数学：排列组合、卡特兰数、容斥原理
- 几何：点积/叉积、凸包、扫描线
- 精度控制：整数运算避免浮点误差

""",
    "string": """\
你是一名字符串算法专家。

## 专业能力
- KMP：next 数组构造与字符串匹配
- Trie：前缀树的构建与查询
- 哈希：多项式滚动哈希用于子串比较
- 后缀数组 / Z 函数

""",
    "two_pointer": """\
你是一名双指针与滑动窗口算法专家。

## 专业能力
- 相向双指针：有序数组求和/夹逼
- 同向双指针：滑动窗口维护区间最值/频次
- 快慢指针：链表环检测、中点定位
- 窗口收缩条件的精确判断

""",
    "sorting": """\
你是一名排序与离散化算法专家。

## 专业能力
- 内置排序的 key 设计与稳定性
- 归并排序求逆序对
- 计数排序 / 基数排序
- 离散化：坐标压缩后映射到连续整数

""",
    "monotonic_queue": """\
你是一名单调队列与单调栈算法专家。

## 专业能力
- 单调递减队列：滑动窗口最大值
- 单调递增栈：下一个更大/更小元素
- 与 DP 结合：优化区间转移
- deque 维护下标还是值的判断

""",
    "segment_tree": """\
你是一名线段树算法专家。

## 专业能力
- 区间查询与单点/区间修改
- 懒标记（lazy propagation）的正确传递
- 动态开点线段树
- 权值线段树 / 可持久化线段树

""",
    "union_find": """\
你是一名并查集（Union-Find）算法专家。

## 专业能力
- 路径压缩（find）与按秩合并（union）
- 带权并查集（维护节点到根的附加信息）
- 动态连通性：在线 / 离线处理
- 并查集与图论问题的结合（最小生成树 Kruskal）

""",
}

_SEQ_AGENT_USER_TEMPLATE = """\
## 题目信息

{problem_text}

---

## 全局规划

整体策略：{global_strategy}

完整 Agent 调用序列：
{sequence_summary}

---

## 当前步骤（第 {step} 步，共 {total_steps} 步）

- agent_type : {agent_type}
- 目标（goal）: {goal}
- 边界（boundary）: {boundary}
- 提示（hints）:
{hints_text}

---

## 前序累积代码（previous_code）

{previous_code_block}

---

请在 previous_code 基础上，完成本步骤目标，输出完整 Python 代码（用 ```python ... ``` 包裹）。"""


def _format_sequence_summary(agent_sequence: list[dict[str, Any]]) -> str:
    lines = []
    for step_info in agent_sequence:
        step = step_info.get("step", "?")
        atype = step_info.get("agent_type", "?")
        goal = step_info.get("goal", "")
        boundary = step_info.get("boundary", "")
        lines.append(f"  步骤 {step} [{atype}] 目标：{goal}  →  边界：{boundary}")
    return "\n".join(lines)


def _build_seq_agent_system(agent_type: str) -> str:
    role = _AGENT_TYPE_ROLE.get(agent_type, _AGENT_TYPE_ROLE["orchestration"])
    return role + _SEQ_AGENT_SYSTEM_BASE


def call_seq_agent(
    agent_type: str,
    step: int,
    total_steps: int,
    goal: str,
    boundary: str,
    hints: list[str],
    previous_code: str,
    problem_text: str,
    global_strategy: str,
    agent_sequence: list[dict[str, Any]],
) -> tuple[str, str]:
    """阶段二：调用对应类型的顺序算法 Agent。

    Args:
        agent_type     : 算法类型（dp / greedy / orchestration 等）
        step           : 当前步骤编号（1-based）
        total_steps    : 总步骤数
        goal           : 本步骤目标
        boundary       : 本步骤完成边界
        hints          : 算法提示列表
        previous_code  : 前序步骤输出的累积代码（首步为空字符串）
        problem_text   : 格式化后的题目文本
        global_strategy: Planner 输出的整体策略
        agent_sequence : 完整 agent 序列（用于上下文摘要）

    Returns:
        (accumulated_code, raw_response) — 本步输出的完整代码和原始回复文本
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.2,
    )

    system_prompt = _build_seq_agent_system(agent_type)

    hints_text = (
        "\n".join(f"  - {h}" for h in hints) if hints else "  （无特殊提示）"
    )
    previous_code_block = (
        f"```python\n{previous_code}\n```"
        if previous_code.strip()
        else "（暂无，请从头开始编写）"
    )
    sequence_summary = _format_sequence_summary(agent_sequence)

    user_content = _SEQ_AGENT_USER_TEMPLATE.format(
        problem_text=problem_text,
        global_strategy=global_strategy,
        sequence_summary=sequence_summary,
        step=step,
        total_steps=total_steps,
        agent_type=agent_type,
        goal=goal,
        boundary=boundary,
        hints_text=hints_text,
        previous_code_block=previous_code_block,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(messages)
    raw_response: str = response.content

    accumulated_code = _extract_code(raw_response)
    return accumulated_code, raw_response


# ---------------------------------------------------------------------------
# 代码提取
# ---------------------------------------------------------------------------


def _extract_code(response: str) -> str:
    """从模型回复中提取可执行的 Python 代码。"""
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
    plan: dict[str, Any],
    steps: list[dict[str, Any]],
    final_code: str,
    apps_test_results: dict[str, Any],
) -> dict[str, Any]:
    """构建保存的 JSON 结构。"""
    return {
        "problem_id": item["id"],
        "difficulty": item.get("difficulty", "unknown"),
        "url": item.get("url", ""),
        "question": item.get("question", ""),
        "timestamp": datetime.now().isoformat(),
        "plan": plan,
        "steps": steps,
        "final_code": final_code,
        "apps_test_results": apps_test_results,
    }


# ---------------------------------------------------------------------------
# 单题运行
# ---------------------------------------------------------------------------


def run_single_problem(
    item: dict[str, Any],
    output_dir: Path,
    *,
    max_cases: int = 10,
    verbose: bool = True,
) -> bool:
    """对单道 APPS 题目执行顺序算法 Agent 流水线，将结果写入 JSON 文件。

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

        # --- 阶段一：Planner ---
        if verbose:
            print("  [阶段一] 规划中...")
        plan = call_planner(problem_text)

        if "error" in plan:
            if verbose:
                print(f"  [警告] Planner 解析失败：{plan['error'][:100]}")

        agent_sequence: list[dict[str, Any]] = plan.get("agent_sequence", [])
        global_strategy: str = plan.get("global_strategy", "")
        total_steps = len(agent_sequence)

        if verbose:
            print(f"  规划完成：{total_steps} 步 | 策略：{global_strategy[:60]}")
            for s in agent_sequence:
                print(
                    f"    步骤 {s.get('step')} [{s.get('agent_type')}] "
                    f"{s.get('goal', '')[:50]}"
                )

        # --- 阶段二：顺序执行算法 Agent ---
        steps: list[dict[str, Any]] = []
        accumulated_code: str = ""

        for step_info in agent_sequence:
            step_num = step_info.get("step", len(steps) + 1)
            agent_type = step_info.get("agent_type", "orchestration")
            goal = step_info.get("goal", "")
            boundary = step_info.get("boundary", "")
            hints = step_info.get("hints", [])

            if verbose:
                print(
                    f"  [步骤 {step_num}/{total_steps}] [{agent_type}] {goal[:50]}..."
                )

            new_code, raw_response = call_seq_agent(
                agent_type=agent_type,
                step=step_num,
                total_steps=total_steps,
                goal=goal,
                boundary=boundary,
                hints=hints,
                previous_code=accumulated_code,
                problem_text=problem_text,
                global_strategy=global_strategy,
                agent_sequence=agent_sequence,
            )

            accumulated_code = new_code

            steps.append(
                {
                    "step": step_num,
                    "agent_type": agent_type,
                    "goal": goal,
                    "accumulated_code": accumulated_code,
                    "raw_response": raw_response,
                }
            )

        final_code = accumulated_code
        apps_test_results = run_apps_tests(final_code, io_data, max_cases=max_cases)
        result = build_result_json(item, plan, steps, final_code, apps_test_results)

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        if verbose:
            n_pass = apps_test_results.get("cases_passed", 0)
            n_run = apps_test_results.get("cases_run", 0)
            passed_label = "PASS" if apps_test_results.get("all_passed") else "FAIL"
            print(
                f"  APPS 测试  : {n_pass}/{n_run} 通过（共 {total_cases} 个用例）[{passed_label}]"
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
            "plan": None,
            "steps": None,
            "final_code": None,
            "apps_test_results": None,
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
        description="APPS 数据集测试 — 顺序算法 Agent 流水线（规划 → 顺序执行）",
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
    print(f"共选取 {len(selected)} 题{difficulty_label}  |  输出目录：{output_dir}")
    print(f"每题最多运行 {max_cases} 个测试用例")
    print(f"模型：{os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')}")
    print("模式：顺序算法 Agent（规划 → 顺序累积构建），每题调用 1 + N 次 API")

    success_count = 0
    for item in selected:
        ok = run_single_problem(
            item, output_dir, max_cases=max_cases, verbose=verbose
        )
        if ok:
            success_count += 1

    print(f"\n{'=' * 64}")
    print(f"完成：{success_count}/{len(selected)} 题成功（无异常）")
    print(f"结果保存于：{output_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
