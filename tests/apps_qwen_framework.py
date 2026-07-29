"""APPS 数据集框架优先算法 Agent 填充测试脚本

三阶段流水线：
  阶段一（规划）  : Planner 分析题目，输出含函数签名的任务列表（tasks）。
  阶段二（骨架）  : Framework Agent 依据 tasks 生成带桩函数的代码骨架，
                    每个算法函数以 # [TASK:Tx][agent_type] 标注存根。
  阶段三（填充）  : 各算法 Agent 依次接收当前代码，只填充自己负责的桩函数，
                    输出完整更新后的代码传给下一个 Agent。

与 run_apps_seq_algo.py 的区别：
  seq_algo   — orchestration Agent 边写骨架边写桩，后续 Agent 重写整个文件。
  本方案     — 专用 Framework Agent 写骨架，算法 Agent 只负责填充指定函数桩。

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
  - plan              : Planner 输出（analysis / global_strategy / tasks）
  - skeleton_code     : Framework Agent 生成的带桩骨架代码
  - algo_steps        : 每步填充详情（task_id / agent_type / function_name / code_after_fill / raw_response）
  - final_code        : 所有桩填充完毕后的最终代码
  - apps_test_results : stdin/stdout 测试结果

Usage:
  # 从第 0 条开始读取 10 条
  python tests/run_apps_framework.py --offset 0 --count 10

  # 指定特定 id（逗号分隔）
  python tests/run_apps_framework.py --ids 0,5,10

  # 按难度过滤
  python tests/run_apps_framework.py --offset 0 --count 50 --difficulty introductory

  # 自定义输出目录
  python tests/run_apps_framework.py --offset 0 --count 10 --output result/apps_framework

  # 安静模式
  python tests/run_apps_framework.py --offset 0 --count 100 --quiet
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

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)
os.environ["AUDITCODER_LLM_PROVIDER"] = "qwen"

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------

APPS_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "apps.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "result" / "apps_framework_qwen"

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
# 通用 LLM 工厂
# ---------------------------------------------------------------------------


def _build_llm(temperature: float = 0.0) -> Any:
    """构造 ChatOpenAI 实例（Qwen）。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("QWEN_MODEL", "qwen-plus"),
        base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        api_key=os.getenv("QWEN_API_KEY"),
        temperature=temperature,
    )


def _parse_json_response(raw_text: str, fallback_keys: dict[str, Any]) -> dict[str, Any]:
    """从模型回复中健壮地解析 JSON。

    解析优先级：
      1. ```json ... ``` 代码块内容
      2. ``` ... ``` 任意代码块内容
      3. 原始文本整体
      4. 正则找最外层 { ... }
    若全部失败则返回 fallback_keys（加 error 字段）。
    """
    candidates: list[str] = []

    for pattern in [r"```json\s*(.*?)```", r"```\s*(.*?)```"]:
        m = re.search(pattern, raw_text, re.DOTALL)
        if m:
            candidates.append(m.group(1).strip())

    candidates.append(raw_text.strip())

    brace_m = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if brace_m:
        candidates.append(brace_m.group())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue

    return {**fallback_keys, "error": f"JSON 解析失败，原始回复（前 500 字符）：{raw_text[:500]}"}


# ---------------------------------------------------------------------------
# 阶段一：Planner — 分析 + 任务列表（含函数签名）
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """\
你是一名资深算法竞赛教练，负责分析编程题并制定算法 Agent 的任务列表。

## 任务

将解题过程分解为 1～5 个独立的算法子函数任务，每个任务由一个专用算法 Agent 负责实现。

## 可用算法 Agent 类型

- dp             — 动态规划
- greedy         — 贪心
- binary_search  — 二分查找
- graph_bfs      — 广度优先搜索
- graph_dfs      — 深度优先搜索
- math           — 数学/数论
- string         — 字符串算法（KMP、Trie 等）
- two_pointer    — 双指针/滑动窗口
- sorting        — 排序/离散化
- monotonic_queue — 单调队列/单调栈
- segment_tree   — 线段树
- union_find     — 并查集

## 规划原则

1. 每个任务对应一个独立的 Python 函数（算法核心逻辑）；I/O 读取和主调用逻辑由 Framework Agent 统一处理，不需要列入任务。
2. 函数签名（function_signature）必须完整，包含参数类型注解和返回类型，如：
   `def longest_increasing_subseq(nums: list[int]) -> int:`
3. 若题目只需一个算法，tasks 只含一项即可；若需多个算法协作，按依赖顺序排列（被依赖的在前）。
4. hints 给出该 Agent 需要注意的关键算法细节。

## 输出格式

严格输出 JSON，不含任何多余文字：

{
  "analysis": "题意理解、关键约束、数据规模（2～4 句）",
  "global_strategy": "整体解题策略（一句话）",
  "tasks": [
    {
      "task_id": "T1",
      "agent_type": "math",
      "function_name": "compute_standing",
      "function_signature": "def compute_standing(n: int, k: int, t: int) -> int:",
      "goal": "根据数学分段公式计算时间 t 时站立的观众人数",
      "hints": ["t<=k 返回 t", "k<t<=n 返回 k", "t>n 返回 n+k-t"]
    }
  ]
}"""

_PLANNER_USER_TEMPLATE = """\
请分析以下编程题，制定算法 Agent 任务列表。

{problem_text}"""


def call_planner(problem_text: str) -> dict[str, Any]:
    """阶段一：分析题目，返回含 tasks 的规划 dict。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _build_llm(temperature=0.0)
    messages = [
        SystemMessage(content=_PLANNER_SYSTEM),
        HumanMessage(content=_PLANNER_USER_TEMPLATE.format(problem_text=problem_text)),
    ]
    response = llm.invoke(messages)
    plan = _parse_json_response(
        response.content,
        fallback_keys={"analysis": "", "global_strategy": "", "tasks": []},
    )
    if "tasks" not in plan:
        plan["tasks"] = []
    return plan


# ---------------------------------------------------------------------------
# 阶段二：Framework Agent — 生成带桩骨架
# ---------------------------------------------------------------------------

_FRAMEWORK_SYSTEM = """\
你是一名资深 Python 程序员，负责根据算法任务列表编写程序的整体框架骨架。

## 职责

1. 编写完整的顶层结构：必要的 import、所有算法函数的存根（stub）、main() 函数。
2. main() 必须完整实现：正确读取 stdin 输入，按顺序调用各算法函数，用 print() 输出结果到 stdout。
3. 每个算法函数（存根）使用以下固定格式：

   def function_name(params) -> return_type:
       # [TASK:task_id][agent_type] goal
       raise NotImplementedError

   其中：
   - # [TASK:T1][math] ... 是必须保留的标注行，算法 Agent 将通过它定位需要填充的函数。
   - raise NotImplementedError 是占位符，将被算法 Agent 替换为真实实现。
   - 函数签名必须与任务列表中的 function_signature 完全一致。

4. main() 中调用算法函数的逻辑必须正确（正确传参、处理返回值、输出结果）。
5. 不要在存根函数中做任何实质性计算，保留 raise NotImplementedError。

## 输出要求

- 只输出一个完整的 Python 代码块（用 ```python ... ``` 包裹）。
- 不输出任何解释文字。"""

_FRAMEWORK_USER_TEMPLATE = """\
## 题目信息

{problem_text}

---

## 整体解题策略

{global_strategy}

---

## 算法任务列表（需生成存根的函数）

{tasks_summary}

---

请生成带桩骨架代码。每个算法函数使用 # [TASK:task_id][agent_type] goal 标注，并以 raise NotImplementedError 占位。
main() 必须完整实现读取输入、调用函数、输出结果的逻辑。"""


def _format_tasks_summary(tasks: list[dict[str, Any]]) -> str:
    lines = []
    for t in tasks:
        lines.append(
            f"  [{t.get('task_id')}] {t.get('agent_type')} | {t.get('function_signature')} | {t.get('goal')}"
        )
        hints = t.get("hints", [])
        if hints:
            lines.append("    提示：" + "; ".join(hints))
    return "\n".join(lines)


def call_framework_agent(problem_text: str, plan: dict[str, Any]) -> tuple[str, str]:
    """阶段二：生成带桩骨架代码。

    Returns:
        (skeleton_code, raw_response)
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = _build_llm(temperature=0.0)
    tasks: list[dict[str, Any]] = plan.get("tasks", [])
    global_strategy: str = plan.get("global_strategy", "")

    user_content = _FRAMEWORK_USER_TEMPLATE.format(
        problem_text=problem_text,
        global_strategy=global_strategy,
        tasks_summary=_format_tasks_summary(tasks),
    )

    messages = [
        SystemMessage(content=_FRAMEWORK_SYSTEM),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(messages)
    raw_response: str = response.content
    skeleton_code = _extract_code(raw_response)
    return skeleton_code, raw_response


# ---------------------------------------------------------------------------
# 阶段三：算法 Agent 填充桩函数
# ---------------------------------------------------------------------------

# 各 agent_type 的专家角色描述（与 run_apps_seq_algo.py 保持一致）
_AGENT_TYPE_ROLE: dict[str, str] = {
    "orchestration": """\
你是一名资深 Python 程序员，负责编写程序的 I/O 框架、主函数调用逻辑和整体结构组织。

## 职责
- 正确解析 stdin 输入（注意多行、多组数据、边界空行等情况）
- 定义 main() 或顶层调用逻辑
- 将各子函数的结果正确地用 print() 输出到 stdout

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

_FILL_AGENT_SYSTEM_SUFFIX = """\
## 填充约束（必须遵守）

1. 你将接收含有一个或多个桩函数（以 `raise NotImplementedError` 结尾）的 Python 代码。
2. 只实现标注为 `# [TASK:{task_id}]` 的那一个函数，将 `raise NotImplementedError` 替换为完整的算法实现。
3. 不得修改、删除或移动代码中的其他任何内容（其他函数、main()、import 等）。
4. 若当前代码中仍有其他未实现的桩函数（`raise NotImplementedError`），保留它们原样不动。
5. 输出完整的 Python 代码（用 ```python ... ``` 包裹），不输出任何解释文字。"""

_FILL_AGENT_USER_TEMPLATE = """\
## 题目信息

{problem_text}

---

## 整体解题策略

{global_strategy}

## 全部算法任务一览（仅供上下文参考）

{tasks_summary}

---

## 当前任务：请实现 [{task_id}] {function_name}

- agent_type     : {agent_type}
- 目标（goal）   : {goal}
- 函数签名       : {function_signature}
- 提示（hints）  :
{hints_text}

---

## 当前代码（请在此基础上只填充 [TASK:{task_id}] 标注的函数）

```python
{current_code}
```

请将 [TASK:{task_id}] 函数中的 `raise NotImplementedError` 替换为完整实现，输出完整代码。"""


def call_algo_fill_agent(
    task: dict[str, Any],
    current_code: str,
    problem_text: str,
    global_strategy: str,
    all_tasks: list[dict[str, Any]],
) -> tuple[str, str]:
    """阶段三：算法 Agent 填充指定桩函数。

    Args:
        task         : 当前任务 dict（含 task_id, agent_type, function_name 等）
        current_code : 当前代码（含桩标注）
        problem_text : 格式化后的题目文本
        global_strategy : 整体策略
        all_tasks    : 完整任务列表（提供上下文）

    Returns:
        (updated_code, raw_response)
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    agent_type: str = task.get("agent_type", "orchestration")
    task_id: str = task.get("task_id", "T?")
    function_name: str = task.get("function_name", "unknown")
    function_signature: str = task.get("function_signature", "")
    goal: str = task.get("goal", "")
    hints: list[str] = task.get("hints", [])

    role = _AGENT_TYPE_ROLE.get(agent_type, _AGENT_TYPE_ROLE["orchestration"])
    system_prompt = role + _FILL_AGENT_SYSTEM_SUFFIX.replace("{task_id}", task_id)

    hints_text = "\n".join(f"  - {h}" for h in hints) if hints else "  （无特殊提示）"

    user_content = _FILL_AGENT_USER_TEMPLATE.format(
        problem_text=problem_text,
        global_strategy=global_strategy,
        tasks_summary=_format_tasks_summary(all_tasks),
        task_id=task_id,
        function_name=function_name,
        agent_type=agent_type,
        goal=goal,
        function_signature=function_signature,
        hints_text=hints_text,
        current_code=current_code,
    )

    llm = _build_llm(temperature=0.2)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(messages)
    raw_response: str = response.content
    updated_code = _extract_code(raw_response)
    return updated_code, raw_response


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
    skeleton_code: str,
    algo_steps: list[dict[str, Any]],
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
        "skeleton_code": skeleton_code,
        "algo_steps": algo_steps,
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
    """对单道 APPS 题目执行框架优先三阶段流水线，将结果写入 JSON 文件。

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
            print("  [阶段一] 规划中（生成任务列表）...")
        plan = call_planner(problem_text)

        if "error" in plan:
            if verbose:
                print(f"  [警告] Planner 解析失败：{plan['error'][:100]}")

        tasks: list[dict[str, Any]] = plan.get("tasks", [])
        global_strategy: str = plan.get("global_strategy", "")

        if verbose:
            print(f"  规划完成：{len(tasks)} 个算法任务 | 策略：{global_strategy[:60]}")
            for t in tasks:
                print(
                    f"    [{t.get('task_id')}] {t.get('agent_type')} "
                    f"{t.get('function_name')}() — {t.get('goal', '')[:50]}"
                )

        # --- 阶段二：Framework Agent ---
        if verbose:
            print("  [阶段二] 生成代码骨架...")
        skeleton_code, _ = call_framework_agent(problem_text, plan)

        if verbose:
            stub_count = skeleton_code.count("raise NotImplementedError")
            print(f"  骨架生成完毕：{stub_count} 个待填充桩函数")

        # --- 阶段三：各算法 Agent 依次填充桩函数 ---
        algo_steps: list[dict[str, Any]] = []
        current_code: str = skeleton_code

        for task in tasks:
            task_id = task.get("task_id", "?")
            agent_type = task.get("agent_type", "?")
            function_name = task.get("function_name", "?")

            if verbose:
                print(
                    f"  [阶段三] [{task_id}][{agent_type}] 填充 {function_name}()..."
                )

            updated_code, raw_response = call_algo_fill_agent(
                task=task,
                current_code=current_code,
                problem_text=problem_text,
                global_strategy=global_strategy,
                all_tasks=tasks,
            )

            current_code = updated_code

            algo_steps.append(
                {
                    "task_id": task_id,
                    "agent_type": agent_type,
                    "function_name": function_name,
                    "code_after_fill": current_code,
                    "raw_response": raw_response,
                }
            )

        final_code = current_code
        apps_test_results = run_apps_tests(final_code, io_data, max_cases=max_cases)
        result = build_result_json(
            item, plan, skeleton_code, algo_steps, final_code, apps_test_results
        )

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
            "skeleton_code": None,
            "algo_steps": None,
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
        description="APPS 数据集测试 — 框架优先算法 Agent 填充（规划 → 骨架 → 填充）",
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
    print(f"模型：Qwen / {os.getenv('QWEN_MODEL', 'qwen-plus')}")
    print("模式：框架优先填充（规划 → 骨架 → 算法填充），每题调用 2 + N 次 API")

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
