"""APPS 数据集批量测试脚本（消融实验：错误子树重规划）

消融实验版本：错误定位后执行两阶段重规划：
  阶段一：用 task_replanner 修改根节点规格（针对错误分支对应的 hints/global_strategy/context），
          其他节点和其他分支不变。
  阶段二：以修改后的根节点 hints 为依据，构建聚焦于错误子任务的子问题文本，
          调用 plan_agent 重新生成错误分支的完整子树（sub_ir），
          然后将 sub_ir 嫁接回原 IR，替换旧的错误子树，其余分支代码完整保留。
  最后调用 code_reassembler 重组最终代码。

四点消融对比（重规划策略由窄到宽）：
  单节点重试：      task_replanner 仅修改失败叶节点（final_verifier_agent 内置）
  路径树重规划：    task_replanner 修改路径子树各叶节点，保留树结构
                    （run_apps_path_replan.py）
  错误子树重规划：  task_replanner 修改根节点 → plan_agent 重建错误分支子树，
                    其余分支不变 ← 本脚本
  全局重规划：      plan_agent 重建整棵树（run_apps_global_replan.py）

核心概念定义：
  EB (Error Branch Root)：根节点的直接子节点中在错误路径上的那个节点。
  错误子树：以 EB 为根的子树，包含失败叶节点及其所有兄弟节点。
  其余分支：不在错误子树中的其他分支（代码完全不变）。

流水线：
  plan_agent → algo_agents → assembler_agent
  （重试：error_locator → _find_error_branch_root
          → _replan_root_node（task_replanner 修改根节点）
          → _build_branch_problem（构建子问题文本）
          → plan_agent（生成错误子树 sub_ir）
          → algo_agents（为 sub_ir 叶节点生成代码）
          → _graft_sub_ir（sub_ir 嫁接回原 IR）
          → code_reassembler）

APPS 数据集字段说明：
  - id           : 题目编号（0-based，共 5000 条）
  - question     : 题目描述（自然语言）
  - solutions    : JSON 字符串，包含参考答案列表
  - input_output : JSON 字符串，格式 {"inputs": [...], "outputs": [...]}，已为标准 stdin/stdout
  - difficulty   : "introductory" | "interview" | "competition"
  - url          : 题目来源链接
  - starter_code : 起始代码模板（可能为空字符串）

Usage:
  python tests/run_apps_subtree_replan.py --offset 0 --count 100
  python tests/run_apps_subtree_replan.py --offset 200 --count 50
  python tests/run_apps_subtree_replan.py --offset 0 --count 100 --difficulty introductory
  python tests/run_apps_subtree_replan.py --ids 0,5,10
  python tests/run_apps_subtree_replan.py --offset 0 --count 10 --max-cases 5
  python tests/run_apps_subtree_replan.py --offset 0 --count 10 --output result/apps_ablation_subtree_replan_deepseek
  python tests/run_apps_subtree_replan.py --offset 0 --count 100 --quiet
"""

from __future__ import annotations

import argparse
import copy
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

from auditcoder.algo_agents import dispatch_algo_agents
from auditcoder.assembler_agent import build_assembler_agent_graph
from auditcoder.final_verifier_agent.nodes.code_reassembler import code_reassembler
from auditcoder.final_verifier_agent.nodes.error_locator import error_locator
from auditcoder.final_verifier_agent.nodes.full_code_runner import full_code_runner
from auditcoder.final_verifier_agent.nodes.task_replanner import task_replanner
from auditcoder.plan_agent.graph import build_plan_agent_graph

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------

APPS_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "apps_sampled_400.jsonl"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "result" / "apps_ablation_subtree_replan_deepseek"
)

VALID_DIFFICULTIES = {"introductory", "interview", "competition"}
LLM_PROVIDER_LABEL = "DeepSeek"
LLM_MODEL_ENV = "DEEPSEEK_MODEL"
LLM_DEFAULT_MODEL = "deepseek-chat"


def _get_deepseek_max_tokens(override: int | None = None) -> int | None:
    if override is not None:
        return override
    raw = os.getenv("DEEPSEEK_MAX_TOKENS") or os.getenv("MAX_TOKENS")
    if not raw:
        return None
    try:
        return int(str(raw).strip().strip('"').strip("'"))
    except ValueError:
        return None


def _build_deepseek_llm(
    temperature: float = 0.0,
    json_mode: bool = True,
    max_tokens: int | None = None,
) -> Any:
    from langchain_openai import ChatOpenAI

    model_kwargs: dict[str, Any] = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=temperature,
        max_tokens=_get_deepseek_max_tokens(max_tokens),
        model_kwargs=model_kwargs,
    )


def _patch_auditcoder_llm_to_deepseek() -> None:
    """Route AuditCoder's get_llm imports to DeepSeek for this script only."""
    import auditcoder.common.llm as common_llm

    os.environ["AUDITCODER_LLM_PROVIDER"] = "deepseek"
    common_llm.get_llm = _build_deepseek_llm

    patched_modules = [
        "auditcoder.common",
        "auditcoder.plan_agent.llm",
        "auditcoder.plan_agent.nodes.algorithm_selector",
        "auditcoder.plan_agent.nodes.constraint_analyzer",
        "auditcoder.plan_agent.nodes.deep_thinker",
        "auditcoder.plan_agent.nodes.interface_designer",
        "auditcoder.plan_agent.nodes.ir_reviewer",
        "auditcoder.plan_agent.nodes.problem_parser",
        "auditcoder.plan_agent.nodes.task_decomposer",
        "auditcoder.plan_agent.nodes.test_case_analyzer",
        "auditcoder.assembler_agent.agent",
        "auditcoder.final_verifier_agent.nodes.task_replanner",
        "auditcoder.verifier_agent.nodes.bruteforce_verifier",
        "auditcoder.verifier_agent.nodes.weak_signal_eval",
    ]
    patched_modules.extend(
        f"auditcoder.algo_agents.agents.{name}_agent"
        for name in [
            "binary_search",
            "dp",
            "graph_bfs",
            "graph_dfs",
            "greedy",
            "math",
            "monotonic_queue",
            "segment_tree",
            "sorting",
            "string",
            "two_pointer",
            "union_find",
        ]
    )

    for module_name in patched_modules:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "get_llm"):
            setattr(module, "get_llm", _build_deepseek_llm)

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

            cases.append({
                "index": idx,
                "input": inp,
                "expected": expected,
                "actual": actual,
                "passed": passed,
                "error": error,
            })

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
# 树操作辅助函数（todo: helpers）
# ---------------------------------------------------------------------------


def _get_ancestor_path(
    failing_id: str,
    node_map: dict[str, dict[str, Any]],
) -> list[str]:
    """从失败叶节点出发沿 parent 链向上追溯，返回完整祖先路径（含根）。

    返回列表从失败节点开始，以根节点结束：[failing_id, parent_id, ..., root_id]
    """
    path: list[str] = []
    current_id: str | None = failing_id
    while current_id is not None:
        path.append(current_id)
        node = node_map.get(current_id, {})
        current_id = node.get("parent")
    return path


def _find_error_branch_root(
    failing_id: str,
    node_map: dict[str, dict[str, Any]],
    root_id: str,
) -> str:
    """找到错误路径中紧邻根节点的直接子节点（Error Branch Root = EB）。

    从失败叶节点沿 parent 链上溯，找到 parent == root_id 的节点即为 EB。
    若失败节点本身就是 root 的直接子节点，则 EB = failing_id。
    若路径中找不到（例如 failing_id 无效），则退为 failing_id。
    """
    ancestor_path = _get_ancestor_path(failing_id, node_map)
    # ancestor_path = [failing_id, ..., root_id]
    # 找到路径中 parent == root_id 的节点，即 root_id 在路径中的前一个节点
    for nid in ancestor_path:
        node = node_map.get(nid, {})
        if node.get("parent") == root_id:
            return nid
    # 兜底：若路径中没有直接父节点为 root_id 的节点，返回 failing_id
    return failing_id


def _collect_subtree_node_ids(
    node_id: str,
    node_map: dict[str, dict[str, Any]],
) -> set[str]:
    """递归收集以 node_id 为根的子树中所有节点的 ID（含非叶节点自身）。

    这里以 parent 字段作为树结构的真实来源，而不是 dependencies。dependencies
    表示执行/数据依赖，LLM 偶尔会漏写子节点，不能用它决定要删除的旧子树边界。
    """
    child_map: dict[str, list[str]] = {}
    for nid, node in node_map.items():
        parent = node.get("parent")
        if parent:
            child_map.setdefault(parent, []).append(nid)

    ids: set[str] = set()

    def _walk(current_id: str) -> None:
        if current_id in ids:
            return
        ids.add(current_id)
        for child_id in child_map.get(current_id, []):
            _walk(child_id)

    _walk(node_id)
    return ids


def _source_task_id(source: str | None) -> str | None:
    """从 DataSlot.source 中提取任务 ID，例如 S1_1.value -> S1_1。"""
    if not source or not isinstance(source, str):
        return None
    return source.split(".", 1)[0]


def _rename_source(source: str | None, rename: dict[str, str]) -> str | None:
    """同步重命名 DataSlot.source 中的任务 ID 部分。"""
    source_id = _source_task_id(source)
    if not source_id or source_id not in rename:
        return source
    suffix = source[len(source_id):]
    return rename[source_id] + suffix


def _rename_slot_sources(
    slots: list[dict[str, Any]],
    rename: dict[str, str],
) -> list[dict[str, Any]]:
    """深拷贝 slots，并重写其中 source 引用。"""
    renamed: list[dict[str, Any]] = []
    for slot in slots or []:
        new_slot = copy.deepcopy(slot)
        new_slot["source"] = _rename_source(new_slot.get("source"), rename)
        renamed.append(new_slot)
    return renamed


def _retarget_subtree_sources(
    slots: list[dict[str, Any]],
    old_subtree_ids: set[str],
    branch_root_id: str,
) -> list[dict[str, Any]]:
    """把外部分支中指向旧 EB 内部节点的 source 收敛到 EB 根节点。"""
    rewritten: list[dict[str, Any]] = []
    for slot in slots or []:
        new_slot = copy.deepcopy(slot)
        source = new_slot.get("source")
        source_id = _source_task_id(source)
        if source_id in old_subtree_ids and source_id != branch_root_id:
            suffix = source[len(source_id):] if isinstance(source, str) else ""
            new_slot["source"] = branch_root_id + suffix
        rewritten.append(new_slot)
    return rewritten


def _format_boundary_slots(slots: list[dict[str, Any]]) -> str:
    if not slots:
        return "[]"
    return json.dumps(slots, ensure_ascii=False, indent=2)


def _build_branch_boundary_contract(
    branch_root_id: str,
    node_map: dict[str, dict[str, Any]],
) -> str:
    """描述旧 EB 子树与外部世界的边界，约束新 sub_ir 必须兼容。"""
    branch_root_node = node_map.get(branch_root_id, {})
    old_subtree_ids = _collect_subtree_node_ids(branch_root_id, node_map)

    external_consumers: list[dict[str, Any]] = []
    external_inputs: list[dict[str, Any]] = []

    for node in node_map.values():
        nid = node.get("id")
        for slot in node.get("inputs", []) or []:
            source_id = _source_task_id(slot.get("source"))
            if not source_id:
                continue
            if nid not in old_subtree_ids and source_id in old_subtree_ids:
                external_consumers.append({
                    "consumer_task": nid,
                    "consumer_goal": node.get("goal", ""),
                    "slot": slot,
                })
            if nid in old_subtree_ids and source_id not in old_subtree_ids:
                external_inputs.append({
                    "subtree_task": nid,
                    "subtree_goal": node.get("goal", ""),
                    "slot": slot,
                })

    lines = [
        "【错误子树边界契约（必须保持兼容）】",
        f"- 新子树根节点嫁接后仍会使用旧 EB ID：{branch_root_id}",
        "- 新子树可以重新分解内部任务，但不能破坏旧 EB 对父节点和其他分支暴露的输入/输出语义。",
        "- 若新子树改变内部节点 ID，必须保证根节点 outputs 能覆盖旧 EB outputs，且外部消费者仍能理解这些数据含义。",
        "",
        "旧 EB inputs：",
        _format_boundary_slots(branch_root_node.get("inputs", [])),
        "旧 EB outputs：",
        _format_boundary_slots(branch_root_node.get("outputs", [])),
        f"旧 EB interface：{json.dumps(branch_root_node.get('interface'), ensure_ascii=False, indent=2)}",
        "",
    ]

    if external_inputs:
        lines.append("旧错误子树依赖的外部分支输出：")
        lines.append(json.dumps(external_inputs, ensure_ascii=False, indent=2))
        lines.append("")

    if external_consumers:
        lines.append("旧错误子树输出的外部消费者：")
        lines.append(json.dumps(external_consumers, ensure_ascii=False, indent=2))
        lines.append("")

    return "\n".join(lines)


def _validate_grafted_ir(ir: dict[str, Any]) -> list[str]:
    """返回嫁接后 IR 的结构问题列表；空列表表示基本一致。"""
    problems: list[str] = []
    nodes = ir.get("task_nodes", [])
    node_ids = [n.get("id") for n in nodes]
    valid_ids = {nid for nid in node_ids if nid}

    if len(valid_ids) != len([nid for nid in node_ids if nid]):
        problems.append("task_nodes 中存在重复 ID")

    node_map = {n.get("id"): n for n in nodes if n.get("id")}
    root_id = ir.get("root_id")

    if root_id and root_id not in valid_ids:
        problems.append(f"root_id 不存在：{root_id}")

    for node in nodes:
        nid = node.get("id", "")
        parent = node.get("parent")
        if parent is not None and parent not in valid_ids:
            problems.append(f"{nid}.parent 引用不存在节点：{parent}")
        for dep in node.get("dependencies", []) or []:
            if dep not in valid_ids:
                problems.append(f"{nid}.dependencies 引用不存在节点：{dep}")
        for slot in node.get("inputs", []) or []:
            source_id = _source_task_id(slot.get("source"))
            if source_id and source_id not in valid_ids:
                problems.append(f"{nid}.inputs.source 引用不存在节点：{slot.get('source')}")

    for task_id in ir.get("execution_order", []) or []:
        node = node_map.get(task_id)
        if node is None:
            problems.append(f"execution_order 引用不存在节点：{task_id}")
        elif not node.get("is_leaf"):
            problems.append(f"execution_order 包含非叶节点：{task_id}")

    return problems


def _normalize_leaf_execution_order(ir: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ir whose execution_order contains only leaf nodes.

    Plan-agent outputs, especially regenerated subtrees, may occasionally include
    orchestration nodes in execution_order or omit some leaves. Algo agents only
    generate code for leaves, so keep the LLM-provided leaf order where possible
    and append any missing leaf nodes in task_nodes order.
    """
    nodes = ir.get("task_nodes", []) or []
    node_map = {n.get("id"): n for n in nodes if n.get("id")}
    leaf_ids = [n["id"] for n in nodes if n.get("id") and n.get("is_leaf")]
    leaf_id_set = set(leaf_ids)

    normalized: list[str] = []
    for task_id in ir.get("execution_order", []) or []:
        if task_id in leaf_id_set and task_id not in normalized:
            normalized.append(task_id)

    for task_id in leaf_ids:
        if task_id not in normalized and task_id in node_map:
            normalized.append(task_id)

    return {**ir, "execution_order": normalized}


def _resolve_sub_ir_root_id(sub_ir: dict[str, Any]) -> str:
    """Resolve the actual root node ID of a generated sub_ir.

    Plan-agent outputs occasionally contain a stale or missing root_id. For grafting,
    the actual root is the declared root_id when present in task_nodes; otherwise the
    unique parent-less node; otherwise the first task node as a last resort.
    """
    nodes = sub_ir.get("task_nodes", []) or []
    node_ids = {n.get("id") for n in nodes if n.get("id")}
    declared_root = sub_ir.get("root_id", "")
    if declared_root in node_ids:
        return declared_root

    parentless = [n.get("id") for n in nodes if n.get("id") and n.get("parent") is None]
    if len(parentless) == 1:
        return parentless[0]

    if nodes and nodes[0].get("id"):
        return nodes[0]["id"]

    return declared_root


def _retarget_subtree_dependencies(
    dependencies: list[str],
    old_subtree_ids: set[str],
    branch_root_id: str,
) -> list[str]:
    """Retarget kept external dependencies from deleted EB internals to EB root."""
    retargeted: list[str] = []
    for dep in dependencies or []:
        new_dep = branch_root_id if dep in old_subtree_ids else dep
        if new_dep not in retargeted:
            retargeted.append(new_dep)
    return retargeted


# ---------------------------------------------------------------------------
# 根节点修改与子问题文本构建（todo: root-replan）
# ---------------------------------------------------------------------------


def _replan_root_node(
    ir: dict[str, Any],
    error_location: dict[str, Any],
) -> dict[str, Any]:
    """调用 task_replanner 修改根节点规格（针对错误分支的 hints/context/global_strategy）。

    将 error_location.task_id 覆盖为根节点 ID，使 task_replanner 聚焦于根节点，
    引导其更新根节点中与错误分支相关的全局策略和提示信息。

    Returns:
        更新了根节点的 ir（其余节点不变）。
    """
    root_id = ir.get("root_id", "")
    if not root_id:
        return ir

    root_error_location = {
        **error_location,
        "task_id": root_id,
        "detail": (
            f"根节点全局策略需要针对失败子任务 {error_location.get('task_id', '?')} 进行修正。"
            f"原始错误：{error_location.get('error_type', 'unknown')} — "
            f"{error_location.get('detail', '')[:300]}"
        ),
    }

    temp_state: dict[str, Any] = {
        "task_plan_ir": ir,
        "error_location": root_error_location,
        "retried_tasks": [],
        "updated_task_node": None,
    }

    result = task_replanner(temp_state)
    updated_ir: dict[str, Any] = result.get("task_plan_ir", ir)
    original_root = next(
        (n for n in ir.get("task_nodes", []) if n.get("id") == root_id),
        {},
    )
    structural_keys = {
        "id",
        "type",
        "is_leaf",
        "agent_type",
        "parent",
        "dependencies",
        "inputs",
        "outputs",
        "interface",
    }
    protected_nodes: list[dict[str, Any]] = []
    protected_root_seen = False
    for node in updated_ir.get("task_nodes", []):
        is_root_candidate = node.get("id") == root_id or node is result.get("updated_task_node")
        if not is_root_candidate:
            protected_nodes.append(node)
            continue
        protected_root = dict(node)
        for key in structural_keys:
            if key in original_root:
                protected_root[key] = original_root[key]
        protected_nodes.append(protected_root)
        protected_root_seen = True
    if not protected_root_seen and original_root:
        candidate_root = dict(result.get("updated_task_node") or original_root)
        for key in structural_keys:
            if key in original_root:
                candidate_root[key] = original_root[key]
        protected_nodes.append(candidate_root)
    updated_ir = {**updated_ir, "task_nodes": protected_nodes}
    updated_root = next(
        (n for n in protected_nodes if n.get("id") == root_id),
        result.get("updated_task_node") or {},
    )
    root_hints = updated_root.get("hints") or []
    root_context = updated_root.get("context", "")
    if root_hints or root_context:
        strategy_note = "；".join(str(h) for h in root_hints[:3])
        if root_context:
            strategy_note = f"{strategy_note}；{root_context}" if strategy_note else root_context
        original_strategy = str(updated_ir.get("global_strategy", "")).strip()
        if strategy_note and strategy_note not in original_strategy:
            updated_ir = {
                **updated_ir,
                "global_strategy": (
                    f"{original_strategy}\n错误子树重规划策略修正：{strategy_note}"
                    if original_strategy
                    else f"错误子树重规划策略修正：{strategy_note}"
                ),
            }
    return updated_ir


def _build_branch_problem(
    updated_root_node: dict[str, Any],
    branch_root_node: dict[str, Any],
    branch_root_id: str,
    node_map: dict[str, dict[str, Any]],
    original_problem_text: str,
    error_location: dict[str, Any],
) -> str:
    """构建聚焦于错误分支子任务的问题文本，供 plan_agent 重新生成该分支的子树。

    组合内容：
      - 原始题目描述（完整保留）
      - 根节点修改后的 hints（反映全局策略调整）
      - 错误分支（EB）的目标和上下文
      - 错误信息（帮助 plan_agent 理解需要规避的问题）
    """
    lines: list[str] = [original_problem_text, ""]

    lines += [
        "【子任务重规划背景（请重新规划以下子任务，整体策略已更新）】",
        "",
    ]

    # 根节点更新后的全局策略 / hints
    root_hints: list[str] = updated_root_node.get("hints", [])
    root_context: str = updated_root_node.get("context", "")
    if root_hints:
        lines.append("▶ 全局策略调整（根节点更新后的 hints）：")
        for h in root_hints:
            lines.append(f"  - {h}")
        lines.append("")
    if root_context:
        lines.append(f"▶ 根节点上下文：{root_context}")
        lines.append("")

    # 错误分支（EB）的子任务说明
    eb_goal: str = branch_root_node.get("goal", "")
    eb_context: str = branch_root_node.get("context", "")
    eb_complexity: str = branch_root_node.get("complexity_budget", "")
    if eb_goal:
        lines.append(f"▶ 本次需重新规划的子任务目标：{eb_goal}")
    if eb_context:
        lines.append(f"▶ 子任务上下文：{eb_context}")
    if eb_complexity:
        lines.append(f"▶ 复杂度预算：{eb_complexity}")
    lines.append("")

    lines.append(_build_branch_boundary_contract(branch_root_id, node_map))
    lines.append("")

    # 错误信息
    error_type = error_location.get("error_type", "unknown")
    error_detail = error_location.get("detail", "")
    failing_case = error_location.get("failing_case")
    confidence = error_location.get("confidence", "unknown")
    method = error_location.get("method", "unknown")
    lines.append(f"▶ 上次执行错误（请在重新规划时规避）：")
    lines.append(f"  错误类型：{error_type}")
    lines.append(f"  定位方法：{method}，置信度：{confidence}")
    if error_detail:
        lines.append(f"  错误详情：{error_detail[:400]}")
    if failing_case:
        lines.append(
            f"  失败用例：输入 {str(failing_case.get('input', failing_case))[:600]}，"
            f"期望输出 {str(failing_case.get('expected', ''))[:300]}，"
            f"实际输出 {str(failing_case.get('actual', ''))[:300]}"
        )
        if failing_case.get("stderr"):
            lines.append(f"  stderr：{str(failing_case.get('stderr'))[:300]}")
    lines += [
        "",
        "请根据上述子任务目标、边界契约和错误信息，重新规划该子任务的完整实现方案。",
        "只重新规划错误子树内部；不要把非错误分支或整题任务树重新规划为全局方案。",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# sub_ir 嫁接（todo: graft）
# ---------------------------------------------------------------------------


def _graft_sub_ir(
    original_ir: dict[str, Any],
    branch_root_id: str,
    sub_ir: dict[str, Any],
    original_node_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """将 sub_ir 嫁接回 original_ir，替换旧的 EB（branch_root_id）子树。

    ID 重命名规则（防止与原 IR 其他分支冲突）：
      - sub_ir 根节点 ID → branch_root_id（保留原分支根 ID，根节点 dependencies 无需改变）
      - 其余 sub_ir 节点 nid → f"{branch_root_id}__{nid}"

    操作步骤：
      1. 收集旧 EB 子树所有节点 ID（含非叶），从 original_ir 中删除。
      2. 按重命名规则将 sub_ir 节点加入，更新每个节点的 parent/dependencies。
      3. sub_ir 根节点的 parent 设为 original_ir 中 branch_root_id 的原父节点（即 root_id）。
      4. 重建 execution_order：移除旧 EB 叶节点，在原位置插入 sub_ir 新叶节点。

    Returns:
        合并后的 merged_ir（dict，可直接用于 code_reassembler）。
    """
    sub_ir = _normalize_leaf_execution_order(sub_ir)
    sub_root_id: str = _resolve_sub_ir_root_id(sub_ir)
    prefix = f"{branch_root_id}__"

    # ── 1. 构建 sub_ir ID 重命名映射 ────────────────────────────────────
    rename: dict[str, str] = {}
    for node in sub_ir.get("task_nodes", []):
        nid = node["id"]
        rename[nid] = branch_root_id if nid == sub_root_id else prefix + nid

    # ── 2. 找到 branch_root 在原 IR 中的父节点（即整棵树的 root_id）────────
    original_branch_root = original_node_map.get(branch_root_id, {})
    branch_parent_id: str | None = original_branch_root.get("parent")

    # ── 3. 收集旧 EB 子树节点 ID，从原 IR 删除 ──────────────────────────
    old_subtree_ids = _collect_subtree_node_ids(branch_root_id, original_node_map)
    kept_nodes: list[dict[str, Any]] = []
    for node in original_ir.get("task_nodes", []):
        if node["id"] in old_subtree_ids:
            continue
        kept_node = copy.deepcopy(node)
        kept_node["inputs"] = _retarget_subtree_sources(
            kept_node.get("inputs", []), old_subtree_ids, branch_root_id
        )
        kept_node["dependencies"] = _retarget_subtree_dependencies(
            kept_node.get("dependencies", []), old_subtree_ids, branch_root_id
        )
        kept_nodes.append(kept_node)

    # ── 4. 将 sub_ir 节点重命名后加入 ───────────────────────────────────
    new_nodes: list[dict[str, Any]] = []
    for node in sub_ir.get("task_nodes", []):
        new_node = copy.deepcopy(node)
        new_node["id"] = rename[node["id"]]

        # 设置 parent 字段
        if node["id"] == sub_root_id:
            # sub_ir 根节点接到原 EB 的父节点（original root）
            new_node["parent"] = branch_parent_id
        else:
            parent_orig = node.get("parent")
            new_node["parent"] = rename.get(parent_orig, parent_orig) if parent_orig else None

        # 更新 dependencies
        new_node["dependencies"] = [
            rename.get(d, d) for d in node.get("dependencies", [])
        ]
        new_node["inputs"] = _rename_slot_sources(node.get("inputs", []), rename)
        new_node["outputs"] = _rename_slot_sources(node.get("outputs", []), rename)
        new_nodes.append(new_node)

    merged_task_nodes = kept_nodes + new_nodes

    # ── 5. 重建 execution_order ─────────────────────────────────────────
    # 找到旧 EB 叶节点在 execution_order 中的位置范围
    old_exec_order: list[str] = original_ir.get("execution_order", [])
    old_eb_leaves = old_subtree_ids  # 旧子树所有节点（叶节点会出现在 exec_order 中）

    # 新 sub_ir 的叶节点执行顺序（重命名后）
    new_sub_exec: list[str] = [
        rename.get(nid, prefix + nid) for nid in sub_ir.get("execution_order", [])
    ]

    # 找到第一个旧 EB 叶节点在 execution_order 中的插入位置
    insert_pos: int | None = None
    for i, nid in enumerate(old_exec_order):
        if nid in old_eb_leaves:
            insert_pos = i
            break

    # 移除旧 EB 叶节点，在原位置插入新叶节点
    filtered_exec = [nid for nid in old_exec_order if nid not in old_eb_leaves]
    if insert_pos is not None:
        # 计算移除后的对应索引（insert_pos 之前有多少非 EB 节点）
        adjusted_pos = sum(1 for nid in old_exec_order[:insert_pos] if nid not in old_eb_leaves)
        merged_exec_order = (
            filtered_exec[:adjusted_pos] + new_sub_exec + filtered_exec[adjusted_pos:]
        )
    else:
        # 旧 EB 无叶节点（罕见），直接追加
        merged_exec_order = filtered_exec + new_sub_exec

    merged_ir = {
        **original_ir,
        "root_id": original_ir.get("root_id", ""),
        "task_nodes": merged_task_nodes,
        "execution_order": merged_exec_order,
    }
    validation_errors = _validate_grafted_ir(merged_ir)
    if validation_errors:
        raise ValueError("嫁接后的 TaskPlanIR 结构不一致：" + "；".join(validation_errors[:8]))
    return merged_ir


# ---------------------------------------------------------------------------
# 子树重规划主控函数（todo: subtree-replan-fn）
# ---------------------------------------------------------------------------


def _run_subtree_replan(
    ir: dict[str, Any],
    code_snippets: dict[str, str],
    verification_reports: dict[str, dict[str, Any]],
    error_location: dict[str, Any],
    original_problem_text: str,
) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, Any]], str, str]:
    """对错误分支执行两阶段重规划，返回更新后的 ir、代码、报告、final_code 和 EB ID。

    阶段一：task_replanner 修改根节点（针对错误分支的全局策略/hints）。
    阶段二：以修改后根节点的 hints 为依据，调用 plan_agent 重建错误分支子树，
            然后将新子树嫁接回原 IR，旧 EB 子树的其他分支代码完整保留。

    Args:
        ir:                    原 task_plan_ir。
        code_snippets:         原各叶节点代码片段。
        verification_reports:  原验证报告。
        error_location:        error_locator 输出的错误定位信息。
        original_problem_text: 未附加任何反馈的原始题目描述文本。

    Returns:
        (merged_ir, merged_snippets, merged_reports, final_code, branch_root_id)
    """
    root_id: str = ir.get("root_id", "")
    failing_task_id: str = error_location.get("task_id", "")

    node_map: dict[str, dict[str, Any]] = {
        n["id"]: n for n in ir.get("task_nodes", [])
    }

    # ── 找到 EB（Error Branch Root）───────────────────────────────────────
    branch_root_id = _find_error_branch_root(failing_task_id, node_map, root_id)
    branch_root_node = node_map.get(branch_root_id, {})

    # ── 阶段一：修改根节点 ────────────────────────────────────────────────
    updated_ir = _replan_root_node(ir, error_location)

    # 取出修改后的根节点（用于构建子问题文本）
    updated_node_map: dict[str, dict[str, Any]] = {
        n["id"]: n for n in updated_ir.get("task_nodes", [])
    }
    updated_root_node = updated_node_map.get(root_id, {})

    # ── 阶段二：构建子问题文本 → plan_agent → algo_agents ─────────────────
    branch_problem = _build_branch_problem(
        updated_root_node,
        branch_root_node,
        branch_root_id,
        node_map,
        original_problem_text,
        error_location,
    )

    plan_result = build_plan_agent_graph().invoke(
        {"problem_raw": branch_problem, "revision_count": 0}
    )
    sub_ir: dict[str, Any] = _normalize_leaf_execution_order(
        plan_result.get("task_plan_ir", {})
    )

    sub_snippets, sub_reports = dispatch_algo_agents(sub_ir)

    # ── 嫁接：将 sub_ir 接入原 IR，替换旧 EB 子树 ─────────────────────────
    merged_ir = _graft_sub_ir(updated_ir, branch_root_id, sub_ir, node_map)

    # 合并 code_snippets：保留非 EB 分支的原始代码，加入 sub_ir 的新代码（ID 前缀化）
    old_subtree_ids = _collect_subtree_node_ids(branch_root_id, node_map)
    sub_root_id: str = _resolve_sub_ir_root_id(sub_ir)
    prefix = f"{branch_root_id}__"

    def _rename_sub_id(nid: str) -> str:
        return branch_root_id if nid == sub_root_id else prefix + nid

    merged_snippets: dict[str, str] = {
        k: v for k, v in code_snippets.items() if k not in old_subtree_ids
    }
    for sub_id, snippet in sub_snippets.items():
        merged_snippets[_rename_sub_id(sub_id)] = snippet

    merged_reports: dict[str, dict[str, Any]] = {
        k: v for k, v in verification_reports.items() if k not in old_subtree_ids
    }
    for sub_id, report in sub_reports.items():
        renamed_id = _rename_sub_id(sub_id)
        renamed_report = copy.deepcopy(report)
        if renamed_report.get("task_id") == sub_id:
            renamed_report["task_id"] = renamed_id
        merged_reports[renamed_id] = renamed_report

    # ── 重组最终代码 ──────────────────────────────────────────────────────
    reassemble_state: dict[str, Any] = {
        "task_plan_ir": merged_ir,
        "code_snippets": merged_snippets,
    }
    reassemble_result = code_reassembler(reassemble_state)
    final_code: str = reassemble_result.get("final_code", "")

    return merged_ir, merged_snippets, merged_reports, final_code, branch_root_id


# ---------------------------------------------------------------------------
# 单轮流水线（todo: pipeline-boilerplate）
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
    """运行错误子树重规划消融版 pipeline 并返回各阶段结果。

    流程：
      第一轮: plan_agent → algo_agents → assembler_agent
      检查:   full_code_runner + error_locator
      若失败: _find_error_branch_root → _replan_root_node → _build_branch_problem
              → plan_agent（错误分支子树）→ algo_agents → _graft_sub_ir → code_reassembler
      最终:   再次运行 full_code_runner，构建 final_report
    """
    # ── 第一轮 ────────────────────────────────────────────────────────────
    ir, code_snippets, verification_reports, final_code = _run_single_pass(problem_text)

    fv_state: dict[str, Any] = {
        "final_code": final_code,
        "task_plan_ir": ir,
        "code_snippets": code_snippets,
        "verification_reports": verification_reports,
    }
    runner_result = full_code_runner(fv_state)
    fv_state.update(runner_result)

    execution_result: dict[str, Any] = fv_state.get("execution_result", {})
    subtree_replan_attempted = False
    error_location_out: dict[str, Any] | None = None
    branch_root_id: str = ""

    if not execution_result.get("all_passed", False):
        locator_result = error_locator(fv_state)
        error_location_out = locator_result.get("error_location")

        if error_location_out:
            subtree_replan_attempted = True
            (
                ir,
                code_snippets,
                verification_reports,
                final_code,
                branch_root_id,
            ) = _run_subtree_replan(
                ir,
                code_snippets,
                verification_reports,
                error_location_out,
                original_problem_text=problem_text,
            )

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
            "subtree_replan_attempted": subtree_replan_attempted,
            "error_branch_root": branch_root_id,
            "retry_strategy": "subtree_replan",
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
    """构建最终保存的 JSON 结构（错误子树重规划消融版）。"""
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
        "ablation": "subtree_replan_on_failure",
        "apps_test_results": apps_test_results,
        "final_code": pipeline_result["final_code"],
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
    """对单道 APPS 题目运行错误子树重规划消融版 pipeline 并将结果写入 JSON 文件。

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
            eb = retry_info.get("error_branch_root", "")
            if retry_info.get("subtree_replan_attempted"):
                replan_label = f"（已子树重规划，错误分支根节点：{eb}）"
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
            "ablation": "subtree_replan_on_failure",
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
    _patch_auditcoder_llm_to_deepseek()

    parser = argparse.ArgumentParser(
        description=(
            "APPS 数据集批量测试 - DeepSeek 消融实验"
            "（错误子树重规划：task_replanner 修改根节点 → plan_agent 重建错误分支子树）"
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
    print("[消融实验：错误子树重规划（task_replanner 修改根节点 → plan_agent 重建错误分支子树）]")
    print(f"共选取 {len(selected)} 题{difficulty_label}  |  输出目录：{output_dir}")
    print(f"每题最多运行 {max_cases} 个测试用例")
    print(f"模型：{LLM_PROVIDER_LABEL} / {os.getenv(LLM_MODEL_ENV, LLM_DEFAULT_MODEL)}")

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
