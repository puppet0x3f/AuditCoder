"""APPS 审计指标测试脚本（基于错误子树重规划 pipeline）。

本脚本不修改 auditcoder 包内代码；它复用 tests/run_apps_subtree_replan.py 中的
pipeline 组件，并在每道题上额外记录初始/修复后的快照，计算适合当前代码库
快速测试的审计指标：

  - DCTC-leaf / DCTC-final-func: 代码单元到任务图节点的可追溯覆盖率
  - LC: 初始失败后的定位覆盖
  - LRSR / ORY: 定位后局部修复成功率 / 整体修复收益
  - RRS-node / RRS-LOC: 修复区域大小
  - CCR-LOC / HRSR-LOC: 实际改动比例 / 复核范围减少
  - UVCR-node / RR-unit-report: 冻结已验证代码保留率 / verifier report 级回归率
  - RR-test: APPS 外部测试中修复前已通过用例的回归率

Usage:
  python tests/run_apps_audit_metrics.py --offset 0 --count 5
  python tests/run_apps_audit_metrics.py --ids 2,13 --max-cases 10
  python tests/run_apps_audit_metrics.py --offset 0 --count 10 --output result/apps_audit_metrics
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import difflib
import hashlib
import inspect
import json
import os
import statistics
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent
APPS_DATA_PATH = REPO_ROOT / "data" / "apps_sampled_400.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "result" / "apps_audit_metrics"
VALID_DIFFICULTIES = {"introductory", "interview", "competition"}

_PROBLEM_STATEMENT_SUPPLEMENTS: dict[int, str] = {
    4824: """
AUDIT DATASET ASSET RECOVERY — authoritative piece diagram transcription:
The imported statement omitted the figure that defines piece numbers.  Using
(x, y) coordinates with y=0 on the bottom, the seven unrotated pieces are:
1: [(0,0), (0,1), (0,2), (0,3)]
2: [(0,0), (1,0), (0,1), (1,1)]
3: [(0,0), (1,0), (1,1), (2,1)]
4: [(1,0), (2,0), (0,1), (1,1)]
5: [(0,0), (1,0), (2,0), (1,1)]
6: [(0,0), (1,0), (2,0), (2,1)]
7: [(0,0), (1,0), (2,0), (0,1)]
This transcription is part of the problem specification.  Generate rotations
from these exact shapes; do not substitute a conventional Tetris numbering.
""".strip(),
    4868: """
AUDIT DATASET I/O CLARIFICATION — authoritative sample contract:
The input is exactly two piece-list lines beginning with "White:" and "Black:".
The output is the 17-line ASCII chessboard.  This direction is demonstrated by
the supplied sample input and sample output.  Do not implement the inverse
board-to-piece-list transformation, even though the statement references a
different problem named "Help Me With The Game".
""".strip(),
}


def _supplement_problem_text(problem_id: int, problem_text: str) -> str:
    """Restore authoritative statement information omitted by APPS ingestion."""
    supplement = _PROBLEM_STATEMENT_SUPPLEMENTS.get(problem_id)
    if not supplement:
        return problem_text
    return f"{problem_text.rstrip()}\n\n{supplement}\n"


class TokenUsageTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.phase = "unscoped"

    @contextlib.contextmanager
    def scoped(self, phase: str) -> Any:
        previous = self.phase
        self.phase = phase
        try:
            yield
        finally:
            self.phase = previous

    def record(self, module: str, response: Any) -> None:
        usage = _extract_token_usage(response)
        call = {
            "phase": self.phase,
            "module": module,
            **usage,
        }
        self.calls.append(call)

    def snapshot(self) -> dict[str, Any]:
        by_module: dict[str, dict[str, Any]] = {}
        totals = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls_missing_usage": 0,
        }
        by_phase: dict[str, dict[str, Any]] = {}
        by_phase_module: dict[str, dict[str, dict[str, Any]]] = {}

        for call in self.calls:
            phase = call["phase"]
            module = call["module"]
            bucket = by_module.setdefault(
                module,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "calls_missing_usage": 0,
                },
            )
            phase_bucket = by_phase.setdefault(
                phase,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "calls_missing_usage": 0,
                },
            )
            phase_module_bucket = by_phase_module.setdefault(phase, {}).setdefault(
                module,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "calls_missing_usage": 0,
                },
            )
            for target in (bucket, phase_bucket, phase_module_bucket, totals):
                target["calls"] += 1

            if call.get("usage_available"):
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = int(call.get(key) or 0)
                    for target in (bucket, phase_bucket, phase_module_bucket, totals):
                        target[key] += value
            else:
                for target in (bucket, phase_bucket, phase_module_bucket, totals):
                    target["calls_missing_usage"] += 1

        return {
            "total": totals,
            "by_phase": by_phase,
            "by_module": by_module,
            "by_phase_module": by_phase_module,
            "calls": self.calls,
        }


TOKEN_TRACKER = TokenUsageTracker()


class InstrumentedLLM:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        module = _infer_llm_call_module()
        response = self._inner.invoke(*args, **kwargs)
        TOKEN_TRACKER.record(module, response)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _extract_token_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")
        if any(v is not None for v in (input_tokens, output_tokens, total_tokens)):
            prompt = int(input_tokens or 0)
            completion = int(output_tokens or 0)
            total = int(total_tokens or (prompt + completion))
            return {
                "usage_available": True,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
                "usage_source": "usage_metadata",
            }

    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
        if isinstance(token_usage, dict):
            prompt = token_usage.get("prompt_tokens", token_usage.get("input_tokens"))
            completion = token_usage.get("completion_tokens", token_usage.get("output_tokens"))
            total = token_usage.get("total_tokens")
            if any(v is not None for v in (prompt, completion, total)):
                prompt_i = int(prompt or 0)
                completion_i = int(completion or 0)
                total_i = int(total or (prompt_i + completion_i))
                return {
                    "usage_available": True,
                    "prompt_tokens": prompt_i,
                    "completion_tokens": completion_i,
                    "total_tokens": total_i,
                    "usage_source": "response_metadata.token_usage",
                }

    return {
        "usage_available": False,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "usage_source": None,
    }


def _infer_llm_call_module() -> str:
    for frame in inspect.stack()[2:]:
        filename = frame.filename.replace("\\", "/")
        if "/auditcoder/" not in filename:
            continue
        rel = filename.split("/auditcoder/", 1)[1]
        stem = rel.removesuffix(".py").replace("/", ".")
        if stem == "common.llm":
            continue
        return f"auditcoder.{stem}.{frame.function}"
    return "unknown"


def _load_subtree_replan() -> Any:
    import run_apps_subtree_replan as subtree_replan

    _patch_auditcoder_llm_to_deepseek()
    return subtree_replan


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

    llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=temperature,
        max_tokens=_get_deepseek_max_tokens(max_tokens),
        model_kwargs=model_kwargs,
    )
    return InstrumentedLLM(llm)


def _patch_auditcoder_llm_to_deepseek() -> None:
    """Route AuditCoder's get_llm imports to DeepSeek for this script only."""
    import auditcoder.common.llm as common_llm

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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _loc(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip())


def _line_change_count(before: str, after: str) -> int:
    """Return review-relevant line churn using a simple line diff.

    For replace blocks, use max(old_lines, new_lines) so a one-line edit counts
    as one changed line instead of one deletion plus one insertion.  Deletions
    still count because a reviewer must inspect removed code.
    """
    before_lines = [ln for ln in before.splitlines() if ln.strip()]
    after_lines = [ln for ln in after.splitlines() if ln.strip()]
    matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines)
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed += max(i2 - i1, j2 - j1)
    return changed


def _changed_final_line_count(before: str, after: str) -> int:
    """Return non-empty LOC in the final program that came from changed hunks."""
    before_lines = [ln for ln in before.splitlines() if ln.strip()]
    after_lines = [ln for ln in after.splitlines() if ln.strip()]
    matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines)
    return sum(
        j2 - j1
        for tag, _i1, _i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    )


def _function_defs(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
    return names


def _function_ast_fingerprints(code: str) -> dict[str, list[str]]:
    """Return normalized AST fingerprints for every function definition."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    fingerprints: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fingerprint = _sha256_text(
                ast.dump(node, annotate_fields=True, include_attributes=False)
            )
            fingerprints.setdefault(node.name, []).append(fingerprint)
    return fingerprints


def _node_map(ir: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {n["id"]: n for n in ir.get("task_nodes", []) if n.get("id")}


def _leaf_node_ids(ir: dict[str, Any]) -> list[str]:
    return [
        n["id"]
        for n in ir.get("task_nodes", [])
        if n.get("id") and n.get("is_leaf")
    ]


def _report_is_pass(report: dict[str, Any] | None) -> bool:
    if not report:
        return False
    return report.get("verdict") == "accept"


def _report_complete(report: dict[str, Any] | None) -> bool:
    """Return whether a non-empty, machine-readable validation record exists."""
    return bool(report) and bool(report.get("verdict"))


def _contract_complete(node: dict[str, Any]) -> bool:
    iface = node.get("interface") or {}
    return bool(iface.get("function_name") and "params" in iface and iface.get("return_type"))


def _provenance_complete(node: dict[str, Any]) -> bool:
    # Existing IR records provenance through DataSlot.source. A null source means
    # problem input, but the key should still be present for machine checking.
    for slot in node.get("inputs", []) or []:
        if not slot.get("name") or not slot.get("type") or "source" not in slot:
            return False
    return True


def compute_dctc(ir: dict[str, Any], code_snippets: dict[str, str], reports: dict[str, dict[str, Any]], final_code: str) -> dict[str, Any]:
    """Compute paper-aligned DCTC-leaf plus diagnostic component coverage.

    The paper's formal DCTC-leaf definition is a *record-completeness* test:
    code, contract, provenance, and validation evidence must all be present for
    each leaf implementation node.  It is not ground-truth evidence that the
    recorded owner is causally correct, so unique-owner and agent coverage are
    reported separately instead of being silently folded into DCTC-leaf.
    """
    nodes = _node_map(ir)
    leaf_ids = _leaf_node_ids(ir)
    covered_leaf_ids: list[str] = []
    component_covered: dict[str, list[str]] = {
        "code": [],
        "contract": [],
        "provenance": [],
        "evidence": [],
        "responsible_agent": [],
    }
    uncovered_reasons: dict[str, list[str]] = {}

    for task_id in leaf_ids:
        node = nodes.get(task_id, {})
        checks = {
            "code": bool(str(code_snippets.get(task_id, "")).strip()),
            "contract": _contract_complete(node),
            "provenance": _provenance_complete(node),
            "evidence": _report_complete(reports.get(task_id)),
            "responsible_agent": bool(node.get("agent_type")),
        }
        for name, passed in checks.items():
            if passed:
                component_covered[name].append(task_id)

        paper_checks = ("code", "contract", "provenance", "evidence")
        if all(checks[name] for name in paper_checks):
            covered_leaf_ids.append(task_id)
        else:
            uncovered_reasons[task_id] = [
                name for name in paper_checks if not checks[name]
            ]

    dctc_leaf = len(covered_leaf_ids) / len(leaf_ids) if leaf_ids else None

    owner_by_function: dict[str, list[str]] = {}
    for task_id in leaf_ids:
        fn = (nodes.get(task_id, {}).get("interface") or {}).get("function_name")
        if fn:
            owner_by_function.setdefault(fn, []).append(task_id)

    final_functions = _function_defs(final_code)
    final_function_fingerprints = _function_ast_fingerprints(final_code)
    assembly_preserved_leaf_ids: list[str] = []
    assembly_preservation_failures: dict[str, str] = {}
    for task_id in leaf_ids:
        node = nodes.get(task_id, {})
        fn = (node.get("interface") or {}).get("function_name")
        snippet = code_snippets.get(task_id, "")
        snippet_fingerprints = _function_ast_fingerprints(snippet).get(fn, [])
        assembled_fingerprints = final_function_fingerprints.get(fn, [])
        if (
            fn
            and len(snippet_fingerprints) == 1
            and len(assembled_fingerprints) == 1
            and snippet_fingerprints[0] == assembled_fingerprints[0]
        ):
            assembly_preserved_leaf_ids.append(task_id)
        elif not fn:
            assembly_preservation_failures[task_id] = (
                "missing_interface_function_name"
            )
        elif len(snippet_fingerprints) != 1:
            assembly_preservation_failures[task_id] = (
                "snippet_function_missing_or_ambiguous"
            )
        elif len(assembled_fingerprints) != 1:
            assembly_preservation_failures[task_id] = (
                "final_function_missing_or_ambiguous"
            )
        else:
            assembly_preservation_failures[task_id] = "function_ast_changed"

    duplicate_function_owners = {
        fn: owners for fn, owners in owner_by_function.items() if len(owners) != 1
    }
    uniquely_owned_leaf_ids = {
        owners[0] for owners in owner_by_function.values() if len(owners) == 1
    }
    unique_mapping_rate = (
        len(uniquely_owned_leaf_ids) / len(leaf_ids) if leaf_ids else None
    )
    covered_leaf_set = set(covered_leaf_ids)
    traced_final_functions = []
    for fn in final_functions:
        owners = owner_by_function.get(fn, [])
        if len(owners) == 1 and owners[0] in covered_leaf_set:
            traced_final_functions.append(fn)
    dctc_final_func = (
        len(traced_final_functions) / len(final_functions)
        if final_functions
        else None
    )
    component_rates = {
        name: (len(ids) / len(leaf_ids) if leaf_ids else None)
        for name, ids in component_covered.items()
    }
    assembly_preservation_rate = (
        len(assembly_preserved_leaf_ids) / len(leaf_ids)
        if leaf_ids
        else None
    )

    return {
        "dctc_leaf": dctc_leaf,
        "dctc_final_func": dctc_final_func,
        "metric_semantics": "responsibility_record_completeness",
        "leaf_code_units": len(leaf_ids),
        "covered_leaf_code_units": len(covered_leaf_ids),
        "covered_leaf_ids": covered_leaf_ids,
        "uncovered_reasons": uncovered_reasons,
        "component_rates": component_rates,
        "unique_mapping_rate": unique_mapping_rate,
        "duplicate_function_owners": duplicate_function_owners,
        "assembly_preservation_rate": assembly_preservation_rate,
        "assembly_preserved_leaf_ids": assembly_preserved_leaf_ids,
        "assembly_preservation_failures": assembly_preservation_failures,
        "final_functions": final_functions,
        "traced_final_functions": traced_final_functions,
    }


def _test_regression_rate(
    initial_apps_tests: dict[str, Any],
    final_apps_tests: dict[str, Any],
) -> float | None:
    before_cases = initial_apps_tests.get("cases") or []
    after_cases = final_apps_tests.get("cases") or []
    before_passed_indices = {
        c.get("index")
        for c in before_cases
        if c.get("passed") and c.get("index") is not None
    }
    if not before_passed_indices:
        return None
    after_by_index = {
        c.get("index"): c for c in after_cases if c.get("index") is not None
    }
    regressed = 0
    for idx in before_passed_indices:
        if not after_by_index.get(idx, {}).get("passed", False):
            regressed += 1
    return regressed / len(before_passed_indices)


def compute_localization_ground_truth_metrics(
    *,
    error_location: dict[str, Any] | None,
    repair_region_nodes: set[str],
    ground_truth_nodes: set[str],
    expect_abstention: bool = False,
) -> dict[str, Any]:
    """Evaluate localization against controlled-fault ground truth.

    Natural benchmark failures do not provide responsibility labels.  This
    helper is intended for deterministic fault-injection tests where the
    injected node or branch is known in advance.
    """
    located_node = (error_location or {}).get("task_id")
    localized = bool(located_node)

    if expect_abstention:
        return {
            "localization_at_1": None,
            "abstention_correct": not localized,
            "repair_boundary_precision": None,
            "repair_boundary_recall": None,
            "repair_boundary_iou": None,
        }

    localization_at_1 = int(bool(located_node in ground_truth_nodes))
    intersection = repair_region_nodes & ground_truth_nodes
    union = repair_region_nodes | ground_truth_nodes
    precision = (
        len(intersection) / len(repair_region_nodes)
        if repair_region_nodes
        else 0.0
    )
    recall = (
        len(intersection) / len(ground_truth_nodes)
        if ground_truth_nodes
        else None
    )
    iou = len(intersection) / len(union) if union else None
    return {
        "localization_at_1": localization_at_1,
        "abstention_correct": None,
        "repair_boundary_precision": precision,
        "repair_boundary_recall": recall,
        "repair_boundary_iou": iou,
    }


def compute_repair_metrics(
    *,
    initial_ir: dict[str, Any],
    final_ir: dict[str, Any],
    initial_snippets: dict[str, str],
    final_snippets: dict[str, str],
    initial_reports: dict[str, dict[str, Any]],
    final_reports: dict[str, dict[str, Any]],
    initial_code: str,
    final_code: str,
    initial_execution: dict[str, Any],
    final_execution: dict[str, Any],
    error_location: dict[str, Any] | None,
    repair_region_nodes: set[str],
    initial_apps_tests: dict[str, Any],
    final_apps_tests: dict[str, Any],
    reverified_node_ids: set[str] | None = None,
) -> dict[str, Any]:
    initial_pass = bool(initial_execution.get("all_passed", False))
    final_pass = bool(final_execution.get("all_passed", False))
    initial_apps_pass = bool(initial_apps_tests.get("all_passed", False))
    final_apps_pass = bool(final_apps_tests.get("all_passed", False))
    initial_failed = not initial_pass or not initial_apps_pass
    initial_nodes = _node_map(initial_ir)
    initial_node_ids = set(initial_nodes)
    initial_leaf_ids = set(_leaf_node_ids(initial_ir))
    implementation_node_ids = initial_leaf_ids | (
        set(initial_snippets) & initial_node_ids
    )

    localized = None
    if initial_failed:
        task_id = (error_location or {}).get("task_id")
        localized = bool(task_id and task_id in initial_nodes and (error_location or {}).get("method"))

    repaired = initial_failed and bool(repair_region_nodes)
    repair_graph_region = repair_region_nodes & initial_node_ids
    repair_implementation_region = repair_region_nodes & implementation_node_ids
    repair_leaf_region = repair_region_nodes & initial_leaf_ids
    total_graph_node_count = len(initial_node_ids)
    total_implementation_node_count = len(implementation_node_ids)
    total_leaf_count = len(initial_leaf_ids)

    region_loc = sum(
        _loc(initial_snippets.get(tid, ""))
        for tid in repair_implementation_region
    )
    total_snippet_loc = sum(
        _loc(initial_snippets.get(tid, ""))
        for tid in implementation_node_ids
    )

    changed_loc = (
        _changed_final_line_count(initial_code, final_code) if repaired else 0
    )
    diff_churn_loc = _line_change_count(initial_code, final_code) if repaired else 0
    initial_final_code_loc = _loc(initial_code)
    final_code_loc = _loc(final_code)

    frozen_implementation_ids = implementation_node_ids - repair_region_nodes
    frozen_verified_ids = {
        tid
        for tid in frozen_implementation_ids
        if _report_is_pass(initial_reports.get(tid))
    }

    unchanged_verified = 0
    for task_id in frozen_verified_ids:
        before_hash = _sha256_text(initial_snippets.get(task_id, ""))
        after_snippet = final_snippets.get(task_id)
        if after_snippet is not None and before_hash == _sha256_text(after_snippet):
            unchanged_verified += 1

    uvcr_node = (
        unchanged_verified / len(frozen_verified_ids)
        if frozen_verified_ids
        else None
    )

    # A copied pre-repair report is not post-repair evidence.  Unit regression
    # is therefore computed only for nodes explicitly re-verified after repair.
    explicitly_reverified = set(reverified_node_ids or ())
    reverified_frozen_ids = {
        tid
        for tid in frozen_verified_ids
        if tid in explicitly_reverified and _report_complete(final_reports.get(tid))
    }
    unit_regressed = sum(
        1
        for task_id in reverified_frozen_ids
        if final_reports[task_id].get("verdict") == "reject"
    )
    rr_unit = (
        unit_regressed / len(reverified_frozen_ids)
        if reverified_frozen_ids
        else None
    )
    rr_unit_evaluation_coverage = (
        len(reverified_frozen_ids) / len(frozen_verified_ids)
        if frozen_verified_ids
        else None
    )

    ccr_loc = (
        changed_loc / final_code_loc
        if repaired and final_code_loc
        else None
    )
    # HRSR uses diff churn because deleted lines still require review, whereas
    # CCR_LOC counts only changed lines present in the final program.
    hrsr_loc = (
        1 - (diff_churn_loc / final_code_loc)
        if repaired and final_code_loc
        else None
    )

    return {
        "initial_pipeline_pass": initial_pass,
        "final_pipeline_pass": final_pass,
        "initial_apps_pass": initial_apps_pass,
        "final_apps_pass": final_apps_pass,
        "localization_coverage": int(localized) if localized is not None else None,
        "localized_repair_success": (
            int(bool(localized) and final_pass) if localized is not None and localized else None
        ),
        "localized_repair_success_apps": (
            int(bool(localized) and final_apps_pass) if localized is not None and localized else None
        ),
        "external_repair_recovery": (
            int(final_apps_pass)
            if bool(localized) and not initial_apps_pass
            else None
        ),
        "external_pass_preservation": (
            int(final_apps_pass)
            if bool(localized) and initial_apps_pass
            else None
        ),
        "overall_repair_yield": int(final_pass) if not initial_pass else None,
        "overall_repair_yield_apps": int(final_apps_pass) if not initial_pass else None,
        "rrs_node": (
            len(repair_graph_region) / total_graph_node_count
            if repaired and total_graph_node_count
            else None
        ),
        "rrs_implementation_node": (
            len(repair_implementation_region) / total_implementation_node_count
            if repaired and total_implementation_node_count
            else None
        ),
        "rrs_loc": (
            region_loc / total_snippet_loc
            if repaired and total_snippet_loc
            else None
        ),
        "ccr_loc": ccr_loc,
        "hrsr_loc": hrsr_loc,
        "uvcr_node": uvcr_node,
        "rr_unit": rr_unit,
        "rr_unit_report_based": rr_unit,
        "rr_unit_evaluation_coverage": rr_unit_evaluation_coverage,
        "rr_test": _test_regression_rate(initial_apps_tests, final_apps_tests),
        "repair_region_node_count": len(repair_region_nodes) if repaired else 0,
        "repair_region_graph_node_count": len(repair_graph_region) if repaired else 0,
        "repair_region_implementation_node_count": (
            len(repair_implementation_region) if repaired else 0
        ),
        "repair_region_leaf_count": len(repair_leaf_region) if repaired else 0,
        "total_graph_node_count": total_graph_node_count,
        "total_implementation_node_count": total_implementation_node_count,
        "total_leaf_count": total_leaf_count,
        "repair_region_loc": region_loc,
        "total_snippet_loc": total_snippet_loc,
        "changed_loc": changed_loc,
        "diff_churn_loc": diff_churn_loc,
        "initial_final_code_loc": initial_final_code_loc,
        "final_code_loc": final_code_loc,
        "review_loc_bounded": diff_churn_loc if repaired else None,
        "review_loc_global": final_code_loc if repaired else None,
        "hrsr_is_ccr_complement": (
            bool(repaired) and diff_churn_loc == changed_loc
        ),
        "frozen_verified_node_count": len(frozen_verified_ids),
        "reverified_frozen_node_count": len(reverified_frozen_ids),
    }


def build_repair_record(
    *,
    initial_ir: dict[str, Any],
    initial_snippets: dict[str, str],
    final_snippets: dict[str, str],
    initial_execution: dict[str, Any],
    final_execution: dict[str, Any],
    initial_apps_tests: dict[str, Any],
    final_apps_tests: dict[str, Any],
    localization_record: dict[str, Any] | None,
    error_branch_root: str,
    repair_region_nodes: set[str],
) -> dict[str, Any] | None:
    """Build one compact, machine-readable repair-history event."""
    if not repair_region_nodes:
        return None

    initial_node_ids = set(_node_map(initial_ir))
    all_code_unit_ids = set(initial_snippets) | set(final_snippets)
    changed_units: list[dict[str, Any]] = []
    reused_units: list[dict[str, Any]] = []
    for task_id in sorted(all_code_unit_ids):
        before = initial_snippets.get(task_id)
        after = final_snippets.get(task_id)
        record = {
            "node_id": task_id,
            "before_sha256": _sha256_text(before) if before is not None else None,
            "after_sha256": _sha256_text(after) if after is not None else None,
        }
        if before is not None and after is not None and before == after:
            reused_units.append(record)
        else:
            changed_units.append(record)

    return {
        "repair_attempt": 1,
        "localization": localization_record,
        "repair_region": {
            "region_type": (
                "global"
                if error_branch_root == initial_ir.get("root_id")
                else ("branch" if error_branch_root else "node")
            ),
            "error_branch_root": error_branch_root or None,
            "nodes": sorted(repair_region_nodes),
        },
        "frozen_nodes": sorted(initial_node_ids - repair_region_nodes),
        "changed_code_units": changed_units,
        "reused_code_units": reused_units,
        "initial_validation_passed": bool(
            initial_execution.get("all_passed", False)
        ),
        "post_repair_validation_passed": bool(
            final_execution.get("all_passed", False)
        ),
        "initial_apps_passed": bool(
            initial_apps_tests.get("all_passed", False)
        ),
        "post_repair_apps_passed": bool(
            final_apps_tests.get("all_passed", False)
        ),
    }


def _canonical_sample_cases(
    io_data: dict[str, Any],
    max_cases: int,
) -> list[dict[str, Any]]:
    """Preserve dataset test bundles verbatim instead of round-tripping via an LLM."""
    inputs = io_data.get("inputs", [])
    outputs = io_data.get("outputs", [])
    return [
        {
            "input": str(test_input),
            "expected_output": str(expected_output),
            "test_source": "dataset.input_output",
        }
        for test_input, expected_output in list(zip(inputs, outputs))[:max_cases]
    ]


def _install_canonical_validation(
    task_plan_ir: dict[str, Any],
    canonical_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    updated = copy.deepcopy(task_plan_ir)
    validation = dict(updated.get("validation") or {})
    validation["llm_sample_test_cases"] = validation.get("sample_test_cases", [])
    validation["sample_test_cases"] = copy.deepcopy(canonical_cases)
    validation["test_source"] = "dataset.input_output"
    updated["validation"] = validation
    return updated


def _apps_failure_as_execution(
    apps_test_results: dict[str, Any],
) -> dict[str, Any]:
    """Convert external APPS evidence to the locator's execution-result schema."""
    converted_results: list[dict[str, Any]] = []
    timed_out = False
    runtime_error = False
    for case in apps_test_results.get("cases", []):
        error = str(case.get("error") or "")
        case_timed_out = "timeout" in error.lower()
        case_runtime_error = bool(error) and not case_timed_out
        timed_out = timed_out or case_timed_out
        runtime_error = runtime_error or case_runtime_error
        converted_results.append(
            {
                "case_index": case.get("index"),
                "input": case.get("input", ""),
                "expected": case.get("expected", ""),
                "actual": case.get("actual", ""),
                "passed": bool(case.get("passed", False)),
                "stderr": error,
                "timed_out": case_timed_out,
                "exit_code": 1 if case_runtime_error else 0,
                "judge_mode": case.get("judge_mode"),
                "judge_detail": case.get("judge_detail"),
            }
        )
    error_type = (
        "timeout"
        if timed_out
        else ("runtime_error" if runtime_error else "wrong_answer")
    )
    return {
        "ran_tests": True,
        "all_passed": bool(apps_test_results.get("all_passed", False)),
        "timed_out": timed_out,
        "exit_code": (
            0 if apps_test_results.get("all_passed", False) else -1
        ),
        "stdout": "",
        "stderr": next(
            (
                str(result.get("stderr") or "")
                for result in converted_results
                if result.get("stderr")
            ),
            "",
        ),
        "time_ms": 0.0,
        "results": converted_results,
        "error_type": error_type,
        "test_source": "dataset.input_output",
    }


def _select_failure_evidence(
    internal_execution: dict[str, Any],
    external_apps_tests: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Select trustworthy repair evidence from internal and external judges."""
    internal_passed = bool(internal_execution.get("all_passed", False))
    external_passed = bool(external_apps_tests.get("all_passed", False))
    if internal_passed and external_passed:
        return None, None
    if not internal_passed and external_passed:
        return None, {
            "status": "abstained",
            "method": "evidence_disagreement",
            "error_type": internal_execution.get("error_type", "unknown"),
            "reason": "internal_failed_but_canonical_external_passed",
        }
    if internal_passed and not external_passed:
        return _apps_failure_as_execution(external_apps_tests), None
    return internal_execution, None


def _candidate_validation_rank(
    execution_result: dict[str, Any],
    apps_tests: dict[str, Any],
) -> tuple[int, int, int, int, int]:
    """Rank a generated candidate using deterministic canonical evidence."""
    internal_results = execution_result.get("results", [])
    external_cases = apps_tests.get("cases", [])
    internal_passes = sum(
        bool(result.get("passed", False)) for result in internal_results
    )
    external_passes = sum(
        bool(case.get("passed", False)) for case in external_cases
    )
    internal_clean_exits = sum(
        not result.get("timed_out", False)
        and int(result.get("exit_code", 0) or 0) == 0
        and not result.get("stderr")
        for result in internal_results
    )
    external_clean_exits = sum(
        not case.get("error") for case in external_cases
    )
    fully_passed = int(
        bool(execution_result.get("all_passed", False))
        and bool(apps_tests.get("all_passed", False))
    )
    return (
        fully_passed,
        external_passes,
        internal_passes,
        external_clean_exits,
        internal_clean_exits,
    )


def _repair_acceptance_record(
    *,
    initial_execution: dict[str, Any],
    initial_apps_tests: dict[str, Any],
    candidate_execution: dict[str, Any],
    candidate_apps_tests: dict[str, Any],
    candidate_code: str,
) -> dict[str, Any]:
    """Accept a repair only when canonical evidence strictly improves."""
    initial_rank = _candidate_validation_rank(
        initial_execution,
        initial_apps_tests,
    )
    candidate_rank = _candidate_validation_rank(
        candidate_execution,
        candidate_apps_tests,
    )
    accepted = candidate_rank > initial_rank
    return {
        "attempted": True,
        "accepted": accepted,
        "decision": "accepted" if accepted else "rolled_back",
        "reason": (
            "candidate_strictly_improved_canonical_validation"
            if accepted
            else "candidate_did_not_strictly_improve_canonical_validation"
        ),
        "rank_fields": [
            "fully_passed",
            "external_cases_passed",
            "internal_cases_passed",
            "external_clean_exits",
            "internal_clean_exits",
        ],
        "initial_rank": list(initial_rank),
        "candidate_rank": list(candidate_rank),
        "candidate_code_sha256": _sha256_text(candidate_code),
    }


def _first_failing_case(execution_result: dict[str, Any]) -> dict[str, Any]:
    return next(
        (
            result
            for result in execution_result.get("results", [])
            if not result.get("passed", True)
        ),
        {},
    )


def _global_failure_context(
    execution_result: dict[str, Any],
    localization_record: dict[str, Any] | None,
) -> str:
    failing_case = _first_failing_case(execution_result)
    return (
        "The assembled solution failed canonical dataset tests, but responsibility "
        "could not be assigned to one leaf. Treat this as a possible global "
        "algorithm or decomposition error and reconsider the strategy from "
        "deep_thinker onward.\n"
        f"error_type={execution_result.get('error_type', 'unknown')}\n"
        f"localization={json.dumps(localization_record or {}, ensure_ascii=False)}\n"
        f"failing_case={json.dumps(failing_case, ensure_ascii=False)}"
    )


def _run_global_replan(
    prior_plan_state: dict[str, Any],
    failure_context: str,
) -> tuple[
    dict[str, Any],
    dict[str, str],
    dict[str, dict[str, Any]],
    str,
]:
    """Run the existing deep-thinker partial graph as an abstention fallback."""
    from auditcoder.algo_agents import dispatch_algo_agents
    from auditcoder.assembler_agent import build_assembler_agent_graph
    from auditcoder.plan_agent.graph import build_partial_replan_graph

    plan_result = build_partial_replan_graph().invoke(
        {
            "problem_structured": prior_plan_state.get("problem_structured", {}),
            "failure_context": failure_context,
            "revision_count": 0,
        }
    )
    task_plan_ir = plan_result.get("task_plan_ir", {})
    snippets, reports = dispatch_algo_agents(task_plan_ir)
    assembled = build_assembler_agent_graph().invoke(
        {
            "task_plan_ir": task_plan_ir,
            "code_snippets": snippets,
        }
    )
    return task_plan_ir, snippets, reports, assembled.get("final_code", "")


def run_audit_pipeline(
    problem_text: str,
    io_data: dict[str, Any],
    max_cases: int,
    problem_id: int,
) -> dict[str, Any]:
    """Run subtree replan pipeline with audit snapshots."""
    from auditcoder.final_verifier_agent.nodes.error_locator import error_locator
    from auditcoder.final_verifier_agent.nodes.full_code_runner import full_code_runner

    try:
        subtree_replan = _load_subtree_replan()
    except ModuleNotFoundError as exc:
        print(
            f"[错误] 缺少运行 AuditCoder pipeline 所需依赖：{exc.name}。"
            "请先安装 requirements.txt；若只想检查指标公式，运行 --self-test。",
            file=sys.stderr,
        )
        sys.exit(1)
    supplement_applied = problem_id in _PROBLEM_STATEMENT_SUPPLEMENTS
    problem_text = _supplement_problem_text(problem_id, problem_text)
    TOKEN_TRACKER.reset()
    with TOKEN_TRACKER.scoped("initial_generation"):
        (
            initial_ir,
            initial_snippets,
            initial_reports,
            initial_code,
            initial_plan_state,
        ) = subtree_replan._run_single_pass_with_state(problem_text)

    canonical_cases = _canonical_sample_cases(io_data, max_cases)
    initial_ir = _install_canonical_validation(initial_ir, canonical_cases)

    initial_fv_state: dict[str, Any] = {
        "problem_id": problem_id,
        "final_code": initial_code,
        "task_plan_ir": initial_ir,
        "code_snippets": initial_snippets,
        "verification_reports": initial_reports,
    }
    initial_runner = full_code_runner(initial_fv_state)
    initial_execution = initial_runner.get("execution_result", {})
    initial_apps_tests = subtree_replan.run_apps_tests(
        initial_code,
        io_data,
        max_cases=max_cases,
        problem_id=problem_id,
    )

    final_ir = copy.deepcopy(initial_ir)
    final_snippets = copy.deepcopy(initial_snippets)
    final_reports = copy.deepcopy(initial_reports)
    final_code = initial_code
    final_execution = initial_execution

    error_location_out: dict[str, Any] | None = None
    localization_record_out: dict[str, Any] | None = {
        "status": "not_needed",
        "method": None,
        "error_type": "none",
    }
    branch_root_id = ""
    repair_region_nodes: set[str] = set()
    subtree_replan_attempted = False
    global_replan_attempted = False
    repair_scope = "none"
    repair_trigger_location: dict[str, Any] | None = None
    repair_acceptance: dict[str, Any] | None = None

    failure_evidence, disagreement_record = _select_failure_evidence(
        initial_execution,
        initial_apps_tests,
    )
    if disagreement_record is not None:
        localization_record_out = disagreement_record

    if failure_evidence is not None:
        locate_state = {**initial_fv_state, "execution_result": failure_evidence}
        with TOKEN_TRACKER.scoped("localization"):
            locator_result = error_locator(locate_state)
        error_location_out = locator_result.get("error_location")
        localization_record_out = locator_result.get("localization_record")

        if error_location_out:
            repair_trigger_location = error_location_out
            node_map = _node_map(initial_ir)
            root_id = initial_ir.get("root_id", "")
            failing_task_id = error_location_out.get("task_id", "")
            branch_root_id = subtree_replan._find_error_branch_root(
                failing_task_id, node_map, root_id
            )
            repair_region_nodes = subtree_replan._collect_subtree_node_ids(
                branch_root_id, node_map
            )

            subtree_replan_attempted = True
            repair_scope = "subtree"
            with TOKEN_TRACKER.scoped("repair"):
                (
                    final_ir,
                    final_snippets,
                    final_reports,
                    final_code,
                    branch_root_id,
                ) = subtree_replan._run_subtree_replan(
                    initial_ir,
                    initial_snippets,
                    initial_reports,
                    error_location_out,
                    original_problem_text=problem_text,
                )

            final_ir = _install_canonical_validation(final_ir, canonical_cases)
            final_fv_state: dict[str, Any] = {
                "problem_id": problem_id,
                "final_code": final_code,
                "task_plan_ir": final_ir,
                "code_snippets": final_snippets,
                "verification_reports": final_reports,
            }
            final_runner = full_code_runner(final_fv_state)
            final_execution = final_runner.get("execution_result", initial_execution)
        else:
            root_id = initial_ir.get("root_id", "")
            repair_trigger_location = {
                "task_id": root_id,
                "method": "global_fallback_after_abstention",
                "confidence": "scope_only",
                "scope": "global",
                "error_type": failure_evidence.get("error_type", "unknown"),
                "failing_case": _first_failing_case(failure_evidence),
            }
            global_replan_attempted = True
            repair_scope = "global"
            branch_root_id = root_id
            repair_region_nodes = set(_node_map(initial_ir))
            with TOKEN_TRACKER.scoped("repair"):
                (
                    final_ir,
                    final_snippets,
                    final_reports,
                    final_code,
                ) = _run_global_replan(
                    initial_plan_state,
                    _global_failure_context(
                        failure_evidence,
                        localization_record_out,
                    ),
                )
            final_ir = _install_canonical_validation(final_ir, canonical_cases)
            final_runner = full_code_runner(
                {
                    "problem_id": problem_id,
                    "final_code": final_code,
                    "task_plan_ir": final_ir,
                    "code_snippets": final_snippets,
                    "verification_reports": final_reports,
                }
            )
            final_execution = final_runner.get(
                "execution_result",
                initial_execution,
            )

    final_apps_tests = subtree_replan.run_apps_tests(
        final_code,
        io_data,
        max_cases=max_cases,
        problem_id=problem_id,
    )

    if repair_scope != "none":
        repair_acceptance = _repair_acceptance_record(
            initial_execution=initial_execution,
            initial_apps_tests=initial_apps_tests,
            candidate_execution=final_execution,
            candidate_apps_tests=final_apps_tests,
            candidate_code=final_code,
        )
        if not repair_acceptance["accepted"]:
            final_ir = copy.deepcopy(initial_ir)
            final_snippets = copy.deepcopy(initial_snippets)
            final_reports = copy.deepcopy(initial_reports)
            final_code = initial_code
            final_execution = copy.deepcopy(initial_execution)
            final_apps_tests = copy.deepcopy(initial_apps_tests)

    initial_dctc = compute_dctc(initial_ir, initial_snippets, initial_reports, initial_code)
    final_dctc = compute_dctc(final_ir, final_snippets, final_reports, final_code)
    repair_metrics = compute_repair_metrics(
        initial_ir=initial_ir,
        final_ir=final_ir,
        initial_snippets=initial_snippets,
        final_snippets=final_snippets,
        initial_reports=initial_reports,
        final_reports=final_reports,
        initial_code=initial_code,
        final_code=final_code,
        initial_execution=initial_execution,
        final_execution=final_execution,
        error_location=error_location_out,
        repair_region_nodes=repair_region_nodes,
        initial_apps_tests=initial_apps_tests,
        final_apps_tests=final_apps_tests,
        # The subtree pipeline regenerates/revalidates only the selected region.
        # Frozen nodes retain their old reports, so RR-unit remains unavailable
        # until a separate frozen-node revalidation pass is implemented.
        reverified_node_ids=set(),
    )
    repair_record = build_repair_record(
        initial_ir=initial_ir,
        initial_snippets=initial_snippets,
        final_snippets=final_snippets,
        initial_execution=initial_execution,
        final_execution=final_execution,
        initial_apps_tests=initial_apps_tests,
        final_apps_tests=final_apps_tests,
        localization_record=localization_record_out,
        error_branch_root=branch_root_id,
        repair_region_nodes=repair_region_nodes,
    )

    return {
        "token_usage": TOKEN_TRACKER.snapshot(),
        "initial": {
            "ir": initial_ir,
            "code_snippets": initial_snippets,
            "verification_reports": initial_reports,
            "final_code": initial_code,
            "execution_result": initial_execution,
            "apps_test_results": initial_apps_tests,
            "dctc": initial_dctc,
        },
        "final": {
            "ir": final_ir,
            "code_snippets": final_snippets,
            "verification_reports": final_reports,
            "final_code": final_code,
            "execution_result": final_execution,
            "apps_test_results": final_apps_tests,
            "dctc": final_dctc,
        },
        "audit": {
            **repair_metrics,
            "initial_dctc_leaf": initial_dctc["dctc_leaf"],
            "final_dctc_leaf": final_dctc["dctc_leaf"],
            "initial_dctc_final_func": initial_dctc["dctc_final_func"],
            "final_dctc_final_func": final_dctc["dctc_final_func"],
            "initial_unique_mapping_rate": initial_dctc["unique_mapping_rate"],
            "final_unique_mapping_rate": final_dctc["unique_mapping_rate"],
            "initial_assembly_preservation_rate": initial_dctc[
                "assembly_preservation_rate"
            ],
            "final_assembly_preservation_rate": final_dctc[
                "assembly_preservation_rate"
            ],
            "subtree_replan_attempted": subtree_replan_attempted,
            "global_replan_attempted": global_replan_attempted,
            "repair_scope": repair_scope,
            "repair_acceptance": repair_acceptance,
            "problem_statement_supplement_applied": supplement_applied,
            "repair_trigger_location": repair_trigger_location,
            "error_location": error_location_out,
            "localization_record": localization_record_out,
            "repair_record": repair_record,
            "error_branch_root": branch_root_id,
            "repair_region_nodes": sorted(repair_region_nodes),
            "evidence_type": (error_location_out or {}).get("method"),
        },
    }


def build_result_json(
    item: dict[str, Any],
    audit_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "problem_id": item["id"],
        "difficulty": item.get("difficulty", "unknown"),
        "url": item.get("url", ""),
        "question": item.get("question", ""),
        "timestamp": datetime.now().isoformat(),
        "experiment": "audit_metrics_subtree_replan",
        "llm_provider": "deepseek",
        "llm_model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "token_usage": audit_result.get("token_usage", {}),
        "audit_overhead": {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "note": (
                "审计日志记录、hash/diff 计算、DCTC/LC/LRSR/RRS/CCR/UVCR/RR 等"
                "指标计算均在本地完成，不额外调用 LLM。token_usage 只统计 AuditCoder pipeline"
                "本身的模型调用。"
            ),
        },
        "selected_metrics": [
            "DCTC-leaf",
            "DCTC-final-func",
            "Unique-function-owner-rate",
            "Assembly-preservation-rate",
            "LC",
            "LRSR",
            "LRSR-APPS",
            "External-repair-recovery",
            "External-pass-preservation",
            "ORY",
            "ORY-APPS",
            "RRS-node",
            "RRS-implementation-node",
            "RRS-LOC",
            "CCR-LOC",
            "HRSR-LOC",
            "UVCR-node",
            "RR-unit-report",
            "RR-unit-evaluation-coverage",
            "RR-test",
        ],
        "metrics": audit_result["audit"],
        "initial": audit_result["initial"],
        "final": audit_result["final"],
    }


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _add_token_averages(bucket: dict[str, Any], task_count: int) -> None:
    calls = int(bucket.get("calls") or 0)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = int(bucket.get(key) or 0)
        bucket[f"avg_{key}_per_call"] = value / calls if calls else None
        bucket[f"avg_{key}_per_task"] = value / task_count if task_count else None


def _finalize_token_averages(token_usage: dict[str, Any], task_count: int) -> None:
    _add_token_averages(token_usage.get("total", {}), task_count)

    for stats in (token_usage.get("by_phase") or {}).values():
        _add_token_averages(stats, task_count)

    for stats in (token_usage.get("by_module") or {}).values():
        _add_token_averages(stats, task_count)

    for modules in (token_usage.get("by_phase_module") or {}).values():
        for stats in modules.values():
            _add_token_averages(stats, task_count)


def summarize_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = [
        "initial_dctc_leaf",
        "final_dctc_leaf",
        "initial_dctc_final_func",
        "final_dctc_final_func",
        "initial_unique_mapping_rate",
        "final_unique_mapping_rate",
        "initial_assembly_preservation_rate",
        "final_assembly_preservation_rate",
        "localization_coverage",
        "localized_repair_success",
        "localized_repair_success_apps",
        "external_repair_recovery",
        "external_pass_preservation",
        "overall_repair_yield",
        "overall_repair_yield_apps",
        "rrs_node",
        "rrs_implementation_node",
        "rrs_loc",
        "ccr_loc",
        "hrsr_loc",
        "uvcr_node",
        "rr_unit",
        "rr_unit_report_based",
        "rr_unit_evaluation_coverage",
        "rr_test",
    ]
    summary: dict[str, Any] = {
        "task_count": len(results),
        "pipeline_exception_count": sum(1 for r in results if r.get("error")),
        "metrics": {},
        "token_usage": {
            "total": {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "calls_missing_usage": 0,
            },
            "by_phase": {},
            "by_module": {},
            "by_phase_module": {},
        },
        "audit_overhead": {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "note": "审计指标计算和 JSON 记录为本地计算，不额外消耗 LLM token。",
        },
    }

    ok_results = [r for r in results if not r.get("error")]
    for key in metric_keys:
        vals = [
            float(r["metrics"][key])
            for r in ok_results
            if r.get("metrics", {}).get(key) is not None
        ]
        summary["metrics"][key] = {
            "n": len(vals),
            "mean": _mean(vals),
            "median": _median(vals),
        }

    for result in ok_results:
        token_usage = result.get("token_usage") or {}
        total = token_usage.get("total") or {}
        for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens", "calls_missing_usage"):
            summary["token_usage"]["total"][key] += int(total.get(key) or 0)

        for module, stats in (token_usage.get("by_module") or {}).items():
            bucket = summary["token_usage"]["by_module"].setdefault(
                module,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "calls_missing_usage": 0,
                },
            )
            for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens", "calls_missing_usage"):
                bucket[key] += int(stats.get(key) or 0)

        for phase, stats in (token_usage.get("by_phase") or {}).items():
            bucket = summary["token_usage"]["by_phase"].setdefault(
                phase,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "calls_missing_usage": 0,
                },
            )
            for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens", "calls_missing_usage"):
                bucket[key] += int(stats.get(key) or 0)

        for phase, modules in (token_usage.get("by_phase_module") or {}).items():
            phase_bucket = summary["token_usage"]["by_phase_module"].setdefault(phase, {})
            for module, stats in modules.items():
                bucket = phase_bucket.setdefault(
                    module,
                    {
                        "calls": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "calls_missing_usage": 0,
                    },
                )
                for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens", "calls_missing_usage"):
                    bucket[key] += int(stats.get(key) or 0)

    _finalize_token_averages(summary["token_usage"], len(ok_results))
    return summary


def run_self_test() -> None:
    """Validate metric formulas on a deterministic synthetic repair case."""
    initial_ir = {
        "root_id": "S0",
        "task_nodes": [
            {"id": "S0", "is_leaf": False, "parent": None},
            {
                "id": "S1",
                "is_leaf": True,
                "parent": "S0",
                "agent_type": "math",
                "inputs": [{"name": "x", "type": "int", "source": None}],
                "interface": {
                    "function_name": "good",
                    "params": [{"name": "x", "type": "int"}],
                    "return_type": "int",
                },
            },
            {
                "id": "S2",
                "is_leaf": True,
                "parent": "S0",
                "agent_type": "math",
                "inputs": [{"name": "x", "type": "int", "source": None}],
                "interface": {
                    "function_name": "bad",
                    "params": [{"name": "x", "type": "int"}],
                    "return_type": "int",
                },
            },
        ],
    }
    final_ir = copy.deepcopy(initial_ir)
    initial_snippets = {
        "S1": "def good(x):\n    return x + 1\n",
        "S2": "def bad(x):\n    return x - 1\n",
    }
    final_snippets = {
        "S1": initial_snippets["S1"],
        "S2": "def bad(x):\n    return x + 1\n",
    }
    initial_reports = {
        "S1": {"verdict": "accept"},
        "S2": {"verdict": "reject"},
    }
    final_reports = {
        "S1": {"verdict": "accept"},
        "S2": {"verdict": "accept"},
    }
    initial_code = initial_snippets["S1"] + "\n" + initial_snippets["S2"]
    final_code = final_snippets["S1"] + "\n" + final_snippets["S2"]
    initial_execution = {"all_passed": False}
    final_execution = {"all_passed": True}
    error_location = {"task_id": "S2", "method": "verification_report"}
    initial_apps_tests = {
        "all_passed": False,
        "cases": [
            {"index": 0, "passed": True},
            {"index": 1, "passed": False},
        ]
    }
    final_apps_tests = {
        "all_passed": True,
        "cases": [
            {"index": 0, "passed": True},
            {"index": 1, "passed": True},
        ]
    }

    dctc = compute_dctc(initial_ir, initial_snippets, initial_reports, initial_code)
    assert dctc["dctc_leaf"] == 1.0, dctc
    assert dctc["dctc_final_func"] == 1.0, dctc

    class FakeResponse:
        usage_metadata = {
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
        }

    TOKEN_TRACKER.reset()
    with TOKEN_TRACKER.scoped("repair"):
        TOKEN_TRACKER.record("self_test.fake_module", FakeResponse())
    token_usage = TOKEN_TRACKER.snapshot()
    assert token_usage["total"]["total_tokens"] == 18, token_usage
    assert token_usage["by_phase"]["repair"]["total_tokens"] == 18, token_usage
    assert token_usage["by_module"]["self_test.fake_module"]["prompt_tokens"] == 11, token_usage
    assert token_usage["calls"][0]["phase"] == "repair", token_usage

    metrics = compute_repair_metrics(
        initial_ir=initial_ir,
        final_ir=final_ir,
        initial_snippets=initial_snippets,
        final_snippets=final_snippets,
        initial_reports=initial_reports,
        final_reports=final_reports,
        initial_code=initial_code,
        final_code=final_code,
        initial_execution=initial_execution,
        final_execution=final_execution,
        error_location=error_location,
        repair_region_nodes={"S2"},
        initial_apps_tests=initial_apps_tests,
        final_apps_tests=final_apps_tests,
        reverified_node_ids={"S1"},
    )
    assert metrics["localization_coverage"] == 1, metrics
    assert metrics["localized_repair_success"] == 1, metrics
    assert metrics["localized_repair_success_apps"] == 1, metrics
    assert metrics["external_repair_recovery"] == 1, metrics
    assert metrics["external_pass_preservation"] is None, metrics
    assert metrics["overall_repair_yield"] == 1, metrics
    assert metrics["overall_repair_yield_apps"] == 1, metrics
    assert metrics["rrs_node"] == 1 / 3, metrics
    assert metrics["rrs_implementation_node"] == 0.5, metrics
    assert metrics["uvcr_node"] == 1.0, metrics
    assert metrics["rr_unit"] == 0.0, metrics
    assert metrics["rr_unit_report_based"] == 0.0, metrics
    assert metrics["rr_test"] == 0.0, metrics
    assert metrics["ccr_loc"] is not None and metrics["ccr_loc"] > 0, metrics
    assert metrics["hrsr_loc"] is not None and 0 <= metrics["hrsr_loc"] <= 1, metrics
    print("self-test passed")


def run_single_apps_problem(
    item: dict[str, Any],
    output_dir: Path,
    *,
    max_cases: int,
    verbose: bool,
) -> dict[str, Any]:
    problem_id = item["id"]
    output_file = output_dir / f"apps_{problem_id}.json"

    if verbose:
        difficulty = item.get("difficulty", "?")
        short_q = item.get("question", "")[:80]
        suffix = "..." if len(item.get("question", "")) > 80 else ""
        print(f"\n[Problem {problem_id} | {difficulty}] {short_q}{suffix}")

    try:
        subtree_replan = _load_subtree_replan()
        io_data = subtree_replan.parse_io_field(item.get("input_output"))
        problem_text = subtree_replan.format_apps_problem(item)
        audit_result = run_audit_pipeline(
            problem_text,
            io_data,
            max_cases,
            problem_id=problem_id,
        )
        result = build_result_json(item, audit_result)

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        if verbose:
            m = result["metrics"]
            print(
                "  pipeline: "
                f"{m['initial_pipeline_pass']} -> {m['final_pipeline_pass']} | "
                f"LC={m['localization_coverage']} "
                f"LRSR={m['localized_repair_success']} "
                f"LRSR-APPS={m['localized_repair_success_apps']} "
                f"RRS-node={m['rrs_node']} "
                f"UVCR={m['uvcr_node']}"
            )
            print(f"  保存至: {output_file}")

        return result

    except Exception as exc:
        error_result: dict[str, Any] = {
            "problem_id": problem_id,
            "difficulty": item.get("difficulty", "unknown"),
            "url": item.get("url", ""),
            "question": item.get("question", ""),
            "timestamp": datetime.now().isoformat(),
            "experiment": "audit_metrics_subtree_replan",
            "llm_provider": "deepseek",
            "llm_model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "audit_overhead": {
                "llm_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "note": "审计指标计算和 JSON 记录为本地计算，不额外消耗 LLM token。",
            },
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(error_result, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"  [错误] Pipeline 异常：{exc}")
            print(f"  错误详情已保存至：{output_file}")
        return error_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="APPS 审计指标批量测试（基于错误子树重规划 pipeline）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="只运行审计指标公式自测，不加载 auditcoder pipeline 或调用 LLM",
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--offset", type=int, metavar="N", default=None)
    group.add_argument("--ids", type=str, metavar="ID,ID,...")
    parser.add_argument("--count", type=int, metavar="N", default=100)
    parser.add_argument("--difficulty", choices=list(VALID_DIFFICULTIES), default=None)
    parser.add_argument("--max-cases", type=int, metavar="N", default=10)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--data", type=str, default=str(APPS_DATA_PATH))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    if args.offset is None and not args.ids:
        parser.error("需要指定 --offset 或 --ids；若只检查指标公式，请使用 --self-test")

    try:
        subtree_replan = _load_subtree_replan()
    except ModuleNotFoundError as exc:
        print(
            f"[错误] 缺少运行 AuditCoder pipeline 所需依赖：{exc.name}。"
            "请先安装 requirements.txt；若只想检查指标公式，运行 --self-test。",
            file=sys.stderr,
        )
        sys.exit(1)
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[错误] 数据集文件不存在：{data_path}", file=sys.stderr)
        sys.exit(1)

    if args.ids:
        id_list = [int(x.strip()) for x in args.ids.split(",") if x.strip()]
        selected = subtree_replan.load_apps_by_ids(data_path, id_list)
    else:
        offset = args.offset if args.offset is not None else 0
        selected = subtree_replan.load_apps_batch(
            data_path,
            offset=offset,
            count=args.count,
            difficulty=args.difficulty,
        )

    if not selected:
        print("[错误] 未选取到任何题目。", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)
    verbose = not args.quiet

    print("[审计指标测试：subtree replan]")
    print(f"模型：DeepSeek / {os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')}")
    print(f"共选取 {len(selected)} 题 | 输出目录：{output_dir}")
    print(f"每题最多运行 {args.max_cases} 个 APPS 用例")

    results: list[dict[str, Any]] = []
    for item in selected:
        results.append(
            run_single_apps_problem(
                item,
                output_dir,
                max_cases=args.max_cases,
                verbose=verbose,
            )
        )

    summary = summarize_metrics(results)
    summary_file = output_dir / "summary_audit_metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n完成审计指标测试")
    print(f"summary: {summary_file}")
    token_total = summary["token_usage"]["total"]
    print(
        "token: "
        f"calls={token_total['calls']}, "
        f"prompt={token_total['prompt_tokens']}, "
        f"completion={token_total['completion_tokens']}, "
        f"total={token_total['total_tokens']}, "
        f"missing_usage={token_total['calls_missing_usage']}"
    )
    repair_token = (summary["token_usage"].get("by_phase") or {}).get("repair")
    if repair_token:
        print(
            "repair token: "
            f"calls={repair_token['calls']}, "
            f"prompt={repair_token['prompt_tokens']}, "
            f"completion={repair_token['completion_tokens']}, "
            f"total={repair_token['total_tokens']}"
        )
    for key, stats in summary["metrics"].items():
        if stats["n"]:
            print(f"  {key}: mean={stats['mean']:.4f}, median={stats['median']:.4f}, n={stats['n']}")


if __name__ == "__main__":
    main()
