"""Controlled-fault experiment for AuditCoder's localization and bounded repair.

This experiment deliberately uses an authored TaskPlanIR.  The graph therefore
has a known responsibility owner for every implementation function, while
DeepSeek is still used for the node implementations, assembly, localization
repair planning, and regenerated node code.

The live run performs the following steps:

1. Generate two leaf implementations with ``dispatch_algo_agents``.
2. Assemble and validate one complete interval-merging program.
3. Inject faults into one leaf at a time (plus an unmapped ``solve`` fault).
4. Run ``final_verifier_agent`` on each faulty program.
5. Record localization, bounded-repair, frozen-complement, and ground-truth
   metrics in JSON.

The ``--self-test`` mode exercises fault mutation and metric calculations using
local deterministic code, so it does not need an API key or AuditCoder's
optional LangGraph dependencies.

Usage (from the repository root)::

    python tests/controlled_fault_injection_deepseek.py --self-test
    python tests/controlled_fault_injection_deepseek.py --output result/fault_injection

The live mode reads ``DEEPSEEK_API_KEY``, ``DEEPSEEK_BASE_URL`` and
``DEEPSEEK_MODEL`` from ``.env`` or the process environment.  It uses the same
OpenAI-compatible DeepSeek adapter as the existing DeepSeek runners.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    # The checkout is commonly nested under a workspace directory that already
    # contains the user's .env.  Load both locations, while letting explicit
    # process variables keep precedence.
    load_dotenv(REPO_ROOT / ".env", override=False)
except ModuleNotFoundError:  # pragma: no cover - only relevant to self-test
    pass


PROBLEM_ID = "controlled_interval_merge_v1"

PROBLEM_TEXT = """\
Given n closed integer intervals [l_i, r_i], merge every pair of intervals
that overlap or touch (the next interval starts at most one greater than the
current right endpoint).  Print the number of resulting connected groups.

Input format:
The first line contains an integer n.  Each of the next n lines contains two
integers l_i and r_i with l_i <= r_i.

Output format:
Print one integer: the number of connected groups after merging.

Constraints:
1 <= n <= 200000, -10^9 <= l_i <= r_i <= 10^9.
The intended time complexity is O(n log n) and the intended extra space is O(n).
"""

SAMPLE_CASES: list[dict[str, str]] = [
    {
        "input": "5\n1 3\n2 4\n5 6\n8 9\n8 10\n",
        "expected_output": "2",
    },
    {
        "input": "4\n1 1\n2 2\n3 3\n4 4\n",
        "expected_output": "1",
    },
    {
        "input": "3\n1 4\n2 3\n6 8\n",
        "expected_output": "2",
    },
    {
        "input": "1\n-5 -5\n",
        "expected_output": "1",
    },
]


def build_ground_truth_ir() -> dict[str, Any]:
    """Return the fixed synthetic task graph used by the experiment.

    The graph is intentionally small, but has two independently owned leaves
    so that a repair can be measured against a non-empty frozen complement.
    """

    return {
        "execution_contract": {
            "io_mode": "stdin_stdout",
            "stdin_encoding": "utf-8",
            "stdout_encoding": "utf-8",
        },
        "problem": {
            "title": "Count connected groups of intervals",
            "description": PROBLEM_TEXT,
            "original_specification": PROBLEM_TEXT,
            "input_format": (
                "n followed by n lines, each containing l_i and r_i"
            ),
            "output_format": "one integer followed by a newline",
            "constraints": {"time_limit": "2s", "memory_limit": "256MB"},
            "sample_cases": SAMPLE_CASES,
        },
        "analysis": {
            "problem_category": "sorting and greedy interval merging",
            "key_observations": [
                "Sorting by left endpoint makes a single linear merge pass sufficient.",
                "Intervals that touch are in the same group, so use next_left <= right + 1.",
            ],
            "complexity_target": {"time": "O(n log n)", "space": "O(n)"},
            "algorithm_selection": ["sorting", "greedy"],
            "reasoning_chain": [
                "Sort intervals by (left, right).",
                "Scan in order and start a new group only after a gap.",
            ],
        },
        "global_strategy": (
            "Parse the intervals, sort them by left endpoint, greedily count a new "
            "group whenever the next left endpoint is greater than current_right + 1, "
            "and print the count."
        ),
        "complexity_budget": {"time": "O(n log n)", "space": "O(n)"},
        "task_nodes": [
            {
                "id": "S0",
                "goal": "Read input, coordinate interval sorting and greedy merging, and print the answer.",
                "type": "orchestration",
                "is_leaf": False,
                "agent_type": None,
                "parent": None,
                "dependencies": ["S1", "S2"],
                "inputs": [],
                "outputs": [],
                "interface": None,
                "hints": [],
                "complexity_budget": "O(n log n)",
                "context": "Root orchestration node; its two child leaves own the algorithmic functions.",
            },
            {
                "id": "S1",
                "goal": "Sort all intervals by left endpoint and then right endpoint.",
                "type": "sorting",
                "is_leaf": True,
                "agent_type": "sorting",
                "parent": "S0",
                "dependencies": [],
                "inputs": [
                    {
                        "name": "intervals",
                        "type": "list[tuple[int, int]]",
                        "description": "the n closed intervals",
                        "source": None,
                    }
                ],
                "outputs": [
                    {
                        "name": "sorted_intervals",
                        "type": "list[tuple[int, int]]",
                        "description": "intervals sorted lexicographically",
                        "source": None,
                    }
                ],
                "interface": {
                    "function_name": "sort_intervals",
                    "params": [
                        {
                            "name": "intervals",
                            "type": "list[tuple[int, int]]",
                            "description": "the intervals to sort",
                        }
                    ],
                    "return_type": "list[tuple[int, int]]",
                    "return_description": "a sorted copy of intervals",
                    "preconditions": ["each interval is a pair (l, r) with l <= r"],
                    "postconditions": [
                        "the output contains exactly the input intervals",
                        "the output is ordered by (l, r)",
                    ],
                },
                "hints": [
                    "Return sorted(intervals); do not mutate the caller's list.",
                    "The caller relies on lexicographic (left, right) ordering.",
                ],
                "complexity_budget": "O(n log n)",
                "context": "Independent sorting leaf. Its output is consumed by S2.",
            },
            {
                "id": "S2",
                "goal": "Count connected interval groups in a list sorted by left endpoint; touching intervals belong to one group.",
                "type": "greedy",
                "is_leaf": True,
                "agent_type": "greedy",
                "parent": "S0",
                "dependencies": ["S1"],
                "inputs": [
                    {
                        "name": "sorted_intervals",
                        "type": "list[tuple[int, int]]",
                        "description": "intervals sorted by (left, right)",
                        "source": "S1.sorted_intervals",
                    }
                ],
                "outputs": [
                    {
                        "name": "group_count",
                        "type": "int",
                        "description": "number of connected groups",
                        "source": None,
                    }
                ],
                "interface": {
                    "function_name": "count_merged_groups",
                    "params": [
                        {
                            "name": "sorted_intervals",
                            "type": "list[tuple[int, int]]",
                            "description": "intervals sorted by left endpoint",
                        }
                    ],
                    "return_type": "int",
                    "return_description": "number of groups after merging overlaps and touching intervals",
                    "preconditions": [
                        "sorted_intervals is sorted by (left, right)",
                        "each interval is a pair (l, r) with l <= r",
                    ],
                    "postconditions": ["the result is at least 1 because n >= 1"],
                },
                "hints": [
                    "Initialize the first group's right endpoint and count one group.",
                    "Start another group only when next_left > current_right + 1.",
                ],
                "complexity_budget": "O(n)",
                "context": "Independent greedy leaf. It consumes S1.sorted_intervals and returns the final count.",
            },
        ],
        "root_id": "S0",
        "execution_order": ["S1", "S2"],
        "validation": {
            "sample_test_cases": SAMPLE_CASES,
            "edge_cases": [
                {"input": "1\n0 0\n", "expected_output": "1"},
                {"input": "2\n0 1\n2 3\n", "expected_output": "1"},
            ],
            "invariants": [
                "The final count is between 1 and n.",
                "Intervals that touch at a boundary are merged.",
            ],
        },
    }


def _node_map(ir: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(node["id"]): node
        for node in ir.get("task_nodes", [])
        if isinstance(node, dict) and node.get("id")
    }


def _function_name(ir: dict[str, Any], task_id: str) -> str:
    node = _node_map(ir).get(task_id, {})
    return str((node.get("interface") or {}).get("function_name") or "")


def _ast_mutate_function(
    source: str,
    function_name: str,
    mutate: Callable[[ast.FunctionDef], None],
) -> str:
    """Apply a deterministic AST mutation to exactly one function."""

    tree = ast.parse(source)
    found = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            if not isinstance(node, ast.FunctionDef):
                raise ValueError(f"target function {function_name!r} is async")
            mutate(node)
            found += 1
    if found != 1:
        raise ValueError(
            f"expected exactly one function named {function_name!r}, found {found}"
        )
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def inject_runtime_fault(source: str, function_name: str, task_id: str) -> str:
    """Raise inside the target function so traceback evidence names its owner."""

    def mutate(node: ast.FunctionDef) -> None:
        node.body.insert(
            0,
            ast.Raise(
                exc=ast.Call(
                    func=ast.Name(id="RuntimeError", ctx=ast.Load()),
                    args=[
                        ast.Constant(
                            value=f"AUDITCODER_INJECTED_RUNTIME:{task_id}"
                        )
                    ],
                    keywords=[],
                ),
                cause=None,
            ),
        )

    return _ast_mutate_function(source, function_name, mutate)


def inject_wrong_answer_fault(source: str, function_name: str) -> str:
    """Return a deterministic impossible count, without producing a traceback."""

    def mutate(node: ast.FunctionDef) -> None:
        node.body = [ast.Return(value=ast.Constant(value=-999999))]

    return _ast_mutate_function(source, function_name, mutate)


def inject_unmapped_runtime_fault(source: str) -> str:
    """Raise from the conventional orchestration function outside leaf ownership."""

    orchestration_name = next(
        (
            name
            for name in ("solve", "main", "solution")
            if name in _function_defs(source)
        ),
        None,
    )
    if orchestration_name is None:
        raise ValueError(
            "assembled program has no conventional unmapped orchestration "
            "function (expected solve, main, or solution)"
        )
    return _ast_mutate_function(
        source,
        orchestration_name,
        lambda node: node.body.insert(
            0,
            ast.Raise(
                exc=ast.Call(
                    func=ast.Name(id="RuntimeError", ctx=ast.Load()),
                    args=[ast.Constant(value="AUDITCODER_INJECTED_GLOBAL")],
                    keywords=[],
                ),
                cause=None,
            ),
        ),
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _function_defs(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _expected_repair_region(
    *,
    ir: dict[str, Any],
    target_task_id: str | None,
) -> set[str]:
    """The node-level verifier graph should only retry the located leaf."""

    if not target_task_id:
        return set()
    if target_task_id not in _node_map(ir):
        return set()
    return {target_task_id}


def evaluate_fault_case(
    *,
    fault: str,
    target_task_id: str | None,
    initial_state: dict[str, Any],
    final_state: dict[str, Any],
    ir: dict[str, Any],
) -> dict[str, Any]:
    """Compute case-level GT localization and bounded-repair metrics."""

    final_report = final_state.get("final_report") or {}
    location = final_report.get("error_location") or {}
    located_task_id = location.get("task_id") or None
    localized = bool(located_task_id)
    expected_abstention = target_task_id is None
    correct = (
        (not localized)
        if expected_abstention
        else located_task_id == target_task_id
    )
    false_localization = localized and (
        expected_abstention or located_task_id != target_task_id
    )

    repair_info = final_report.get("repair_info") or {}
    repair_history = final_report.get("repair_history") or []
    region: set[str] = set()
    if repair_history and isinstance(repair_history[0], dict):
        region = set((repair_history[0].get("repair_region") or {}).get("nodes") or [])
    if not region:
        region = set(final_state.get("retried_tasks") or [])

    all_node_ids = set(_node_map(ir))
    expected_region = _expected_repair_region(ir=ir, target_task_id=target_task_id)
    intersection = region & expected_region
    union = region | expected_region
    final_exec = final_state.get("execution_result") or {}
    initial_exec = (
        final_state.get("initial_execution_result")
        or initial_state.get("execution_result")
        or {}
    )

    initial_snippets = initial_state.get("code_snippets") or {}
    final_snippets = final_state.get("code_snippets") or {}
    frozen_ids = all_node_ids - expected_region
    frozen_ids &= set(initial_snippets)
    preserved_frozen = sum(
        1
        for task_id in frozen_ids
        if final_snippets.get(task_id) == initial_snippets.get(task_id)
    )
    frozen_rate = preserved_frozen / len(frozen_ids) if frozen_ids else None

    return {
        "fault": fault,
        "ground_truth_task_id": target_task_id,
        "expected_abstention": expected_abstention,
        "localized": localized,
        "located_task_id": located_task_id,
        "localization_correct": bool(correct),
        "false_localization": bool(false_localization),
        "localization_method": location.get("method"),
        "localization_confidence": location.get("confidence"),
        "initial_error_type": (initial_exec or {}).get("error_type"),
        "initial_all_passed": bool((initial_exec or {}).get("all_passed")),
        "final_all_passed": bool(final_exec.get("all_passed")),
        "repair_attempted": bool(final_state.get("retried_tasks")),
        "repair_success": bool(final_exec.get("all_passed")),
        "repair_region_nodes": sorted(region),
        "ground_truth_repair_region": sorted(expected_region),
        "repair_boundary_precision": (
            len(intersection) / len(region) if region else 0.0
        ),
        "repair_boundary_recall": (
            len(intersection) / len(expected_region)
            if expected_region
            else None
        ),
        "repair_boundary_iou": len(intersection) / len(union) if union else None,
        "frozen_snippet_preservation": frozen_rate,
        "initial_code_sha256": _sha256(str(initial_state.get("final_code", ""))),
        "final_code_sha256": _sha256(str(final_state.get("final_code", ""))),
        "final_report": final_report,
    }


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate precision, false-localization, abstention, and repair rates."""

    total = len(cases)
    localized = sum(bool(case.get("localized")) for case in cases)
    # Precision@1 is defined over decisions that actually name a task.  A
    # negative-control abstention is tracked separately and must not inflate
    # localization precision above 1.0.
    correct_localized = sum(
        bool(case.get("localized"))
        and not bool(case.get("expected_abstention"))
        and bool(case.get("localization_correct"))
        for case in cases
    )
    wrong = sum(bool(case.get("false_localization")) for case in cases)
    expected_targets = [case for case in cases if not case.get("expected_abstention")]
    negative_controls = [case for case in cases if case.get("expected_abstention")]
    localized_targets = [case for case in expected_targets if case.get("localized")]
    correctly_localized_targets = [
        case
        for case in localized_targets
        if case.get("localization_correct")
    ]
    repair_attempts = [case for case in cases if case.get("repair_attempted")]

    def rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    def mean(key: str, selected: list[dict[str, Any]]) -> float | None:
        values = [
            float(case[key])
            for case in selected
            if case.get(key) is not None
        ]
        return sum(values) / len(values) if values else None

    return {
        "case_count": total,
        "targeted_fault_count": len(expected_targets),
        "negative_control_count": len(negative_controls),
        "localized_count": localized,
        "correct_localization_count": correct_localized,
        "false_localization_count": wrong,
        "localization_precision_when_decided": rate(correct_localized, localized),
        "ground_truth_localization_rate": rate(
            sum(bool(case.get("localization_correct")) for case in expected_targets),
            len(expected_targets),
        ),
        "false_localization_rate_over_all_cases": rate(wrong, total),
        "false_localization_rate_when_decided": rate(wrong, localized),
        "abstention_rate": rate(total - localized, total),
        "correct_abstention_rate_on_negative_controls": rate(
            sum(
                bool(case.get("expected_abstention")) and not bool(case.get("localized"))
                for case in cases
            ),
            len(negative_controls),
        ),
        "repair_attempt_count": len(repair_attempts),
        "repair_success_count": sum(bool(case.get("repair_success")) for case in repair_attempts),
        "repair_success_rate_when_attempted": rate(
            sum(bool(case.get("repair_success")) for case in repair_attempts),
            len(repair_attempts),
        ),
        "targeted_repair_success_rate": rate(
            sum(bool(case.get("repair_success")) for case in correctly_localized_targets),
            len(correctly_localized_targets),
        ),
        "mean_repair_boundary_precision": mean("repair_boundary_precision", repair_attempts),
        "mean_repair_boundary_recall": mean("repair_boundary_recall", repair_attempts),
        "mean_repair_boundary_iou": mean("repair_boundary_iou", repair_attempts),
        "mean_frozen_snippet_preservation": mean(
            "frozen_snippet_preservation", repair_attempts
        ),
    }


@dataclass(frozen=True)
class FaultSpec:
    name: str
    target_task_id: str | None
    apply: Callable[[dict[str, Any], str, dict[str, Any]], tuple[dict[str, Any], str]]


def _leaf_fault(
    mode: str,
    task_id: str,
) -> Callable[[dict[str, Any], str, dict[str, Any]], tuple[dict[str, Any], str]]:
    def apply(
        ir: dict[str, Any],
        clean_code: str,
        clean_snippets: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        snippets = copy.deepcopy(clean_snippets)
        function_name = _function_name(ir, task_id)
        if not function_name or not snippets.get(task_id):
            raise ValueError(
                f"cannot inject {mode} fault: missing function/snippet for {task_id}"
            )
        if mode == "runtime":
            snippets[task_id] = inject_runtime_fault(
                str(snippets[task_id]), function_name, task_id
            )
            faulty_code = inject_runtime_fault(clean_code, function_name, task_id)
        elif mode == "wrong_answer":
            snippets[task_id] = inject_wrong_answer_fault(
                str(snippets[task_id]), function_name
            )
            faulty_code = inject_wrong_answer_fault(clean_code, function_name)
        else:  # pragma: no cover - FaultSpec construction guards this
            raise ValueError(f"unknown leaf fault mode: {mode}")
        # The mutation is applied again to the assembled program so that the
        # fault is guaranteed to survive the experiment.  A second LLM
        # assembly pass is deliberately not placed between injection and
        # execution: that would make the injected error depend on whether the
        # assembler happened to rewrite it.
        return snippets, faulty_code

    return apply


def _global_fault(
    ir: dict[str, Any],
    clean_code: str,
    clean_snippets: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    # Keep leaf snippets unchanged: the only fault is in the unmapped
    # orchestration function.
    return copy.deepcopy(clean_snippets), inject_unmapped_runtime_fault(clean_code)


def fault_specs() -> list[FaultSpec]:
    return [
        FaultSpec("runtime_in_S1_sorting", "S1", _leaf_fault("runtime", "S1")),
        FaultSpec("runtime_in_S2_greedy", "S2", _leaf_fault("runtime", "S2")),
        FaultSpec("wrong_answer_in_S2_greedy", "S2", _leaf_fault("wrong_answer", "S2")),
        FaultSpec("runtime_in_unmapped_solve", None, _global_fault),
    ]


def _build_deepseek_llm(
    temperature: float = 0.0,
    json_mode: bool = True,
    max_tokens: int | None = None,
) -> Any:
    from langchain_openai import ChatOpenAI

    model_kwargs: dict[str, Any] = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}
    raw_max = os.getenv("DEEPSEEK_MAX_TOKENS") or os.getenv("MAX_TOKENS")
    if max_tokens is None and raw_max:
        try:
            max_tokens = int(str(raw_max).strip().strip('"').strip("'"))
        except ValueError:
            max_tokens = None
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs=model_kwargs,
    )


def _patch_auditcoder_for_deepseek() -> None:
    """Patch imported ``get_llm`` bindings without changing the generic factory."""

    if not os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError(
            "DEEPSEEK_API_KEY is empty. Put it in .env or export it before live mode."
        )

    import auditcoder.common as common
    import auditcoder.common.llm as common_llm

    common_llm.get_llm = _build_deepseek_llm
    common.get_llm = _build_deepseek_llm

    # Some modules may already have been imported by a caller or test runner.
    # Replace their local binding too; modules imported after this point obtain
    # the patched function from common.llm automatically.
    for module in list(sys.modules.values()):
        if module is not None and getattr(module, "__name__", "").startswith("auditcoder"):
            if hasattr(module, "get_llm"):
                try:
                    setattr(module, "get_llm", _build_deepseek_llm)
                except Exception:
                    pass


def _validate_clean_generation(
    *,
    ir: dict[str, Any],
    code_snippets: dict[str, str],
    final_code: str,
    execution_result: dict[str, Any],
) -> None:
    missing = [
        task_id
        for task_id in ("S1", "S2")
        if not str(code_snippets.get(task_id, "")).strip()
    ]
    if missing:
        raise RuntimeError(f"DeepSeek did not return required leaf snippets: {missing}")
    for task_id in ("S1", "S2"):
        fn = _function_name(ir, task_id)
        if fn not in _function_defs(code_snippets[task_id]):
            raise RuntimeError(
                f"snippet for {task_id} does not define its contracted function {fn!r}"
            )
        if fn not in _function_defs(final_code):
            raise RuntimeError(
                f"assembled program does not preserve contracted function {fn!r}"
            )
    if not final_code.strip():
        raise RuntimeError("assembler returned an empty final program")
    if not execution_result.get("all_passed", False):
        raise RuntimeError(
            "the clean baseline failed the canonical cases; controlled-fault metrics "
            "would be uninterpretable.\n"
            + json.dumps(execution_result, ensure_ascii=False, indent=2)
        )


def run_live(output_dir: Path) -> dict[str, Any]:
    """Run the one-problem DeepSeek experiment and save machine-readable output."""

    _patch_auditcoder_for_deepseek()

    from auditcoder.algo_agents import dispatch_algo_agents
    from auditcoder.assembler_agent import build_assembler_agent_graph
    from auditcoder.final_verifier_agent import build_final_verifier_graph
    from auditcoder.final_verifier_agent.nodes.full_code_runner import full_code_runner

    ir = build_ground_truth_ir()
    code_snippets, verification_reports = dispatch_algo_agents(copy.deepcopy(ir))
    assembler_result = build_assembler_agent_graph().invoke(
        {"task_plan_ir": ir, "code_snippets": code_snippets}
    )
    clean_code = str(assembler_result.get("final_code") or "")
    clean_state = {
        "problem_id": PROBLEM_ID,
        "final_code": clean_code,
        "task_plan_ir": ir,
        "code_snippets": code_snippets,
        "verification_reports": verification_reports,
    }
    clean_execution = full_code_runner(clean_state).get("execution_result", {})
    _validate_clean_generation(
        ir=ir,
        code_snippets=code_snippets,
        final_code=clean_code,
        execution_result=clean_execution,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "baseline_clean.py").write_text(clean_code, encoding="utf-8")
    (artifacts_dir / "baseline_ir.json").write_text(
        json.dumps(ir, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifacts_dir / "baseline_snippets.json").write_text(
        json.dumps(code_snippets, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    final_graph = build_final_verifier_graph()
    case_records: list[dict[str, Any]] = []
    for spec in fault_specs():
        faulty_snippets, faulty_code = spec.apply(ir, clean_code, code_snippets)

        initial_state: dict[str, Any] = {
            "problem_id": PROBLEM_ID,
            "final_code": faulty_code,
            "task_plan_ir": copy.deepcopy(ir),
            "code_snippets": faulty_snippets,
            "verification_reports": copy.deepcopy(verification_reports),
        }
        # Keep an immutable pre-repair snapshot for frozen-complement metrics;
        # LangGraph is normally copy-on-write, but the experiment should not
        # depend on that implementation detail.
        initial_state_snapshot = copy.deepcopy(initial_state)
        final_state = final_graph.invoke(initial_state_snapshot)
        case_record = evaluate_fault_case(
            fault=spec.name,
            target_task_id=spec.target_task_id,
            initial_state=initial_state,
            final_state=final_state,
            ir=ir,
        )
        case_record["initial_execution"] = final_state.get(
            "initial_execution_result", {}
        ) or final_state.get("execution_result", {})
        case_record["final_execution"] = final_state.get("execution_result", {})
        case_record["initial_snippet_sha256"] = {
            task_id: _sha256(str(value)) for task_id, value in faulty_snippets.items()
        }
        case_record["final_snippet_sha256"] = {
            task_id: _sha256(str(value))
            for task_id, value in (final_state.get("code_snippets") or {}).items()
        }
        case_records.append(case_record)

        case_dir = artifacts_dir / spec.name
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "faulty.py").write_text(faulty_code, encoding="utf-8")
        (case_dir / "repaired_or_unchanged.py").write_text(
            str(final_state.get("final_code") or ""), encoding="utf-8"
        )
        (case_dir / "state_summary.json").write_text(
            json.dumps(
                {
                    "case": case_record,
                    "final_report": final_state.get("final_report"),
                    "final_snippet_sha256": case_record["final_snippet_sha256"],
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    result = {
        "experiment": "controlled_fault_injection",
        "problem_id": PROBLEM_ID,
        "llm_provider": "deepseek",
        "llm_model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "problem": PROBLEM_TEXT,
        "ground_truth_leaf_nodes": {
            task_id: {
                "function_name": _function_name(ir, task_id),
                "agent_type": _node_map(ir)[task_id].get("agent_type"),
            }
            for task_id in ("S1", "S2")
        },
        "baseline": {
            "execution": clean_execution,
            "functions": _function_defs(clean_code),
            "snippet_sha256": {
                task_id: _sha256(str(value)) for task_id, value in code_snippets.items()
            },
        },
        "cases": case_records,
        "summary": summarize_cases(case_records),
        "notes": [
            "Ground-truth labels are used only after the run for evaluation; they are not passed to AuditCoder.",
            "Wrong-answer cases may use the low-confidence GlobalAttribution fallback; localization is scored against the injected target.",
            "The unmapped solve fault is a negative control for false localization.",
        ],
    }
    (output_dir / "controlled_fault_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return result


def _self_test_code() -> tuple[dict[str, Any], str, dict[str, str]]:
    ir = build_ground_truth_ir()
    snippets = {
        "S1": "def sort_intervals(intervals):\n    return sorted(intervals)\n",
        "S2": (
            "def count_merged_groups(sorted_intervals):\n"
            "    groups = 1\n"
            "    current_right = sorted_intervals[0][1]\n"
            "    for left, right in sorted_intervals[1:]:\n"
            "        if left > current_right + 1:\n"
            "            groups += 1\n"
            "        current_right = max(current_right, right)\n"
            "    return groups\n"
        ),
    }
    code = (
        snippets["S1"]
        + "\n"
        + snippets["S2"]
        + "\n"
        + "def solve():\n"
        + "    import sys\n"
        + "    data = list(map(int, sys.stdin.read().split()))\n"
        + "    n = data[0]\n"
        + "    intervals = [(data[1 + 2*i], data[2 + 2*i]) for i in range(n)]\n"
        + "    print(count_merged_groups(sort_intervals(intervals)))\n"
        + "\nif __name__ == '__main__':\n    solve()\n"
    )
    return ir, code, snippets


def _run_revised_case(
    *,
    ir: dict[str, Any],
    code: str,
    snippets: dict[str, str],
    target_task_id: str | None,
    fault: str,
    expected_rule: str | None = None,
) -> dict[str, Any]:
    """Execute one injected case through the paper's deterministic rules.

    This path intentionally avoids LangGraph/LLM dependencies.  It is the
    offline counterpart of the live experiment and verifies that the revised
    Trace/Resource/Local/GlobalAttribution rules see the same owner map as
    the generated program. Heuristic attribution is reported separately from
    correct localization against the known mutation target.
    """

    from auditcoder.final_verifier_agent.audit import (
        build_owner_map,
        locate,
        rank_execution,
        select_scope,
    )

    results: list[dict[str, Any]] = []
    for index, case in enumerate(SAMPLE_CASES):
        completed = subprocess.run(
            [sys.executable, "-c", code],
            input=case["input"],
            text=True,
            capture_output=True,
            timeout=3.0,
            check=False,
        )
        passed = completed.returncode == 0 and completed.stdout.strip() == case["expected_output"]
        results.append(
            {
                "case_index": index,
                "input": case["input"],
                "expected": case["expected_output"],
                "actual": completed.stdout.strip(),
                "stderr": completed.stderr,
                "exit_code": completed.returncode,
                "timed_out": False,
                "passed": passed,
                "symptom": (
                    "Pass"
                    if passed
                    else ("RE" if completed.returncode else "WA")
                ),
            }
        )
    execution = {
        "all_passed": all(item["passed"] for item in results),
        "error_type": (
            "none"
            if all(item["passed"] for item in results)
            else ("runtime_error" if any(item["exit_code"] for item in results) else "wrong_answer")
        ),
        "results": results,
    }
    owner_map = build_owner_map(ir, snippets, code)
    location = locate(execution, ir, [], owner_map, {}, code, snippets)
    abstained = not location or location.get("status") == "abstained"
    scope = select_scope(location, ir, snippets) if not abstained else None
    localization_correct = (
        abstained if target_task_id is None
        else not abstained and (location or {}).get("task_id") == target_task_id
    )
    expected_behavior = localization_correct
    if expected_rule is not None:
        expected_behavior = (
            not abstained
            and (location or {}).get("rule") == expected_rule
            and (location or {}).get("confidence") == "Low"
        )
    return {
        "fault": fault,
        "target_task_id": target_task_id,
        "all_passed": execution["all_passed"],
        "located_task_id": (location or {}).get("task_id"),
        "rule": (location or {}).get("rule"),
        "confidence": (location or {}).get("confidence"),
        "abstained": abstained,
        "scope_region": (scope or {}).get("region", []),
        "frozen_region": (scope or {}).get("frozen_region", []),
        "rank": list(rank_execution(execution, {})),
        "localization_correct": localization_correct,
        "expected_behavior": expected_behavior,
    }


def _run_revised_audit_self_test(
    *,
    ir: dict[str, Any],
    clean_code: str,
    clean_snippets: dict[str, str],
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for spec in fault_specs():
        faulty_snippets, faulty_code = spec.apply(ir, clean_code, clean_snippets)
        # This snapshot uses a low-confidence global fallback when a wrong
        # answer has no traceback or local witness. Test that routing without
        # counting a heuristic target as correct ground-truth localization.
        expected_rule = "GlobalAttribution" if spec.name.startswith("wrong_answer") else None
        cases.append(
            _run_revised_case(
                ir=ir,
                code=faulty_code,
                snippets=faulty_snippets,
                target_task_id=spec.target_task_id,
                fault=spec.name,
                expected_rule=expected_rule,
            )
        )
    transaction = _run_revised_transaction_self_test(
        ir=ir,
        clean_code=clean_code,
        clean_snippets=clean_snippets,
    )
    return {
        "case_count": len(cases),
        "cases": cases,
        "all_passed": all(item["expected_behavior"] for item in cases),
        "targeted_localization": sum(
            item["localization_correct"]
            for item in cases
            if item["target_task_id"] is not None
        ),
        "heuristic_attribution_count": sum(item["rule"] == "GlobalAttribution" for item in cases),
        "heuristic_correct_localization_count": sum(
            item["localization_correct"] for item in cases if item["rule"] == "GlobalAttribution"
        ),
        "negative_control_abstention": all(
            item["abstained"]
            for item in cases
            if item["target_task_id"] is None
        ),
        "bounded_repair_transaction": transaction,
    }


def _run_revised_transaction_self_test(
    *,
    ir: dict[str, Any],
    clean_code: str,
    clean_snippets: dict[str, str],
) -> dict[str, Any]:
    """Exercise strict-improvement acceptance and rollback without an LLM."""

    from auditcoder.final_verifier_agent.audit import (
        frozen_complement_ok,
        repair_event,
        rank_execution,
    )

    faulty_snippets = copy.deepcopy(clean_snippets)
    faulty_snippets["S2"] = inject_wrong_answer_fault(
        faulty_snippets["S2"], "count_merged_groups"
    )
    before = {
        "final_code": clean_code,
        "code_snippets": faulty_snippets,
        "task_plan_ir": copy.deepcopy(ir),
        "verification_reports": {"S1": {"verdict": "accept"}, "S2": {"verdict": "reject"}},
        "execution_result": {
            "all_passed": False,
            "error_type": "wrong_answer",
            "results": [{"case_index": 0, "passed": False, "exit_code": 0}],
        },
    }
    candidate = {
        **before,
        "code_snippets": copy.deepcopy(clean_snippets),
        "execution_result": {
            "all_passed": True,
            "error_type": "none",
            "results": [{"case_index": 0, "passed": True, "exit_code": 0}],
        },
    }
    scope = {
        "nominal_region": ["S2"],
        "region": ["S2"],
        "frozen_region": ["S0", "S1"],
        "integration_exceptions": [],
    }
    frozen_ok, frozen_violations = frozen_complement_ok(
        before["code_snippets"], candidate["code_snippets"], scope["region"]
    )
    accepted_event = repair_event(
        location={"task_id": "S2", "rule": "Trace", "confidence": "High"},
        scope=scope,
        before_state=before,
        candidate_state=candidate,
        candidate_execution=candidate["execution_result"],
        decision="Accept",
        rollback_status="NotApplicable",
        attempt=1,
    )
    rejected_event = repair_event(
        location={"task_id": "S2", "rule": "Local", "confidence": "Low"},
        scope=scope,
        before_state=before,
        candidate_state=before,
        candidate_execution=before["execution_result"],
        decision="Reject",
        rollback_status="Restored",
        attempt=2,
    )
    return {
        "frozen_complement_ok": frozen_ok,
        "frozen_violations": frozen_violations,
        "rank_before": list(rank_execution(before["execution_result"], before["verification_reports"])),
        "rank_after": list(rank_execution(candidate["execution_result"], candidate["verification_reports"])),
        "strict_improvement": rank_execution(candidate["execution_result"], candidate["verification_reports"]) > rank_execution(before["execution_result"], before["verification_reports"]),
        "accepted_event_decision": accepted_event["decision"],
        "rollback_event_decision": rejected_event["decision"],
        "rollback_status": rejected_event["rollback_status"],
        "frozen_region": scope["frozen_region"],
    }


def run_self_test() -> dict[str, Any]:
    """Run deterministic checks without importing LangGraph or calling DeepSeek."""

    ir, clean_code, clean_snippets = _self_test_code()
    for task_id in ("S1", "S2"):
        function_name = _function_name(ir, task_id)
        mutated = inject_runtime_fault(clean_snippets[task_id], function_name, task_id)
        assert f"AUDITCODER_INJECTED_RUNTIME:{task_id}" in mutated
    wrong = inject_wrong_answer_fault(clean_snippets["S2"], "count_merged_groups")
    assert "return -999999" in wrong
    global_fault = inject_unmapped_runtime_fault(clean_code)
    assert "AUDITCODER_INJECTED_GLOBAL" in global_fault

    def fake_state(
        *,
        target: str | None,
        location: dict[str, Any] | None,
        passed: bool,
        retried: list[str] | None = None,
        final_snippets: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        initial = {
            "final_code": clean_code,
            "code_snippets": copy.deepcopy(clean_snippets),
            "execution_result": {"all_passed": False, "error_type": "runtime_error"},
        }
        final = {
            "final_code": clean_code,
            "code_snippets": final_snippets or copy.deepcopy(clean_snippets),
            "retried_tasks": retried or [],
            "execution_result": {"all_passed": passed, "error_type": "none" if passed else "runtime_error"},
            "initial_execution_result": initial["execution_result"],
            "final_report": {
                "error_location": location,
                "retry_info": {"retry_attempted": bool(retried)},
                "repair_history": (
                    [{"repair_region": {"nodes": retried or []}}] if retried else []
                ),
            },
        }
        return initial, final

    initial, final = fake_state(
        target="S1",
        location={"task_id": "S1", "method": "traceback", "confidence": "high"},
        passed=True,
        retried=["S1"],
    )
    good = evaluate_fault_case(
        fault="self_test_runtime", target_task_id="S1", initial_state=initial, final_state=final, ir=ir
    )
    assert good["localization_correct"] and good["repair_success"]
    initial, final = fake_state(
        target=None,
        location=None,
        passed=False,
    )
    negative = evaluate_fault_case(
        fault="self_test_negative", target_task_id=None, initial_state=initial, final_state=final, ir=ir
    )
    assert negative["localization_correct"] and not negative["localized"]
    summary = summarize_cases([good, negative])
    assert summary["false_localization_count"] == 0
    revised_cases = _run_revised_audit_self_test(
        ir=ir,
        clean_code=clean_code,
        clean_snippets=clean_snippets,
    )
    assert revised_cases["all_passed"] is True
    transaction = revised_cases["bounded_repair_transaction"]
    assert transaction["frozen_complement_ok"]
    assert transaction["strict_improvement"]
    assert transaction["accepted_event_decision"] == "Accept"
    assert transaction["rollback_event_decision"] == "Reject"
    result = {"status": "ok", "summary": summary, "revised_audit": revised_cases, "checks": [
        "AST runtime mutation",
        "AST wrong-answer mutation",
        "unmapped orchestration mutation",
        "ground-truth localization metrics",
        "negative-control abstention",
    ]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "controlled_fault_injection_deepseek",
        help="directory for JSON results and inspectable code artifacts",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run deterministic mutation/metric checks without API calls",
    )
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return

    try:
        result = run_live(args.output)
    except Exception as exc:
        print(f"[controlled-fault experiment failed] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Results written to: {args.output / 'controlled_fault_results.json'}")


if __name__ == "__main__":
    main()
