"""Node: complexity_verifier — 复杂度预算检查。

第一层：静态启发式（AST 分析）
第二层：动态拟合（多规模运行 + 最小二乘法）
"""

from __future__ import annotations

import ast
import json
import math
import re

import numpy as np

from auditcoder.verifier_agent.sandbox import SandboxExecutor
from auditcoder.verifier_agent.state import VerifierState

FITTING_SIZES = [100, 200, 400, 800, 1600, 3200]
RUNS_PER_SIZE = 3


# ── Layer 1: Static heuristic ──────────────────────────────────

def _max_loop_depth(tree: ast.Module) -> int:
    """Return the maximum nested loop depth in the code."""

    def _depth(node: ast.AST, current: int) -> int:
        best = current
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While)):
                best = max(best, _depth(child, current + 1))
            else:
                best = max(best, _depth(child, current))
        return best

    return _depth(tree, 0)


def _has_memoization(tree: ast.Module) -> bool:
    """Check for @lru_cache, @cache, or functools.cache decorators."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                dec_str = ast.unparse(dec)
                if any(kw in dec_str for kw in ("lru_cache", "cache", "memoize")):
                    return True
    return False


def _has_sorting(source: str) -> bool:
    return bool(re.search(r"\b(sorted|\.sort)\s*\(", source))


def _has_heap(source: str) -> bool:
    return "heapq" in source or "heappush" in source or "heappop" in source


def _static_estimate(source_code: str) -> dict:
    tree = ast.parse(source_code)
    loop_depth = _max_loop_depth(tree)
    memo = _has_memoization(tree)
    sorting = _has_sorting(source_code)
    heap = _has_heap(source_code)

    if memo:
        estimated = "O(states × transition)"
    elif sorting and loop_depth <= 1:
        estimated = "O(n log n)"
    elif heap and loop_depth <= 1:
        estimated = "O(n log n)"
    elif loop_depth == 0:
        estimated = "O(n)"
    elif loop_depth == 1:
        estimated = "O(n)"
    elif loop_depth == 2:
        estimated = "O(n^2)"
    elif loop_depth == 3:
        estimated = "O(n^3)"
    else:
        estimated = f"O(n^{loop_depth})"

    return {
        "loop_depth": loop_depth,
        "has_memoization": memo,
        "has_sorting": sorting,
        "has_heap": heap,
        "estimated_complexity": estimated,
    }


# ── Layer 2: Dynamic fitting ───────────────────────────────────

def _build_timing_code(
    source_code: str,
    function_name: str,
    input_gen_code: str,
    sizes: list[int],
    runs: int,
) -> str:
    """Build code that times function execution at multiple scales."""
    return f"""\
import time, json

{source_code}

{input_gen_code}

_sizes = {sizes}
_runs = {runs}
_results = []

for _n in _sizes:
    _times = []
    for _ in range(_runs):
        _args = generate_scaled_input(_n)
        _start = time.perf_counter()
        try:
            {function_name}(**_args)
        except Exception:
            _times.append(-1)
            continue
        _times.append(time.perf_counter() - _start)

    _valid = [t for t in _times if t >= 0]
    _avg = sum(_valid) / len(_valid) if _valid else -1
    _results.append({{"n": _n, "avg_time_s": _avg}})

print(json.dumps(_results))
"""


def _fit_power_law(sizes: list[int], times: list[float]) -> tuple[float, float]:
    """Fit T(n) = a * n^k using least squares in log space. Returns (a, k)."""
    valid = [(n, t) for n, t in zip(sizes, times) if t > 0]
    if len(valid) < 2:
        return 0.0, 0.0

    log_n = np.array([math.log(n) for n, _ in valid])
    log_t = np.array([math.log(t) for _, t in valid])

    k, log_a = np.polyfit(log_n, log_t, 1)
    return math.exp(log_a), k


_COMPLEXITY_BUDGETS = {
    "O(1)": 0.0,
    "O(log n)": 0.5,
    "O(n)": 1.0,
    "O(n log n)": 1.15,
    "O(n^2)": 2.0,
    "O(n^3)": 3.0,
    "O(n^4)": 4.0,
    "O(2^n)": float("inf"),
}


def _parse_expected_k(target: str) -> float:
    """Parse the expected exponent from a complexity string."""
    target = target.strip()
    if target in _COMPLEXITY_BUDGETS:
        return _COMPLEXITY_BUDGETS[target]

    m = re.match(r"O\(n\^(\d+(?:\.\d+)?)\)", target)
    if m:
        return float(m.group(1))
    if "n log n" in target:
        return 1.15
    if "log n" in target:
        return 0.5
    return float("inf")


def complexity_verifier(state: VerifierState) -> dict:
    compile_report = state.get("compile_report", {})
    if not compile_report.get("compile_ok", False):
        return {
            "complexity_report": {
                "static_ok": False,
                "dynamic_ok": False,
                "estimated_complexity": "unknown",
                "skipped": True,
            }
        }

    snippet = state["code_snippet"]
    source_code = snippet["source_code"]
    contract = state.get("task_contract", {})
    complexity_target = state.get("complexity_target", {})
    expected_time = (
        complexity_target.get("time")
        or complexity_target.get("budget")
        or contract.get("complexity_expectation", "")
    )

    static_result = _static_estimate(source_code)

    static_ok = True
    if expected_time:
        expected_k = _parse_expected_k(expected_time)
        static_est = static_result["estimated_complexity"]
        static_k = _parse_expected_k(static_est)
        if static_k > expected_k + 0.5:
            static_ok = False

    # Dynamic fitting requires a scaled input generator from bruteforce step
    bruteforce_report = state.get("bruteforce_report", {})
    input_gen_code = bruteforce_report.get("input_generator_code", "")
    interface = contract.get("interface", {})
    function_name = interface.get("function_name", "")

    dynamic_ok = True
    dynamic_result: dict = {}

    if input_gen_code and function_name and not bruteforce_report.get("skipped"):
        scaled_gen = input_gen_code.replace(
            "generate_test_input", "generate_scaled_input"
        )
        if "generate_scaled_input" not in scaled_gen:
            scaled_gen = (
                "def generate_scaled_input(n):\n"
                "    import random\n"
                "    return {'nums': [random.randint(0, n) for _ in range(n)]}\n"
            )

        timing_code = _build_timing_code(
            source_code, function_name, scaled_gen, FITTING_SIZES, RUNS_PER_SIZE
        )
        executor = SandboxExecutor(default_timeout_s=60.0)
        result = executor.run(timing_code)

        if result.exit_code == 0 and not result.timed_out:
            try:
                timing_data = json.loads(result.stdout.strip())
                sizes = [d["n"] for d in timing_data]
                times = [d["avg_time_s"] for d in timing_data]
                a, k = _fit_power_law(sizes, times)

                dynamic_result = {
                    "timing_data": timing_data,
                    "fitted_a": a,
                    "fitted_k": round(k, 2),
                }

                if expected_time:
                    expected_k = _parse_expected_k(expected_time)
                    if k > expected_k + 0.5:
                        dynamic_ok = False
            except (json.JSONDecodeError, KeyError, ValueError):
                dynamic_result = {"error": "拟合数据解析失败"}
        else:
            dynamic_result = {"error": f"计时脚本执行失败: {result.stderr[:300]}"}
            # A broken verifier-generated timing harness is absence of
            # evidence, not evidence that the candidate violates its budget.
            dynamic_result["skipped"] = True

    return {
        "complexity_report": {
            "static_ok": static_ok,
            "dynamic_ok": dynamic_ok,
            "estimated_complexity": static_result["estimated_complexity"],
            "static_details": static_result,
            "dynamic_details": dynamic_result,
            "skipped": False,
        }
    }
