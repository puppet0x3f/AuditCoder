"""Tests for the Verifier Agent."""

from __future__ import annotations

import ast
import math
import textwrap

import pytest

from auditcoder.schemas.verification import (
    CodeSnippet,
    Counterexample,
    MediumSignals,
    StrongSignals,
    VerificationReport,
    WeakSignals,
)
from auditcoder.verifier_agent.graph import build_verifier_agent_graph
from auditcoder.verifier_agent.nodes.contract_verifier import contract_verifier
from auditcoder.verifier_agent.nodes.compile_verifier import compile_verifier
from auditcoder.verifier_agent.nodes.complexity_verifier import _static_estimate, _fit_power_law
from auditcoder.verifier_agent.nodes.bruteforce_verifier import _build_comparison_code
from auditcoder.verifier_agent.nodes.sample_test_runner import sample_test_runner
from auditcoder.verifier_agent.nodes.verdict_assembler import verdict_assembler
from auditcoder.verifier_agent.sandbox import SandboxExecutor


# ── Graph compilation ──────────────────────────────────────────

def test_graph_compiles():
    app = build_verifier_agent_graph()
    assert app is not None


# ── Schema validation ──────────────────────────────────────────

def test_verification_report_schema():
    report = VerificationReport(
        task_id="task_dp_1",
        verdict="accept",
        strong_signals=StrongSignals(
            syntax_ok=True,
            compile_ok=True,
            sample_tests_passed=True,
            bruteforce_match=True,
            resource_ok=True,
        ),
        medium_signals=MediumSignals(
            complexity_static_ok=True,
            complexity_dynamic_ok=True,
            estimated_complexity="O(n^2)",
            contract_ok=True,
        ),
        weak_signals=WeakSignals(
            readability_score=4,
            style_score=4,
            explanation_consistent=True,
        ),
        summary="所有检查通过",
    )
    assert report.verdict == "accept"
    assert report.counterexample is None


def test_code_snippet_schema():
    snippet = CodeSnippet(
        task_id="task_1",
        agent_type="dp",
        source_code="def solve(n): return n",
        explanation="直接返回 n",
    )
    assert snippet.agent_type == "dp"


def test_counterexample_schema():
    ce = Counterexample(
        minimal_input="3\n1 2 3",
        expected_output="6",
        actual_output="5",
        error_category="wrong_answer",
        suspected_reason="求和逻辑错误",
    )
    assert ce.error_category == "wrong_answer"


# ── Sandbox executor ───────────────────────────────────────────

def test_sandbox_executor_basic():
    executor = SandboxExecutor()
    result = executor.run("print('hello')")
    assert result.exit_code == 0
    assert "hello" in result.stdout


def test_sandbox_executor_stdin():
    executor = SandboxExecutor()
    result = executor.run("x = input(); print(f'got {x}')", stdin="world\n")
    assert result.exit_code == 0
    assert "got world" in result.stdout


def test_sandbox_executor_timeout():
    executor = SandboxExecutor()
    result = executor.run("import time; time.sleep(100)", timeout_s=1.0)
    assert result.timed_out


def test_sandbox_executor_syntax_error():
    executor = SandboxExecutor()
    result = executor.run("def f(\n")
    assert result.exit_code != 0


def test_sandbox_function_call():
    executor = SandboxExecutor()
    result = executor.run_function_call(
        function_code="def add(a, b): return a + b",
        function_name="add",
        args_code="3, 4",
    )
    assert result.exit_code == 0
    assert "7" in result.stdout


# ── Contract verifier ──────────────────────────────────────────

def test_contract_verifier_pass():
    state = {
        "code_snippet": {
            "source_code": textwrap.dedent("""\
                def solve(nums: list[int], target: int) -> int:
                    return 0
            """),
        },
        "task_contract": {
            "interface": {
                "function_name": "solve",
                "params": [
                    {"name": "nums", "type": "list[int]"},
                    {"name": "target", "type": "int"},
                ],
                "return_type": "int",
            }
        },
    }
    result = contract_verifier(state)
    assert result["contract_report"]["contract_ok"] is True
    assert result["contract_report"]["violations"] == []


def test_contract_verifier_missing_function():
    state = {
        "code_snippet": {"source_code": "def other(): pass"},
        "task_contract": {
            "interface": {"function_name": "solve", "params": [], "return_type": "int"}
        },
    }
    result = contract_verifier(state)
    assert result["contract_report"]["contract_ok"] is False
    assert any("solve" in v for v in result["contract_report"]["violations"])


def test_contract_verifier_extra_param():
    state = {
        "code_snippet": {
            "source_code": "def solve(a: int, b: int, c: int) -> int: return 0",
        },
        "task_contract": {
            "interface": {
                "function_name": "solve",
                "params": [
                    {"name": "a", "type": "int"},
                    {"name": "b", "type": "int"},
                ],
                "return_type": "int",
            }
        },
    }
    result = contract_verifier(state)
    assert result["contract_report"]["contract_ok"] is False
    assert any("c" in v for v in result["contract_report"]["violations"])


def test_contract_verifier_global_usage():
    state = {
        "code_snippet": {
            "source_code": textwrap.dedent("""\
                x = 0
                def solve() -> int:
                    global x
                    x += 1
                    return x
            """),
        },
        "task_contract": {
            "interface": {"function_name": "solve", "params": [], "return_type": "int"}
        },
    }
    result = contract_verifier(state)
    assert result["contract_report"]["contract_ok"] is False
    assert any("global" in v for v in result["contract_report"]["violations"])


# ── Compile verifier ───────────────────────────────────────────

def test_compile_verifier_valid():
    state = {
        "code_snippet": {"source_code": "def solve(): return 42"},
    }
    result = compile_verifier(state)
    assert result["compile_report"]["syntax_ok"] is True
    assert result["compile_report"]["compile_ok"] is True


def test_compile_verifier_syntax_error():
    state = {
        "code_snippet": {"source_code": "def solve(\n"},
    }
    result = compile_verifier(state)
    assert result["compile_report"]["syntax_ok"] is False
    assert result["compile_report"]["compile_ok"] is False


def test_sample_runner_filters_extra_global_arguments():
    state = {
        "code_snippet": {
            "source_code": "def add(a: int, b: int) -> int: return a + b",
        },
        "task_contract": {
            "interface": {
                "function_name": "add",
                "params": [{"name": "a"}, {"name": "b"}],
            }
        },
        "compile_report": {"compile_ok": True},
        "validation_info": {
            "scope": "single_leaf_global",
            "sample_test_cases": [
                {
                    "input": {"a": 2, "b": 3, "unused": 99},
                    "expected_output": "5",
                }
            ],
        },
        "problem_info": {"constraints": {}},
    }
    report = sample_test_runner(state)["sample_test_report"]
    assert report["passed"] is True
    assert report["results"][0]["actual"] == "5"


def test_bruteforce_comparison_tolerates_numeric_roundoff():
    code = _build_comparison_code(
        candidate_code="def solve(x): return {'value': x + 1e-15}",
        bruteforce_code="def solve(x): return {'value': x}",
        input_generator_code="def generate_test_input(): return {'x': 1.0}",
        function_name="solve",
        num_tests=3,
    )
    result = SandboxExecutor().run(code)
    assert result.exit_code == 0
    payload = __import__("json").loads(result.stdout)
    assert payload["mismatches"] == []


# ── Complexity static analysis ─────────────────────────────────

def test_static_estimate_linear():
    code = "def f(n):\n    for i in range(n):\n        pass"
    result = _static_estimate(code)
    assert result["loop_depth"] == 1
    assert "n" in result["estimated_complexity"]


def test_static_estimate_quadratic():
    code = textwrap.dedent("""\
        def f(n):
            for i in range(n):
                for j in range(n):
                    pass
    """)
    result = _static_estimate(code)
    assert result["loop_depth"] == 2
    assert "n^2" in result["estimated_complexity"]


def test_static_estimate_memoized():
    code = textwrap.dedent("""\
        from functools import lru_cache
        @lru_cache(maxsize=None)
        def f(n):
            if n <= 1: return n
            return f(n-1) + f(n-2)
    """)
    result = _static_estimate(code)
    assert result["has_memoization"] is True


def test_static_estimate_sorting():
    code = "def f(arr):\n    return sorted(arr)"
    result = _static_estimate(code)
    assert result["has_sorting"] is True


# ── Complexity dynamic fitting ─────────────────────────────────

def test_fit_power_law_linear():
    sizes = [100, 200, 400, 800]
    times = [s * 1e-6 for s in sizes]
    _, k = _fit_power_law(sizes, times)
    assert abs(k - 1.0) < 0.1


def test_fit_power_law_quadratic():
    sizes = [100, 200, 400, 800]
    times = [s**2 * 1e-9 for s in sizes]
    _, k = _fit_power_law(sizes, times)
    assert abs(k - 2.0) < 0.1


# ── Verdict assembler ─────────────────────────────────────────

def test_verdict_accept():
    state = {
        "code_snippet": {"task_id": "t1"},
        "compile_report": {"syntax_ok": True, "compile_ok": True},
        "contract_report": {"contract_ok": True, "violations": []},
        "sample_test_report": {"passed": True, "resource_ok": True, "max_time_ms": 10},
        "bruteforce_report": {"match": True, "skipped": False},
        "complexity_report": {"static_ok": True, "dynamic_ok": True, "estimated_complexity": "O(n)"},
        "counterexample": None,
        "weak_signal_report": {"readability_score": 4, "style_score": 4, "explanation_consistent": True},
    }
    result = verdict_assembler(state)
    assert result["verification_report"]["verdict"] == "accept"


def test_verdict_reject_on_compile_fail():
    state = {
        "code_snippet": {"task_id": "t1"},
        "compile_report": {"syntax_ok": False, "compile_ok": False},
        "contract_report": {"contract_ok": True, "violations": []},
        "sample_test_report": {"passed": False, "resource_ok": True},
        "bruteforce_report": {"match": False, "skipped": True},
        "complexity_report": {"static_ok": False, "dynamic_ok": False, "estimated_complexity": "unknown"},
        "counterexample": None,
        "weak_signal_report": {},
    }
    result = verdict_assembler(state)
    assert result["verification_report"]["verdict"] == "reject"


def test_verdict_warning_on_medium_fail():
    state = {
        "code_snippet": {"task_id": "t1"},
        "compile_report": {"syntax_ok": True, "compile_ok": True},
        "contract_report": {"contract_ok": False, "violations": ["缺少参数: n"]},
        "sample_test_report": {"passed": True, "resource_ok": True, "max_time_ms": 10},
        "bruteforce_report": {"match": True, "skipped": False},
        "complexity_report": {"static_ok": True, "dynamic_ok": True, "estimated_complexity": "O(n)"},
        "counterexample": None,
        "weak_signal_report": {},
    }
    result = verdict_assembler(state)
    assert result["verification_report"]["verdict"] == "warning"
