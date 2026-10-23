"""Offline checks of the shipped baseline imports and subprocess APPS judges."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("method", ["codesim", "mapcoder"])
@pytest.mark.parametrize("provider", ["deepseek", "qwen"])
def test_upstream_check_needs_no_credentials(method, provider, tmp_path):
    env = {
        key: value for key, value in os.environ.items()
        if not any(word in key.upper() for word in ("API", "TOKEN", "SECRET", "PYTHONPATH"))
    }
    completed = subprocess.run(
        [sys.executable, str(ROOT / "baselines" / method / f"run_apps_200_{provider}.py"), "--check"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "200 tasks" in completed.stdout
    assert "no API calls" in completed.stdout


@pytest.fixture(params=["deepseek", "qwen"])
def adapter(request):
    path = ROOT / "baselines" / "mapcoder" / f"run_apps_200_{request.param}.py"
    spec = importlib.util.spec_from_file_location(f"mapcoder_{request.param}_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_judge_distinguishes_correct_wrong_and_crashing_programs(adapter):
    judge = adapter.LocalAPPSJudge(timeout=2, max_cases=0)
    cases = [{"input": "3\n", "expected": "6", "output": ["6"]}]
    assert judge.check("print(int(input()) * 2)", cases, None, sample=False)[0]
    assert not judge.check("print(0)", cases, None, sample=False)[0]
    assert not judge.check("raise RuntimeError('boom')", cases, None, sample=False)[0]


def test_callable_judge_and_code_fence_parser(adapter):
    judge = adapter.LocalAPPSJudge(timeout=2, max_cases=0)
    cases = [{"args": [2, 3], "input": "[2, 3]", "expected": 5, "output": [5]}]
    assert judge.check("def add(a, b): return a + b", cases, "add", sample=False)[0]
    assert not judge.check("def add(a, b): return a - b", cases, "add", sample=False)[0]
    assert adapter.code_from_response("```python\nprint(42)\n```") == "print(42)"
