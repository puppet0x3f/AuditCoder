"""Regression tests for explicit file-I/O execution contracts."""

from __future__ import annotations

import textwrap

import pytest

from auditcoder.assembler_agent.agent import _code_assembler, _validate_assembled_contract
from auditcoder.common.execution_contract import (
    parse_execution_contract,
    render_execution_contract,
)
from auditcoder.final_verifier_agent.nodes.full_code_runner import full_code_runner
from auditcoder.plan_agent.nodes.ir_assembler import ir_assembler


def _file_contract(app_dir, outputs=None, inputs=None):
    contract = {
        "schema_version": "1.0",
        "io_mode": "file",
        "explicit": True,
        "app_dir": str(app_dir),
        "required_outputs": outputs or ["output/results.json"],
    }
    if inputs:
        contract["required_inputs"] = inputs
    return contract


def test_contract_render_parse_round_trip(tmp_path):
    rendered = render_execution_contract(
        _file_contract(tmp_path, inputs=["data/input.json"])
    )
    parsed = parse_execution_contract(f"before\n{rendered}\nafter")
    assert parsed["io_mode"] == "file"
    assert parsed["app_dir"] == str(tmp_path)
    assert parsed["required_outputs"] == ["output/results.json"]
    assert parsed["required_inputs"] == ["data/input.json"]
    assert parsed["stdin"] == "closed_eof"


def test_contract_rejects_output_escape(tmp_path):
    with pytest.raises(ValueError, match="unsafe required output"):
        render_execution_contract(
            _file_contract(tmp_path, outputs=["../private/result.json"])
        )


def test_ir_assembler_preserves_explicit_contract(tmp_path):
    raw = render_execution_contract(_file_contract(tmp_path))
    state = {
        "problem_raw": raw,
        "problem_structured": {
            "title": "file task",
            "description": "write a file",
            "input_format": "input.json",
            "output_format": "output/results.json",
            "constraints": {},
            "sample_cases": [],
        },
        "constraint_analysis": {
            "time_complexity_upper_bound": "O(n)",
            "space_complexity_upper_bound": "O(n)",
        },
        "algorithm_selection": [],
        "solution_strategy": {
            "global_strategy": "read, compute, write",
            "root_id": "S0",
            "execution_order": ["S1"],
        },
        "task_tree": [
            {
                "id": "S0",
                "is_leaf": False,
                "parent": None,
            },
            {
                "id": "S1",
                "is_leaf": True,
                "parent": "S0",
            },
        ],
        "interface_definitions": [],
        "test_cases": {
            "sample_test_cases": [],
            "edge_cases": [],
            "invariants": [],
        },
    }
    result = ir_assembler(state)["task_plan_ir"]
    assert result["execution_contract"]["io_mode"] == "file"
    assert result["execution_contract"]["required_outputs"] == [
        "output/results.json"
    ]
    assert raw in result["problem"]["original_specification"]
    assert result["problem"]["description"] == raw


def test_assembler_static_gate_rejects_stdin_and_missing_outputs(tmp_path):
    code = textwrap.dedent(
        """\
        import sys
        def solve():
            return input()
        if __name__ == "__main__":
            print(solve())
        """
    )
    violations = _validate_assembled_contract(
        code, _file_contract(tmp_path)
    )
    assert "file mode must not call input()" in violations
    assert "required output is not referenced: output/results.json" in violations
    assert "file mode code does not reference the contract app_dir" in violations


def test_assembler_static_gate_accepts_file_entrypoint(tmp_path):
    output_path = tmp_path / "output/results.json"
    code = textwrap.dedent(
        f"""\
        import json
        from pathlib import Path
        APP = Path({str(tmp_path)!r})
        def solve():
            path = APP / "output/results.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({{"ok": True}}))
        if __name__ == "__main__":
            solve()
        """
    )
    assert output_path.name in code
    assert _validate_assembled_contract(
        code, _file_contract(tmp_path)
    ) == []


def test_assembler_static_gate_requires_complete_input_path(tmp_path):
    code = textwrap.dedent(
        f"""\
        from pathlib import Path
        APP = Path({str(tmp_path)!r})
        def solve():
            data = (APP / "returns_raw.csv").read_text()
            path = APP / "output/results.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data)
        if __name__ == "__main__":
            solve()
        """
    )
    violations = _validate_assembled_contract(
        code,
        _file_contract(
            tmp_path,
            inputs=["data/returns_raw.csv"],
        ),
    )
    assert "required input path is not referenced: data/returns_raw.csv" in violations


def test_assembler_blocks_missing_verified_leaf_code():
    result = _code_assembler(
        {
            "task_plan_ir": {
                "problem": {},
                "task_nodes": [
                    {
                        "id": "S1",
                        "is_leaf": True,
                        "agent_type": "math",
                    }
                ],
                "execution_order": ["S1"],
            },
            "code_snippets": {},
        }
    )
    assert result["final_code"] == ""
    assert "S1" in result["assembly_contract_violations"][0]


def test_file_runner_ignores_generated_stdin_samples_and_checks_outputs(tmp_path):
    output_path = tmp_path / "output/results.json"
    code = textwrap.dedent(
        f"""\
        import json
        from pathlib import Path
        path = Path({str(output_path)!r})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({{"ok": True}}))
        """
    )
    state = {
        "final_code": code,
        "task_plan_ir": {
            "execution_contract": _file_contract(tmp_path),
            "problem": {"constraints": {"time_limit": "2s"}},
            "validation": {
                "sample_test_cases": [
                    {"input": "hallucinated stdin", "expected_output": "wrong"}
                ]
            },
        },
    }
    execution = full_code_runner(state)["execution_result"]
    assert execution["all_passed"] is True
    assert execution["test_source"] == "execution_contract"
    assert execution["contract_checks"] is True
    assert execution["results"][0]["path"] == "output/results.json"


def test_file_runner_removes_stale_output_before_execution(tmp_path):
    output_path = tmp_path / "output/results.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_text('{"stale": true}')
    state = {
        "final_code": "pass\n",
        "task_plan_ir": {
            "execution_contract": _file_contract(tmp_path),
            "problem": {"constraints": {}},
            "validation": {"sample_test_cases": []},
        },
    }
    execution = full_code_runner(state)["execution_result"]
    assert execution["all_passed"] is False
    assert execution["error_type"] == "contract_error"
    assert not output_path.exists()


def test_file_runner_rejects_missing_required_input(tmp_path):
    execution = full_code_runner(
        {
            "final_code": "pass\n",
            "task_plan_ir": {
                "execution_contract": _file_contract(
                    tmp_path,
                    inputs=["data/input.json"],
                ),
                "problem": {"constraints": {}},
                "validation": {"sample_test_cases": []},
            },
        }
    )["execution_result"]
    assert execution["all_passed"] is False
    assert execution["error_type"] == "contract_error"
    assert "data/input.json" in execution["stderr"]


def test_legacy_stdin_runner_remains_default():
    state = {
        "problem_id": None,
        "final_code": "value = int(input())\nprint(value * 2)\n",
        "task_plan_ir": {
            "problem": {"constraints": {"time_limit": "2s"}},
            "validation": {
                "sample_test_cases": [
                    {"input": "3\n", "expected_output": "6"}
                ]
            },
        },
    }
    execution = full_code_runner(state)["execution_result"]
    assert execution["all_passed"] is True
    assert execution["test_source"] == "task_plan_ir.validation.sample_test_cases"
