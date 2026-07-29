"""Deterministic tests for the audit metrics defined in aaai/main.pdf.

These tests do not call an LLM or a remote API.  They separate:

- paper-formula correctness (DCTC, LocCov/LRS, RRS, CCR, HRSR, regression);
- ground-truth localization checks for controlled fault injection;
- conservative locator behavior under ambiguous evidence;
- retry-boundary and post-repair validation invariants.
"""

from __future__ import annotations

import copy
import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from auditcoder.common.output_judging import judge_output
from auditcoder.final_verifier_agent.graph import build_final_verifier_graph
from auditcoder.final_verifier_agent.nodes.error_locator import (
    _build_interface_map,
    _locate_by_traceback,
    _locate_by_verification_reports,
    error_locator,
)
from auditcoder.final_verifier_agent.nodes.full_code_runner import post_repair_code_runner
from auditcoder.final_verifier_agent.nodes.final_report_assembler import (
    final_report_assembler,
)
from auditcoder.final_verifier_agent.nodes.task_replanner import task_replanner
from tests.run_apps_audit_metrics import (
    _canonical_sample_cases,
    _candidate_validation_rank,
    _changed_final_line_count,
    _install_canonical_validation,
    _line_change_count,
    _loc,
    _repair_acceptance_record,
    _select_failure_evidence,
    _supplement_problem_text,
    _test_regression_rate,
    build_repair_record,
    compute_dctc,
    compute_localization_ground_truth_metrics,
    compute_repair_metrics,
)


def _leaf(
    task_id: str,
    function_name: str,
    *,
    parent: str,
    agent_type: str | None = "math",
) -> dict:
    return {
        "id": task_id,
        "goal": f"implement {function_name}",
        "type": "math",
        "is_leaf": True,
        "agent_type": agent_type,
        "parent": parent,
        "dependencies": [],
        "inputs": [{"name": "x", "type": "int", "source": None}],
        "outputs": [{"name": "result", "type": "int", "source": None}],
        "interface": {
            "function_name": function_name,
            "params": [{"name": "x", "type": "int"}],
            "return_type": "int",
            "preconditions": ["x is an int"],
            "postconditions": ["result is an int"],
        },
        "hints": [],
        "complexity_budget": "O(1)",
        "context": "",
    }


def _audit_ir() -> dict:
    return {
        "root_id": "S0",
        "task_nodes": [
            {
                "id": "S0",
                "goal": "orchestrate",
                "type": "orchestration",
                "is_leaf": False,
                "parent": None,
                "dependencies": ["S1", "S2"],
            },
            {
                "id": "S1",
                "goal": "first branch",
                "type": "orchestration",
                "is_leaf": False,
                "parent": "S0",
                "dependencies": ["S1_1"],
            },
            _leaf("S1_1", "first", parent="S1"),
            _leaf("S2", "second", parent="S0"),
        ],
    }


def _repair_inputs() -> dict:
    ir = _audit_ir()
    initial_snippets = {
        "S1_1": "def first(x):\n    return x - 1\n",
        "S2": "def second(x):\n    return x + 2\n",
    }
    final_snippets = {
        "S1_1": "def first(x):\n    y = x + 1\n    return y\n",
        "S2": initial_snippets["S2"],
    }
    return {
        "initial_ir": ir,
        "final_ir": copy.deepcopy(ir),
        "initial_snippets": initial_snippets,
        "final_snippets": final_snippets,
        "initial_reports": {
            "S1_1": {"verdict": "reject"},
            "S2": {"verdict": "accept"},
        },
        "final_reports": {
            "S1_1": {"verdict": "accept"},
            "S2": {"verdict": "accept"},
        },
        "initial_code": "\n".join(initial_snippets.values()),
        "final_code": "\n".join(final_snippets.values()),
        "initial_execution": {"all_passed": False},
        "final_execution": {"all_passed": True},
        "error_location": {
            "task_id": "S1_1",
            "method": "verification_report",
        },
        "repair_region_nodes": {"S1", "S1_1"},
        "initial_apps_tests": {
            "all_passed": False,
            "cases": [
                {"index": 0, "passed": True},
                {"index": 1, "passed": False},
            ],
        },
        "final_apps_tests": {
            "all_passed": True,
            "cases": [
                {"index": 0, "passed": True},
                {"index": 1, "passed": True},
            ],
        },
    }


def test_dctc_leaf_requires_nonempty_code_contract_provenance_and_evidence():
    ir = _audit_ir()
    result = compute_dctc(
        ir,
        {
            "S1_1": "def first(x):\n    return x\n",
            "S2": "",
        },
        {
            "S1_1": {"verdict": "accept"},
            "S2": {},
        },
        "def first(x):\n    return x\n",
    )

    assert result["dctc_leaf"] == 0.5
    assert result["metric_semantics"] == "responsibility_record_completeness"
    assert result["uncovered_reasons"]["S2"] == ["code", "evidence"]


def test_dctc_reports_agent_and_unique_mapping_as_separate_diagnostics():
    ir = _audit_ir()
    leaf_s2 = next(n for n in ir["task_nodes"] if n["id"] == "S2")
    leaf_s2["agent_type"] = None
    leaf_s2["interface"]["function_name"] = "first"
    snippets = {
        "S1_1": "def first(x):\n    return x\n",
        "S2": "def first(x):\n    return x + 1\n",
    }
    reports = {
        "S1_1": {"verdict": "accept"},
        "S2": {"verdict": "accept"},
    }

    result = compute_dctc(ir, snippets, reports, snippets["S1_1"])

    # The paper formula is record completeness and does not contain agent_type
    # or ownership correctness; those weaknesses must remain visible separately.
    assert result["dctc_leaf"] == 1.0
    assert result["component_rates"]["responsible_agent"] == 0.5
    assert result["unique_mapping_rate"] == 0.0
    assert result["duplicate_function_owners"] == {
        "first": ["S1_1", "S2"]
    }
    assert result["dctc_final_func"] == 0.0


def test_dctc_separates_record_completeness_from_assembly_preservation():
    ir = _audit_ir()
    snippets = {
        "S1_1": "def first(x):\n    return x + 1\n",
        "S2": "def second(x):\n    return x + 2\n",
    }
    reports = {
        "S1_1": {"verdict": "accept"},
        "S2": {"verdict": "accept"},
    }
    final_code = (
        "def first(x):\n"
        "    return x - 1\n\n"
        "def second(x):\n"
        "    return x + 2\n"
    )

    result = compute_dctc(ir, snippets, reports, final_code)

    assert result["dctc_leaf"] == 1.0
    assert result["assembly_preservation_rate"] == 0.5
    assert result["assembly_preservation_failures"] == {
        "S1_1": "function_ast_changed"
    }


def test_repair_metrics_follow_pdf_denominators():
    inputs = _repair_inputs()
    metrics = compute_repair_metrics(**inputs)

    # PDF: RRS_nodes = |R| / |V|.  R contains S1 and S1_1; V has four nodes.
    assert metrics["rrs_node"] == 0.5
    # Implementation-node locality remains a useful separate diagnostic.
    assert metrics["rrs_implementation_node"] == 0.5
    assert metrics["rrs_loc"] == 0.5

    changed = _changed_final_line_count(
        inputs["initial_code"], inputs["final_code"]
    )
    assert metrics["changed_loc"] == changed
    assert metrics["ccr_loc"] == pytest.approx(
        changed / _loc(inputs["final_code"])
    )
    assert metrics["hrsr_loc"] == pytest.approx(
        1
        - (
            _line_change_count(inputs["initial_code"], inputs["final_code"])
            / _loc(inputs["final_code"])
        )
    )
    assert metrics["hrsr_is_ccr_complement"] is True
    assert metrics["localized_repair_success_apps"] == 1
    assert metrics["external_repair_recovery"] == 1
    assert metrics["external_pass_preservation"] is None


def test_ccr_counts_changed_final_lines_while_hrsr_counts_deleted_churn():
    inputs = _repair_inputs()
    inputs["initial_code"] = (
        "def obsolete():\n"
        "    a = 1\n"
        "    b = 2\n"
        "    c = 3\n"
        "    return a + b + c\n\n"
        + inputs["initial_code"]
    )

    metrics = compute_repair_metrics(**inputs)

    assert 0 <= metrics["ccr_loc"] <= 1
    assert metrics["diff_churn_loc"] > metrics["changed_loc"]
    assert metrics["hrsr_is_ccr_complement"] is False


def test_unit_regression_is_unavailable_without_post_repair_reverification():
    inputs = _repair_inputs()
    inputs["final_reports"]["S2"] = {"verdict": "reject"}

    not_reverified = compute_repair_metrics(**inputs)
    assert not_reverified["rr_unit"] is None
    assert not_reverified["rr_unit_evaluation_coverage"] == 0.0

    reverified = compute_repair_metrics(
        **inputs,
        reverified_node_ids={"S2"},
    )
    assert reverified["rr_unit"] == 1.0
    assert reverified["rr_unit_evaluation_coverage"] == 1.0


def test_test_regression_counts_missing_previously_passing_case_as_regression():
    rate = _test_regression_rate(
        {
            "cases": [
                {"index": 0, "passed": True},
                {"index": 1, "passed": True},
                {"index": 2, "passed": False},
            ]
        },
        {"cases": [{"index": 0, "passed": True}]},
    )
    assert rate == 0.5


def test_controlled_fault_localization_metrics_cover_hit_boundary_and_abstention():
    hit = compute_localization_ground_truth_metrics(
        error_location={"task_id": "S1_1"},
        repair_region_nodes={"S1", "S1_1"},
        ground_truth_nodes={"S1_1"},
    )
    assert hit == {
        "localization_at_1": 1,
        "abstention_correct": None,
        "repair_boundary_precision": 0.5,
        "repair_boundary_recall": 1.0,
        "repair_boundary_iou": 0.5,
    }

    abstained = compute_localization_ground_truth_metrics(
        error_location=None,
        repair_region_nodes=set(),
        ground_truth_nodes={"S1_1"},
        expect_abstention=True,
    )
    assert abstained["abstention_correct"] is True
    assert abstained["localization_at_1"] is None


def test_repair_record_preserves_hashes_region_and_frozen_complement():
    inputs = _repair_inputs()
    record = build_repair_record(
        initial_ir=inputs["initial_ir"],
        initial_snippets=inputs["initial_snippets"],
        final_snippets=inputs["final_snippets"],
        initial_execution=inputs["initial_execution"],
        final_execution=inputs["final_execution"],
        initial_apps_tests=inputs["initial_apps_tests"],
        final_apps_tests=inputs["final_apps_tests"],
        localization_record={
            "status": "localized",
            **inputs["error_location"],
        },
        error_branch_root="S1",
        repair_region_nodes=inputs["repair_region_nodes"],
    )

    assert record is not None
    assert record["repair_region"] == {
        "region_type": "branch",
        "error_branch_root": "S1",
        "nodes": ["S1", "S1_1"],
    }
    assert record["frozen_nodes"] == ["S0", "S2"]
    assert [r["node_id"] for r in record["changed_code_units"]] == ["S1_1"]
    assert [r["node_id"] for r in record["reused_code_units"]] == ["S2"]
    assert record["post_repair_apps_passed"] is True


def test_traceback_locator_abstains_on_duplicate_function_owners():
    ir = _audit_ir()
    next(n for n in ir["task_nodes"] if n["id"] == "S2")[
        "interface"
    ]["function_name"] = "first"
    interface_map = _build_interface_map(ir)

    assert len(interface_map["first"]) == 2
    assert (
        _locate_by_traceback(
            'Traceback (most recent call last):\n  File "x.py", line 1, in first\n',
            interface_map,
        )
        is None
    )


def test_report_locator_abstains_when_multiple_nodes_are_rejected():
    assert (
        _locate_by_verification_reports(
            {
                "S1_1": {"verdict": "reject"},
                "S2": {"verdict": "reject"},
            }
        )
        is None
    )


def test_error_locator_emits_machine_readable_abstention_record():
    ir = _audit_ir()
    next(n for n in ir["task_nodes"] if n["id"] == "S2")[
        "interface"
    ]["function_name"] = "first"
    result = error_locator(
        {
            "task_plan_ir": ir,
            "code_snippets": {},
            "verification_reports": {},
            "execution_result": {
                "all_passed": False,
                "error_type": "runtime_error",
                "stderr": 'File "x.py", line 1, in first\n',
                "results": [],
            },
        }
    )

    assert result["error_location"] is None
    assert result["localization_record"]["status"] == "abstained"
    assert result["localization_record"]["ambiguous_function_names"] == ["first"]


def test_task_replanner_programmatically_preserves_responsibility_boundary():
    original = _leaf("S1_1", "first", parent="S1")
    ir = {
        "problem": {"title": "demo", "description": "demo"},
        "task_nodes": [original],
    }
    malicious_proposal = {
        **original,
        "id": "S9",
        "goal": "rewrite the whole program",
        "parent": None,
        "dependencies": ["S8"],
        "inputs": [],
        "outputs": [],
        "interface": {
            "function_name": "renamed",
            "params": [],
            "return_type": "str",
        },
        "hints": ["handle x == 0 before the general case"],
        "context": "The supplied failing case exposes the zero boundary.",
        "complexity_budget": "O(1)",
    }
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content=json.dumps(malicious_proposal)
    )

    with patch(
        "auditcoder.final_verifier_agent.nodes.task_replanner.get_llm",
        return_value=mock_llm,
    ):
        result = task_replanner(
            {
                "task_plan_ir": ir,
                "error_location": {
                    "task_id": "S1_1",
                    "error_type": "wrong_answer",
                    "detail": "expected 1, got 0",
                    "method": "verification_report",
                    "confidence": "low",
                },
            }
        )

    updated = result["updated_task_node"]
    for field in (
        "id",
        "goal",
        "type",
        "is_leaf",
        "agent_type",
        "parent",
        "dependencies",
        "inputs",
        "outputs",
        "interface",
    ):
        assert updated[field] == original[field]
    assert updated["hints"] == malicious_proposal["hints"]
    assert updated["context"] == malicious_proposal["context"]


def test_post_repair_runner_retains_initial_outcome_and_validates_new_code():
    state = {
        "execution_result": {
            "all_passed": False,
            "error_type": "wrong_answer",
        },
        "final_code": (
            "def solve():\n"
            "    x = int(input())\n"
            "    print(x + 1)\n\n"
            "if __name__ == '__main__':\n"
            "    solve()\n"
        ),
        "task_plan_ir": {
            "problem": {"constraints": {"time_limit": "1s"}},
            "validation": {
                "sample_test_cases": [
                    {"input": "1\n", "expected_output": "2"}
                ]
            },
        },
    }

    result = post_repair_code_runner(state)

    assert result["initial_execution_result"]["all_passed"] is False
    assert result["post_repair_execution_result"]["all_passed"] is True
    assert result["execution_result"]["all_passed"] is True


def test_token_judge_ignores_trailing_and_intertoken_whitespace():
    outcome = judge_output(
        problem_id=4406,
        input_data="",
        expected="2\n2 1 ",
        actual="2   \n  2 1",
    )

    assert outcome["passed"] is True
    assert outcome["judge_mode"] == "token"


def test_constructive_judges_accept_alternative_valid_witnesses():
    coloring = judge_output(
        problem_id=4105,
        input_data="4 3\n",
        expected="YES\n3 1\n1 3\n3 2\n2 3\n",
        actual="YES\n1 2\n2 3\n3 1\n1 3\n",
    )
    room = judge_output(
        problem_id=4974,
        input_data="100 5\n42\n3\n2\n99\n1\n",
        expected="23\n",
        actual="4\n",
    )

    assert coloring["passed"] is True
    assert room["passed"] is True
    assert coloring["judge_mode"] == room["judge_mode"] == "semantic_checker"


def test_team_assignment_judge_accepts_whitespace_and_valid_optimal_assignment():
    outcome = judge_output(
        problem_id=4451,
        input_data="5\n1 1 3 4 2\n",
        expected="3 1\n1 1 1 1 1 \n",
        actual="3 1\n1 1 1 1 1\n",
    )

    assert outcome["passed"] is True
    assert outcome["judge_mode"] == "semantic_checker"


@pytest.mark.parametrize(
    ("problem_id", "input_data", "expected", "actual"),
    [
        (
            4078,
            "5 4\n2 -2 3 1 2\n1 3\n4 5\n2 5\n1 3\n",
            "6\n2\n4 1\n",
            "6\n2\n1 4\n",
        ),
        (
            4091,
            "8 3\n5 4 2 6 5 1 9 2\n",
            "20\n4 1 3\n",
            "20\n1 3 4\n",
        ),
        (
            4377,
            "3 6 5 4\n",
            "1 2 3\n",
            "3 2 1\n",
        ),
        (
            4518,
            "1\n5\n1 2 2 1 3\n",
            "YES\n1 2\n1 3\n1 5\n5 4\n",
            "YES\n1 2\n1 3\n1 5\n2 4\n",
        ),
        (
            4763,
            "96\n",
            "triple 19\ndouble 15\nsingle 9\n",
            "triple 12\ntriple 20\n",
        ),
        (
            4897,
            (
                "6\n"
                "ASHMEADE 9.90 8.85\n"
                "BLAKE 9.69 8.72\n"
                "BOLT 9.58 8.43\n"
                "CARTER 9.78 8.93\n"
                "FRATER 9.88 8.92\n"
                "POWELL 9.72 8.61\n"
            ),
            "35.540000\nCARTER\nBOLT\nPOWELL\nBLAKE\n",
            "35.54\nCARTER\nBOLT\nPOWELL\nBLAKE\n",
        ),
        (
            4966,
            "2 20 15\n3 10\n",
            "18.400\n",
            "18.400000\n",
        ),
    ],
)
def test_new_semantic_judges_accept_equivalent_outputs(
    problem_id, input_data, expected, actual
):
    outcome = judge_output(
        problem_id=problem_id,
        input_data=input_data,
        expected=expected,
        actual=actual,
    )

    assert outcome["passed"] is True
    assert outcome["judge_mode"] == "semantic_checker"


@pytest.mark.parametrize(
    ("problem_id", "input_data", "expected", "actual"),
    [
        (
            4078,
            "5 4\n2 -2 3 1 2\n1 3\n4 5\n2 5\n1 3\n",
            "6\n2\n4 1\n",
            "6\n1\n1\n",
        ),
        (
            4091,
            "8 3\n5 4 2 6 5 1 9 2\n",
            "20\n4 1 3\n",
            "20\n1 1 6\n",
        ),
        (4377, "3 6 5 4\n", "1 2 3\n", "3 2 2\n"),
        (
            4518,
            "1\n5\n1 2 2 1 3\n",
            "YES\n1 2\n1 3\n1 5\n5 4\n",
            "YES\n1 2\n1 3\n1 5\n1 4\n",
        ),
        (
            4763,
            "96\n",
            "triple 19\ndouble 15\nsingle 9\n",
            "triple 12\ntriple 19\n",
        ),
        (
            4966,
            "2 20 15\n3 10\n",
            "18.400\n",
            "12.4047619048\n",
        ),
    ],
)
def test_new_semantic_judges_reject_invalid_outputs(
    problem_id, input_data, expected, actual
):
    outcome = judge_output(
        problem_id=problem_id,
        input_data=input_data,
        expected=expected,
        actual=actual,
    )

    assert outcome["passed"] is False
    assert outcome["judge_mode"] == "semantic_checker"


def test_problem_statement_supplements_restore_missing_contract_evidence():
    tetris = _supplement_problem_text(4824, "seven pieces are shown below")
    chess = _supplement_problem_text(4868, "reverse case")
    untouched = _supplement_problem_text(1, "ordinary statement")

    assert "3: [(0,0), (1,0), (1,1), (2,1)]" in tetris
    assert "5: [(0,0), (1,0), (2,0), (1,1)]" in tetris
    assert 'input is exactly two piece-list lines beginning with "White:"' in chess
    assert "output is the 17-line ASCII chessboard" in chess
    assert untouched == "ordinary statement"


def test_repair_acceptance_rolls_back_non_improving_candidate():
    initial_execution = {
        "all_passed": False,
        "results": [
            {"passed": True, "exit_code": 0, "stderr": "", "timed_out": False},
            {"passed": False, "exit_code": 0, "stderr": "", "timed_out": False},
        ],
    }
    initial_apps = {
        "all_passed": False,
        "cases": [
            {"passed": True, "error": None},
            {"passed": False, "error": None},
        ],
    }
    crashing_execution = {
        "all_passed": False,
        "results": [
            {
                "passed": False,
                "exit_code": 1,
                "stderr": "IndexError",
                "timed_out": False,
            },
            {
                "passed": False,
                "exit_code": 1,
                "stderr": "IndexError",
                "timed_out": False,
            },
        ],
    }
    crashing_apps = {
        "all_passed": False,
        "cases": [
            {"passed": False, "error": "IndexError"},
            {"passed": False, "error": "IndexError"},
        ],
    }

    assert _candidate_validation_rank(
        initial_execution, initial_apps
    ) > _candidate_validation_rank(crashing_execution, crashing_apps)
    record = _repair_acceptance_record(
        initial_execution=initial_execution,
        initial_apps_tests=initial_apps,
        candidate_execution=crashing_execution,
        candidate_apps_tests=crashing_apps,
        candidate_code="raise IndexError\n",
    )

    assert record["accepted"] is False
    assert record["decision"] == "rolled_back"


def test_canonical_cases_preserve_multi_test_bundle_verbatim():
    bundled_input = "2\n1\n1000\n3\n1 1 1\n"
    bundled_output = "1 1000 0\n2 1 2\n"
    cases = _canonical_sample_cases(
        {"inputs": [bundled_input], "outputs": [bundled_output]},
        max_cases=10,
    )
    ir = {
        "validation": {
            "sample_test_cases": [
                {"input": "1\n1000", "expected_output": "1 1000 0"}
            ]
        }
    }

    updated = _install_canonical_validation(ir, cases)

    assert updated["validation"]["sample_test_cases"][0]["input"] == bundled_input
    assert (
        updated["validation"]["sample_test_cases"][0]["expected_output"]
        == bundled_output
    )
    assert updated["validation"]["test_source"] == "dataset.input_output"
    assert updated["validation"]["llm_sample_test_cases"] == ir["validation"][
        "sample_test_cases"
    ]


def test_repair_gate_abstains_when_internal_oracle_disagrees_with_external():
    evidence, record = _select_failure_evidence(
        {"all_passed": False, "error_type": "wrong_answer"},
        {"all_passed": True, "cases": []},
    )

    assert evidence is None
    assert record == {
        "status": "abstained",
        "method": "evidence_disagreement",
        "error_type": "wrong_answer",
        "reason": "internal_failed_but_canonical_external_passed",
    }


def test_repair_gate_uses_external_failure_when_internal_tests_pass():
    evidence, record = _select_failure_evidence(
        {"all_passed": True, "error_type": "none"},
        {
            "all_passed": False,
            "cases": [
                {
                    "index": 0,
                    "input": "1\n",
                    "expected": "2",
                    "actual": "1",
                    "passed": False,
                    "error": None,
                    "judge_mode": "token",
                }
            ],
        },
    )

    assert record is None
    assert evidence is not None
    assert evidence["error_type"] == "wrong_answer"
    assert evidence["results"][0]["input"] == "1\n"


def test_final_report_contains_auditable_repair_history():
    localization = {
        "status": "localized",
        "task_id": "S1_1",
        "method": "traceback",
        "confidence": "high",
    }
    report = final_report_assembler(
        {
            "execution_result": {"all_passed": True, "error_type": "none"},
            "initial_execution_result": {
                "all_passed": False,
                "error_type": "runtime_error",
            },
            "post_repair_execution_result": {
                "all_passed": True,
                "error_type": "none",
            },
            "error_location": localization,
            "localization_record": localization,
            "retried_tasks": ["S1_1"],
        }
    )["final_report"]

    assert report["retry_info"]["global_validation_after_repair"] is True
    assert report["repair_history"] == [
        {
            "repair_attempt": 1,
            "localization": localization,
            "repair_region": {
                "region_type": "node",
                "nodes": ["S1_1"],
            },
            "initial_validation_passed": False,
            "post_repair_validation_passed": True,
        }
    ]


def test_final_verifier_graph_revalidates_after_reassembly():
    graph = build_final_verifier_graph().get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}

    assert ("code_reassembler", "post_repair_code_runner") in edges
    assert ("post_repair_code_runner", "final_report_assembler") in edges
    assert ("code_reassembler", "final_report_assembler") not in edges
