"""Keep intermediate-function oracles separate from whole-program answers."""

import json

import pytest
from langchain_core.messages import AIMessage

from auditcoder.algo_agents.dispatcher import _build_verifier_input
from auditcoder.final_verifier_agent.audit import locate
from auditcoder.final_verifier_agent.nodes.full_code_runner import full_code_runner
from auditcoder.final_verifier_agent.nodes.final_report_assembler import final_report_assembler
from auditcoder.verifier_agent.nodes import bruteforce_verifier as verify_bruteforce


def _input(source: str) -> dict:
    node = {
        "id": "S1", "is_leaf": True, "agent_type": "math",
        "goal": "Compute total apple pieces: 3*A + P.",
        "inputs": [{"name": "A", "type": "int"}, {"name": "P", "type": "int"}],
        "outputs": [{"name": "total_pieces", "type": "int"}],
        "interface": {
            "function_name": "total_pieces",
            "params": [{"name": "A", "type": "int"}, {"name": "P", "type": "int"}],
            "return_type": "int",
            "return_description": "Total pieces before making pies: 3*A + P.",
        },
    }
    ir = {
        "task_nodes": [node, {"id": "S2", "is_leaf": True, "agent_type": "math"}],
        "problem": {
            "description": "Compute number of pies: (3*A + P) // 2.",
            "constraints": {},
        },
        "validation": {
            "sample_test_cases": [{"input": "3 2", "expected_output": "5"}],
        },
    }
    state = _build_verifier_input("S1", "math", source, node, ir)
    state["compile_report"] = {"compile_ok": True}
    return state


def test_dispatch_preserves_helper_goal_and_excludes_global_samples():
    state = _input("def total_pieces(A, P): return 3*A + P")
    assert state["task_contract"]["goal"] == "Compute total apple pieces: 3*A + P."
    assert state["task_contract"]["outputs"][0]["name"] == "total_pieces"
    assert state["validation_info"]["sample_test_cases"] == []
    assert state["validation_info"]["scope"] == "explicit_task_id_only"


@pytest.mark.parametrize("result_expression, matched", [("3*A + P", True), ("(3*A + P)//2", False)])
def test_scoped_oracle_checks_intermediate_result(monkeypatch, result_expression, matched):
    class LocalOracle:
        def invoke(self, messages):
            assert "Task Responsibility Contract" in messages[1].content
            assert "Compute total apple pieces: 3*A + P." in messages[1].content
            assert "Task-Scoped Test Cases" in messages[1].content
            assert "Whole-program examples" in messages[0].content
            return AIMessage(content=json.dumps({
                "bruteforce_code": "def total_pieces(A, P): return 3*A + P",
                "input_generator_code": "def generate_test_input(): return {'A': 3, 'P': 2}",
            }))

    monkeypatch.setattr(verify_bruteforce.__globals__["get_llm"].__module__ + ".get_llm", lambda **kw: LocalOracle())
    monkeypatch.setitem(verify_bruteforce.__globals__, "get_llm", lambda **kw: LocalOracle())
    state = _input(f"def total_pieces(A, P): return {result_expression}")
    report = verify_bruteforce(state)["bruteforce_report"]
    assert not report["skipped"]
    assert report["match"] is matched
    if not matched:
        assert report["mismatches"][0]["bruteforce_output"] == "11"
        assert report["mismatches"][0]["candidate_output"] == "5"


def test_missing_local_semantics_abstains_instead_of_using_root_answer(monkeypatch):
    def unexpected_model_call(**kw):
        pytest.fail("An unspecified helper must not use the root problem's oracle")

    monkeypatch.setitem(verify_bruteforce.__globals__, "get_llm", unexpected_model_call)
    state = _input("def total_pieces(A, P): return 3*A + P")
    state["task_contract"]["goal"] = ""
    state["task_contract"]["interface"].pop("return_description")
    report = verify_bruteforce(state)["bruteforce_report"]
    assert report["skipped"]
    assert report["comparison_mode"] == "not_applicable_unscoped_leaf"


def test_dataset_cases_override_advisory_planner_mismatch():
    result = full_code_runner(
        {
            "problem_id": 1,
            "final_code": 'print("NO")',
            "task_plan_ir": {
                "problem": {"constraints": {}},
                "validation": {
                    "sample_test_cases": [
                        {"input": "x\n", "expected_output": "YES"}
                    ]
                },
                "development_checks": [
                    {"id": "wrong-edge", "input": "x\n", "expected_output": "YES"}
                ],
            },
            "external_test_cases": [
                {
                    "input": "x\n",
                    "expected_output": "NO",
                    "check_id": "apps:0",
                    "trusted": True,
                }
            ],
        }
    )["execution_result"]
    assert result["all_passed"]
    assert result["trusted_test_count"] == 1
    assert result["trusted_failures"] == []
    assert len(result["advisory_failures"]) == 1
    assert result["error_type"] == "advisory_mismatch"


def test_global_output_attribution_uses_explicit_root_owner():
    ir = {
        "root_id": "S0",
        "task_nodes": [
            {"id": "S0", "parent": None, "owns_code": True, "is_leaf": False},
            {"id": "S1", "parent": "S0", "is_leaf": True, "agent_type": "math"},
        ],
    }
    execution = {
        "all_passed": False,
        "error_type": "wrong_answer",
        "trusted_failures": [{"case_index": 0}],
        "results": [
            {
                "case_index": 0,
                "input": "1 2\n",
                "expected": "3",
                "actual": "4",
                "passed": False,
                "exit_code": 0,
                "stderr": "",
                "symptom": "WA",
            }
        ],
    }
    location = locate(execution, ir, [], code_snippets={"S1": "def f(): pass"})
    assert location["rule"] == "GlobalAttribution"
    assert location["method"] == "global_output_attribution"
    assert location["task_id"] == "S0"
    assert location["confidence"] == "Low"
    assert location["evidence"][0]["witness"]["applicability"] == "observed"


def test_global_output_attribution_skips_assembly_only_root():
    ir = {
        "root_id": "S0",
        "task_nodes": [
            {
                "id": "S0",
                "parent": None,
                "owns_code": True,
                "owned_units": ["solve", "__main__"],
                "is_leaf": False,
            },
            {"id": "S1", "parent": "S0", "is_leaf": True, "agent_type": "math"},
        ],
    }
    execution = {
        "all_passed": False,
        "error_type": "wrong_answer",
        "trusted_failures": [{"case_index": 0}],
        "results": [
            {
                "case_index": 0,
                "input": "1\n",
                "expected": "1",
                "actual": "2",
                "passed": False,
                "exit_code": 0,
                "stderr": "",
                "symptom": "WA",
            }
        ],
    }
    location = locate(execution, ir, [], code_snippets={"S1": "def f(): pass"})
    assert location["task_id"] == "S1"


def test_global_output_attribution_can_use_weak_local_signal_after_input_mismatch():
    ir = {
        "root_id": "S0",
        "task_nodes": [
            {"id": "S0", "parent": None, "is_leaf": False},
            {"id": "S1", "parent": "S0", "is_leaf": True, "agent_type": "math"},
        ],
    }
    execution = {
        "all_passed": False,
        "error_type": "wrong_answer",
        "trusted_failures": [{"case_index": 0}],
        "results": [
            {
                "case_index": 0,
                "input": "1 2\n",
                "expected": "3",
                "actual": "4",
                "passed": False,
                "exit_code": 0,
                "stderr": "",
                "symptom": "WA",
            }
        ],
    }
    records = [
        {
            "id": "S1:local",
            "owners": ["S1"],
            "verdict": "Reject",
            "obligation": {"class": "semantic"},
            "witness": {"input": "helper-only"},
        }
    ]
    location = locate(execution, ir, records, code_snippets={"S1": "def f(): pass"})
    assert location["rule"] == "GlobalAttribution"
    assert location["task_id"] == "S1"
    assert "local_rejection_signal" in location["attribution_basis"]


def test_global_output_attribution_ignores_advisory_only_failure():
    ir = {
        "root_id": "S0",
        "task_nodes": [{"id": "S0", "parent": None, "owns_code": True}],
    }
    execution = {
        "all_passed": False,
        "error_type": "advisory_mismatch",
        "trusted_failures": [],
        "results": [{"passed": False, "input": "x", "expected": "1", "actual": "2"}],
    }
    location = locate(execution, ir, [], code_snippets={})
    assert location["status"] == "abstained"
    assert location["reason"] == "no_compatible_evidence"


def test_accepted_repair_supersedes_stale_local_rejection_in_final_report():
    report = final_report_assembler(
        {
            "execution_result": {"all_passed": True, "error_type": "none"},
            "verification_reports": {"S1": {"verdict": "reject"}},
            "repair_history": [
                {
                    "decision": "Accept",
                    "repair_region": {"nodes": ["S1"]},
                }
            ],
        }
    )["final_report"]
    assert report["verdict"] == "accept"
    assert report["rejected_task_ids"] == []
    assert report["raw_rejected_task_ids"] == ["S1"]
    assert report["superseded_rejected_task_ids"] == ["S1"]


def test_trusted_benchmark_pass_overrides_advisory_leaf_rejection():
    report = final_report_assembler(
        {
            "execution_result": {
                "all_passed": True,
                "trusted_test_count": 2,
                "trusted_failures": [],
                "advisory_failures": [{"case_index": 4}],
                "error_type": "advisory_mismatch",
            },
            "verification_reports": {"S1": {"verdict": "reject"}},
        }
    )["final_report"]
    assert report["verdict"] == "accept"
    assert report["all_passed"] is True
    assert report["rejected_task_ids"] == ["S1"]
    assert report["verification_only_rejections"] == ["S1"]
