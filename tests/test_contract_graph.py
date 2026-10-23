"""Deterministic regression tests for the paper-facing CATG adapter."""

from __future__ import annotations

from auditcoder.schemas.contract_graph import (
    catg_summary,
    ensure_catg,
    record_local_evidence,
    register_assembly_units,
    register_code_owner,
    validate_contract_graph,
)


def _ir() -> dict:
    return {
        "root_id": "S0",
        "task_nodes": [
            {"id": "S0", "goal": "orchestrate", "is_leaf": False, "parent": None, "dependencies": ["S1"]},
            {
                "id": "S1",
                "goal": "compute",
                "is_leaf": True,
                "agent_type": "dp",
                "parent": "S0",
                "dependencies": [],
                "inputs": [{"name": "x", "type": "list[int]", "source": None}],
                "outputs": [{"name": "y", "type": "int"}],
                "interface": {
                    "function_name": "compute",
                    "params": [{"name": "x", "type": "list[int]"}],
                    "return_type": "int",
                    "preconditions": ["x is non-empty"],
                    "postconditions": ["result >= 0"],
                },
            },
        ],
        "execution_order": ["S1"],
        "validation": {"sample_test_cases": [{"input": "1", "expected_output": "1"}], "edge_cases": []},
    }


def test_catg_allocates_ids_edges_q_and_owner_registry():
    ir = ensure_catg(_ir())
    assert not validate_contract_graph(ir)
    graph = ir["contract_annotated_task_graph"]
    assert graph["root_id"] == "S0"
    assert graph["dependency_edges"] == [{"source": "S1", "target": "S0"}]
    assert graph["development_checks"][0]["id"] == "sample:0"
    assert {record["id"] for record in ir["responsibility_records"]} == {"S0", "S1"}

    ir = register_code_owner(ir, "S1", "def compute(x):\n    return 1\n")
    ir = register_assembly_units(ir, "def compute(x):\n    return 1\ndef solve(): pass\nif __name__ == '__main__': solve()\n")
    names = {item["name"] for item in ir["ownership_map"]["units"]}
    assert names >= {"compute", "solve", "__main__"}
    assert not validate_contract_graph(ir)


def test_local_rejection_is_versioned_and_addressed_to_owner():
    ir = ensure_catg(_ir())
    records = record_local_evidence(
        ir,
        "S1",
        {
            "verdict": "reject",
            "summary": "counterexample",
            "counterexample": {
                "minimal_input": "[]",
                "expected_output": "0",
                "actual_output": "1",
            },
        },
    )
    assert len(records) == 1
    assert records[0]["owners"] == ["S1"]
    assert records[0]["obligation"]["class"] == "semantic"
    assert records[0]["kind"] == "judged"
    assert catg_summary(ir)["evidence_record_count"] == 1


def test_catg_rebuild_keeps_append_only_evidence_history():
    ir = ensure_catg(_ir())
    ir = register_code_owner(
        ir,
        "S1",
        "def compute(x):\n    return 1\n",
        verification_report={"verdict": "reject", "summary": "first"},
    )
    record_local_evidence(ir, "S1", {"verdict": "reject", "summary": "first"})
    ir = register_code_owner(
        ir,
        "S1",
        "def compute(x):\n    return 2\n",
        verification_report={"verdict": "reject", "summary": "second"},
    )
    record_local_evidence(ir, "S1", {"verdict": "reject", "summary": "second"})
    assert len(ir["evidence_records"]) == 2
    assert sum(bool(item.get("current")) for item in ir["evidence_records"]) == 1
    owner = next(item for item in ir["responsibility_records"] if item["id"] == "S1")
    assert len(owner["validation"]) >= 4
