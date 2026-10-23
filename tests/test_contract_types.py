from auditcoder.verifier_agent.nodes.contract_verifier import contract_verifier


def test_contract_verifier_accepts_equivalent_builtin_and_typing_annotations():
    result = contract_verifier(
        {
            "code_snippet": {
                "source_code": "def f(xs: list[int]) -> tuple[int, ...]: return tuple(xs)"
            },
            "task_contract": {
                "interface": {
                    "function_name": "f",
                    "params": [{"name": "xs", "type": "List[int]"}],
                    "return_type": "Tuple[int, ...]",
                }
            },
        }
    )
    assert result["contract_report"]["contract_ok"]
    assert result["contract_report"]["violations"] == []
