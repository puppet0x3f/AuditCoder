"""Node: bruteforce_verifier — 小规模暴力对拍。

LLM 生成暴力参考解 + 输入生成器，程序化执行对拍。
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from auditcoder.common.llm import get_llm, invoke_llm
from auditcoder.common.utils import extract_json
from auditcoder.verifier_agent.prompts.bruteforce import (
    BRUTEFORCE_GEN_SYSTEM,
    BRUTEFORCE_GEN_USER,
)
from auditcoder.verifier_agent.sandbox import SandboxExecutor
from auditcoder.verifier_agent.state import VerifierState

NUM_RANDOM_TESTS = 100


def _build_comparison_code(
    candidate_code: str,
    bruteforce_code: str,
    input_generator_code: str,
    function_name: str,
    num_tests: int,
) -> str:
    """Build a script that runs candidate vs bruteforce on random inputs."""
    return f"""\
import json
import math

def _normalise(value):
    if hasattr(value, "to_dict") and hasattr(value, "columns"):
        return {{
            "__dataframe__": True,
            "columns": list(value.columns),
            "index": [_normalise(item) for item in list(value.index)],
            "data": [
                [_normalise(item) for item in row]
                for row in value.to_numpy().tolist()
            ],
        }}
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return _normalise(value.tolist())
    if hasattr(value, "item"):
        try:
            return _normalise(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {{key: _normalise(item) for key, item in value.items()}}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    return value

def _equivalent(left, right):
    left = _normalise(left)
    right = _normalise(right)
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-8, abs_tol=1e-10)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _equivalent(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _equivalent(a, b) for a, b in zip(left, right)
        )
    return left == right

# --- candidate ---
{candidate_code}
_candidate_fn = {function_name}

# --- bruteforce (renamed to avoid collision) ---
{bruteforce_code}
_bruteforce_fn = {function_name}

# --- input generator ---
{input_generator_code}

_mismatches = []
for _i in range({num_tests}):
    _args = generate_test_input()

    # Reload functions to avoid state leakage between candidate/bruteforce.
    # We call by unpacking the dict as keyword args.
    try:
        _candidate_result = _candidate_fn(**_args)
    except Exception as _e:
        _mismatches.append({{"index": _i, "input": _args, "candidate_error": str(_e)}})
        continue

    try:
        _bruteforce_result = _bruteforce_fn(**_args)
    except Exception as _e:
        _mismatches.append({{"index": _i, "input": _args, "bruteforce_error": str(_e)}})
        continue

    if not _equivalent(_candidate_result, _bruteforce_result):
        _mismatches.append({{
            "index": _i,
            "input": _args,
            "candidate_output": repr(_candidate_result),
            "bruteforce_output": repr(_bruteforce_result),
        }})

    if len(_mismatches) >= 5:
        break

print(json.dumps({{"total": {num_tests}, "mismatches": _mismatches}}))
"""


def bruteforce_verifier(state: VerifierState) -> dict:
    compile_report = state.get("compile_report", {})
    if not compile_report.get("compile_ok", False):
        return {
            "bruteforce_report": {
                "match": False,
                "skipped": True,
                "mismatches": [],
                "error": "编译未通过，跳过暴力对拍",
            }
        }

    snippet = state["code_snippet"]
    contract = state.get("task_contract", {})
    problem = state.get("problem_info", {})
    interface = contract.get("interface", {})
    function_name = interface.get("function_name", "")
    validation_info = state.get("validation_info", {})

    # Global APPS answers describe the root task.  An internal helper needs
    # its own semantic contract or task-scoped cases to generate an oracle.
    if (
        validation_info.get("scope") == "explicit_task_id_only"
        and not contract.get("goal")
        and not interface.get("return_description")
        and not validation_info.get("sample_test_cases")
        and not validation_info.get("edge_cases")
    ):
        return {
            "bruteforce_report": {
                "match": True,
                "skipped": True,
                "mismatches": [],
                "error": "缺少节点语义契约和任务级用例，跳过全局题目对拍",
                "comparison_mode": "not_applicable_unscoped_leaf",
            }
        }

    # A callable/function-valued result cannot be compared by object identity:
    # two independently constructed closures are semantically equivalent even
    # though ``left == right`` is false. Keep compile, sample, contract, and
    # complexity checks, and leave differential testing to the caller that can
    # invoke the returned function.
    return_type = str(interface.get("return_type", ""))
    if "callable" in return_type.lower() or "function" in return_type.lower():
        return {
            "bruteforce_report": {
                "match": True,
                "skipped": True,
                "mismatches": [],
                "error": "高阶函数返回值不进行对象身份对拍",
                "comparison_mode": "not_applicable_higher_order_return",
            }
        }

    # Stateful containers (DSU/heap/graph objects) do not have a meaningful
    # equality relation.  Comparing their reprs only compares memory
    # addresses, so every correct implementation was previously rejected and
    # withheld from the assembler.  Their public methods are checked by the
    # root execution instead.
    return_description = str(interface.get("return_description", ""))
    if (
        return_type.strip().lower() in {"object", "any"}
        or any(
            marker in return_description.lower()
            for marker in ("dsu", "disjoint", "heap", "data structure", "class with")
        )
    ):
        return {
            "bruteforce_report": {
                "match": True,
                "skipped": True,
                "mismatches": [],
                "error": "状态容器返回值跳过对象身份对拍，由全局执行检查行为",
                "comparison_mode": "not_applicable_stateful_object",
            }
        }

    if not function_name:
        return {
            "bruteforce_report": {
                "match": False,
                "skipped": True,
                "mismatches": [],
                "error": "无函数接口定义，跳过暴力对拍",
            }
        }

    data_ranges = problem.get("constraints", {}).get("data_ranges", [])
    problem_desc = problem.get("description", problem.get("title", ""))

    llm = get_llm(temperature=0.2)
    messages = [
        SystemMessage(content=BRUTEFORCE_GEN_SYSTEM),
        HumanMessage(
            content=BRUTEFORCE_GEN_USER.format(
                problem_description=problem_desc,
                task_contract_json=json.dumps(contract, ensure_ascii=False, indent=2),
                local_cases_json=json.dumps(
                    {
                        "sample_test_cases": validation_info.get("sample_test_cases", []),
                        "edge_cases": validation_info.get("edge_cases", []),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                interface_json=json.dumps(interface, ensure_ascii=False, indent=2),
                data_ranges_json=json.dumps(data_ranges, ensure_ascii=False, indent=2),
                candidate_code=snippet["source_code"],
            )
        ),
    ]
    response = invoke_llm(llm, messages)
    parsed = extract_json(response)

    if not isinstance(parsed, dict):
        return {
            "bruteforce_report": {
                "match": False,
                "skipped": True,
                "mismatches": [],
                "error": "LLM 暴力对拍响应不是 JSON 对象，已跳过对拍",
                "response_type": type(parsed).__name__,
            },
            "messages": messages + [response],
        }

    bruteforce_code = parsed.get("bruteforce_code", "")
    input_gen_code = parsed.get("input_generator_code", "")

    if not bruteforce_code or not input_gen_code:
        return {
            "bruteforce_report": {
                "match": False,
                "skipped": True,
                "mismatches": [],
                "error": "LLM 未能生成有效的暴力解或输入生成器",
            },
            "messages": messages + [response],
        }

    comparison_code = _build_comparison_code(
        candidate_code=snippet["source_code"],
        bruteforce_code=bruteforce_code,
        input_generator_code=input_gen_code,
        function_name=function_name,
        num_tests=NUM_RANDOM_TESTS,
    )

    executor = SandboxExecutor(default_timeout_s=30.0)
    result = executor.run(comparison_code)

    if result.exit_code != 0 or result.timed_out:
        return {
            "bruteforce_report": {
                "match": False,
                "skipped": True,
                "mismatches": [],
                "error": f"对拍脚本执行失败: {result.stderr[:500]}",
                "bruteforce_code": bruteforce_code,
                "input_generator_code": input_gen_code,
            },
            "messages": messages + [response],
        }

    try:
        comparison_result = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return {
            "bruteforce_report": {
                "match": False,
                "skipped": True,
                "mismatches": [],
                "error": f"对拍输出解析失败: {result.stdout[:500]}",
            },
            "messages": messages + [response],
        }

    mismatches = comparison_result.get("mismatches", [])
    match = len(mismatches) == 0

    return {
        "bruteforce_report": {
            "match": match,
            "skipped": False,
            "total_tests": comparison_result.get("total", NUM_RANDOM_TESTS),
            "mismatches": mismatches,
            "bruteforce_code": bruteforce_code,
            "input_generator_code": input_gen_code,
            "comparison_mode": "recursive numeric tolerance rtol=1e-8 atol=1e-10",
        },
        "messages": messages + [response],
    }
