"""Node: verdict_assembler — 汇总所有信号输出最终裁决。

纯程序化，不调用 LLM。
"""

from __future__ import annotations

from auditcoder.verifier_agent.state import VerifierState


def verdict_assembler(state: VerifierState) -> dict:
    snippet = state.get("code_snippet", {})
    task_id = snippet.get("task_id", "")

    compile_report = state.get("compile_report", {})
    contract_report = state.get("contract_report", {})
    sample_report = state.get("sample_test_report", {})
    bf_report = state.get("bruteforce_report", {})
    cx_report = state.get("complexity_report", {})
    ce = state.get("counterexample")
    ws_report = state.get("weak_signal_report", {})

    # ── Strong signals ──
    syntax_ok = compile_report.get("syntax_ok", False)
    compile_ok = compile_report.get("compile_ok", False)
    sample_passed = sample_report.get("passed", False)
    bf_match = bf_report.get("match", False)
    resource_ok = sample_report.get("resource_ok", True)

    strong = {
        "syntax_ok": syntax_ok,
        "compile_ok": compile_ok,
        "sample_tests_passed": sample_passed,
        "bruteforce_match": bf_match,
        "resource_ok": resource_ok,
        "resource_usage": {
            "max_time_ms": sample_report.get("max_time_ms", 0.0),
            "max_memory_kb": sample_report.get("max_memory_kb", 0.0),
        },
    }

    # ── Medium signals ──
    contract_ok = contract_report.get("contract_ok", False)
    static_ok = cx_report.get("static_ok", False)
    dynamic_ok = cx_report.get("dynamic_ok", False)
    estimated_cx = cx_report.get("estimated_complexity", "unknown")

    medium = {
        "complexity_static_ok": static_ok,
        "complexity_dynamic_ok": dynamic_ok,
        "estimated_complexity": estimated_cx,
        "no_redundant_states": True,
        "contract_ok": contract_ok,
        "contract_violations": contract_report.get("violations", []),
    }

    # ── Weak signals ──
    weak = {
        "readability_score": ws_report.get("readability_score", 0),
        "style_score": ws_report.get("style_score", 0),
        "explanation_consistent": ws_report.get("explanation_consistent", False),
        "notes": ws_report.get("notes", []),
    }

    # ── Verdict logic ──
    strong_values = [syntax_ok, compile_ok, sample_passed, bf_match, resource_ok]

    # Skip bruteforce match check if it was skipped
    if bf_report.get("skipped", False):
        strong_values = [syntax_ok, compile_ok, sample_passed, resource_ok]

    any_strong_fail = not all(strong_values)
    medium_values = [contract_ok, static_ok, dynamic_ok]
    any_medium_fail = not all(medium_values)

    if any_strong_fail:
        verdict = "reject"
    elif any_medium_fail:
        verdict = "warning"
    else:
        verdict = "accept"

    # ── Summary ──
    reasons: list[str] = []
    if not syntax_ok:
        reasons.append("语法错误")
    if not compile_ok:
        reasons.append("编译失败")
    if not sample_passed:
        reasons.append("公共测试未通过")
    if not bf_match and not bf_report.get("skipped", False):
        reasons.append("暴力对拍不一致")
    if not resource_ok:
        reasons.append("资源超限")
    if not contract_ok:
        reasons.append(f"接口契约违反: {', '.join(contract_report.get('violations', []))}")
    if not static_ok:
        reasons.append(f"静态复杂度超预算 (估计 {estimated_cx})")
    if not dynamic_ok:
        reasons.append("动态复杂度超预算")

    if not reasons:
        summary = "所有检查通过"
    else:
        summary = "; ".join(reasons)

    return {
        "verification_report": {
            "task_id": task_id,
            "verdict": verdict,
            "strong_signals": strong,
            "medium_signals": medium,
            "weak_signals": weak,
            "counterexample": ce,
            "summary": summary,
        }
    }
