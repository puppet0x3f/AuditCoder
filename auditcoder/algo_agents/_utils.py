"""Shared utilities for all algorithm agent code-generator nodes."""

from __future__ import annotations

import json
from typing import Any


def format_interface(interface: dict | None) -> str:
    if not interface:
        return "（接口未定义）"
    lines = [
        f"函数名：{interface.get('function_name', '?')}",
        f"参数：{json.dumps(interface.get('params', []), ensure_ascii=False)}",
        f"返回类型：{interface.get('return_type', '?')}",
        f"返回说明：{interface.get('return_description', '')}",
        (
            "片段自包含要求：必须在代码片段中包含函数定义、类型注解和函数体"
            "所需的全部 import；不得依赖最终装配器补充导入。"
        ),
    ]
    pre = interface.get("preconditions", [])
    post = interface.get("postconditions", [])
    if pre:
        lines.append(f"前置条件：{'; '.join(pre)}")
    if post:
        lines.append(f"后置条件：{'; '.join(post)}")
    return "\n".join(lines)


def format_slots(slots: list[dict]) -> str:
    if not slots:
        return "（无）"
    return "\n".join(
        f"  - {s.get('name')} ({s.get('type')}): {s.get('description', '')}  [source={s.get('source')}]"
        for s in slots
    )


def format_feedback(feedback: dict[str, Any] | None) -> str:
    """Render verifier feedback into a human-readable prompt section.

    Returns an empty string when feedback is None or verdict is not 'reject'.
    """
    if not feedback:
        return ""

    verdict = feedback.get("verdict", "")
    if verdict != "reject":
        return ""

    lines = [
        "",
        "## ⚠️ 上次生成的代码未通过验证，请根据以下反馈修复",
        "",
    ]

    summary = feedback.get("summary", "")
    if summary:
        lines.append(f"错误摘要：{summary}")

    strong = feedback.get("strong_signals", {})

    if not strong.get("compile_ok", True):
        lines.append("问题：代码编译失败，请检查语法错误。")

    if not strong.get("sample_tests_passed", True):
        lines.append("问题：样例测试未通过。")

    if not strong.get("bruteforce_match", True):
        lines.append("问题：与暴力解对拍不一致，存在逻辑错误。")

    if not strong.get("resource_ok", True):
        usage = strong.get("resource_usage", {})
        lines.append(
            f"问题：超时或超内存（耗时 {usage.get('max_time_ms', '?')} ms，"
            f"内存 {usage.get('max_memory_kb', '?')} KB）。"
        )

    medium = feedback.get("medium_signals", {})
    violations = medium.get("contract_violations", [])
    if violations:
        lines.append("接口契约违规：")
        for v in violations:
            lines.append(f"  - {v}")

    counterexample = feedback.get("counterexample")
    if counterexample:
        lines.append("")
        lines.append("反例（用于复现问题）：")
        lines.append(f"  输入：{counterexample.get('minimal_input', '')}")
        lines.append(f"  期望输出：{counterexample.get('expected_output', '')}")
        lines.append(f"  实际输出：{counterexample.get('actual_output', '')}")
        error_cat = counterexample.get("error_category", "")
        if error_cat:
            lines.append(f"  错误类别：{error_cat}")
        reason = counterexample.get("suspected_reason", "")
        if reason:
            lines.append(f"  疑似原因：{reason}")

    lines.append("")
    lines.append("请重新实现该函数，修复上述所有问题，输出 JSON：{\"code_snippet\": \"...\"}")

    return "\n".join(lines)
