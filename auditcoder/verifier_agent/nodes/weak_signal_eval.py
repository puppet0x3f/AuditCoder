"""Node: weak_signal_eval — 弱信号评估（可读性、风格、解释自洽性）。

使用 LLM 进行评估。
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from auditcoder.common.llm import get_llm
from auditcoder.common.utils import extract_json
from auditcoder.verifier_agent.prompts.weak_signal import (
    WEAK_SIGNAL_SYSTEM,
    WEAK_SIGNAL_USER,
)
from auditcoder.verifier_agent.state import VerifierState


MAX_WEAK_SIGNAL_CODE_CHARS = 6000
MAX_WEAK_SIGNAL_EXPLANATION_CHARS = 1500


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n# ... truncated for weak-signal review ..."


def _fallback_report(reason: str) -> dict:
    return {
        "weak_signal_report": {
            "readability_score": 3,
            "style_score": 3,
            "explanation_consistent": True,
            "notes": [f"weak_signal_eval skipped: {reason}"],
        }
    }


def weak_signal_eval(state: VerifierState) -> dict:
    snippet = state["code_snippet"]
    contract = state.get("task_contract") or {}
    interface = contract if contract.get("function_name") else contract.get("interface", {})
    source_code = _truncate_text(
        str(snippet.get("source_code", "")),
        MAX_WEAK_SIGNAL_CODE_CHARS,
    )
    explanation = _truncate_text(
        str(snippet.get("explanation", "（未提供解释）")),
        MAX_WEAK_SIGNAL_EXPLANATION_CHARS,
    )

    messages = [
        SystemMessage(content=WEAK_SIGNAL_SYSTEM),
        HumanMessage(
            content=WEAK_SIGNAL_USER.format(
                source_code=source_code,
                interface_json=json.dumps(interface, ensure_ascii=False, indent=2),
                explanation=explanation,
            )
        ),
    ]

    try:
        llm = get_llm(temperature=0.0)
        response = llm.invoke(messages)
        parsed = extract_json(response)
    except Exception as exc:
        return _fallback_report(f"{type(exc).__name__}: {str(exc)[:200]}")

    if not isinstance(parsed, dict):
        return _fallback_report("LLM returned non-object JSON")

    return {
        "weak_signal_report": {
            "readability_score": parsed.get("readability_score", 0),
            "style_score": parsed.get("style_score", 0),
            "explanation_consistent": parsed.get("explanation_consistent", False),
            "notes": parsed.get("notes", []),
        },
        "messages": messages + [response],
    }
