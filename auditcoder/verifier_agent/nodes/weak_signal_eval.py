"""Node: weak_signal_eval — 弱信号评估（可读性、风格、解释自洽性）。

使用 LLM 进行评估。
"""

from __future__ import annotations

import ast
import json

from langchain_core.messages import HumanMessage, SystemMessage

from auditcoder.common.llm import get_llm, invoke_llm
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
    """Compute non-authoritative style signals locally.

    These signals never participate in the correctness verdict.  Calling the
    remote model here added one unnecessary network round trip per leaf and
    could stall an entire APPS batch while semantic/runtime evidence was
    already complete.
    """
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

    try:
        tree = ast.parse(source_code)
    except Exception as exc:
        return _fallback_report(f"{type(exc).__name__}: {str(exc)[:200]}")
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    has_docstrings = any(ast.get_docstring(node) for node in functions)
    line_count = len(source_code.splitlines())
    # Keep the scale intentionally coarse: these values are weak diagnostics,
    # never a release or rejection signal.
    readability = 4 if functions and line_count <= 220 else 3
    style = 4 if not any(isinstance(node, ast.Global) for node in ast.walk(tree)) else 2
    return {
        "weak_signal_report": {
            "readability_score": readability,
            "style_score": style,
            "explanation_consistent": True,
            "notes": [
                "deterministic local weak-signal check",
                "docstrings_present" if has_docstrings else "no_docstrings_required",
            ],
        }
    }
