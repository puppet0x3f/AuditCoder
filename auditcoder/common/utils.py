"""Shared utilities across all agents."""

from __future__ import annotations

import re
from typing import Any

from json_repair import repair_json
from langchain_core.messages import BaseMessage


def extract_json(response: BaseMessage) -> Any:
    """Parse JSON from an LLM response using json_repair for robustness."""
    content = response.content
    if not isinstance(content, str):
        return content

    content = content.strip()

    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    result = repair_json(content, return_objects=True)
    if isinstance(result, (dict, list)):
        return result

    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", content)
    if match:
        result = repair_json(match.group(1), return_objects=True)
        if isinstance(result, (dict, list)):
            return result

    raise ValueError(f"Failed to extract JSON from LLM response:\n{content[:500]}")
