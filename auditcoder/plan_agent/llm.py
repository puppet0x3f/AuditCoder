"""Configurable LLM provider for the Plan Agent.

Re-exports from auditcoder.common.llm for backward compatibility.
"""

from auditcoder.common.llm import get_llm

__all__ = ["get_llm"]
