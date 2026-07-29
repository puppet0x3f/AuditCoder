"""Configurable LLM provider shared across all agents."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)


def _get_max_tokens(override: int | None = None, provider: str = "glm") -> int | None:
    if override is not None:
        return override

    raw = (
        os.getenv(f"{provider.upper()}_MAX_TOKENS")
        or os.getenv("GLM_MAX_TOKENS")
        or os.getenv("MAX_TOKENS")
    )
    if not raw:
        return None

    try:
        return int(str(raw).strip().strip('"').strip("'"))
    except ValueError:
        return None


def get_llm(
    temperature: float = 0.0,
    json_mode: bool = True,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """Return the configured chat model."""
    from langchain_openai import ChatOpenAI

    provider = os.getenv("AUDITCODER_LLM_PROVIDER", "glm").strip().lower()
    if provider == "qwen":
        model = os.getenv("QWEN_MODEL", "qwen-plus")
        base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/")
        api_key = os.getenv("QWEN_API_KEY")
        extra_body = {
            "enable_thinking": False,
        }
    else:
        provider = "glm"
        model = os.getenv("GLM_MODEL", "glm-4.7")
        base_url = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
        api_key = os.getenv("GLM_API_KEY")
        extra_body = {
            "thinking": {
                "type": "disabled"
            }
        }

    model_kwargs: dict = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    resolved_max_tokens = _get_max_tokens(max_tokens, provider=provider)

    kwargs: dict = {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": resolved_max_tokens,
        "model_kwargs": model_kwargs,
    }
    if extra_body is not None:
        kwargs["extra_body"] = extra_body

    return ChatOpenAI(
        **kwargs,
    )
