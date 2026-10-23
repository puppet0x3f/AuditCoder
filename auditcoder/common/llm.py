"""Configurable LLM provider shared across all agents."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


def _normalize_model(provider: str, model: str, base_url: str) -> str:
    """Map repository model aliases to the provider's API model IDs."""

    normalized = model.strip()
    if (
        provider == "qwen"
        and normalized.lower() == "qwen-3.6-plus"
        and "dashscope.aliyuncs.com" in base_url.lower()
    ):
        return "qwen3.6-plus"
    return normalized


def _get_max_tokens(override: int | None = None, provider: str = "glm") -> int | None:
    if override is not None:
        return override

    raw = (
        os.getenv(f"{provider.upper()}_MAX_TOKENS")
        or os.getenv("GLM_MAX_TOKENS")
        or os.getenv("MAX_TOKENS")
    )
    if not raw:
        # DeepSeek's API default completion limit is large enough to exhaust
        # the JSON/code response before an agent closes its object.  Keep a
        # bounded default so planner and assembler calls terminate with a
        # usable response; callers that need more (the standalone recovery
        # writer) pass an explicit budget.
        if provider == "deepseek":
            return 5500
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
    elif provider == "deepseek":
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        # The planning graph already has a dedicated deep-thinker node.  Keep
        # the API-side reasoning channel off for structured Agent responses;
        # otherwise a small per-agent token budget can be consumed entirely by
        # reasoning and leave ``message.content`` empty.
        extra_body = {
            "thinking": {
                "type": "disabled"
            }
        }
    elif provider == "glm":
        provider = "glm"
        model = os.getenv("GLM_MODEL", "glm-4.7")
        base_url = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
        api_key = os.getenv("GLM_API_KEY")
        extra_body = {
            "thinking": {
                "type": "disabled"
            }
        }
    else:
        raise ValueError(
            "AUDITCODER_LLM_PROVIDER must be one of: qwen, deepseek, glm"
        )

    model = _normalize_model(provider, model, base_url)

    model_kwargs: dict = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    resolved_max_tokens = _get_max_tokens(max_tokens, provider=provider)

    # Some DeepSeek-compatible gateways still read the legacy ``max_tokens``
    # field and silently ignore OpenAI's newer ``max_completion_tokens``.
    # Supplying both keeps the response bounded on either API variant and
    # prevents JSON/code agents from being cut off at the gateway default.
    if provider == "deepseek" and resolved_max_tokens is not None:
        extra_body = dict(extra_body or {})
        extra_body.setdefault("max_tokens", resolved_max_tokens)

    try:
        request_timeout = float(os.getenv("AUDITCODER_LLM_TIMEOUT", "180"))
    except ValueError:
        request_timeout = 180.0

    kwargs: dict = {
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "temperature": temperature,
        "max_tokens": resolved_max_tokens,
        "model_kwargs": model_kwargs,
        "timeout": request_timeout,
        "max_retries": 1,
    }
    if extra_body is not None:
        kwargs["extra_body"] = extra_body

    return ChatOpenAI(
        **kwargs,
    )


def invoke_llm(llm: BaseChatModel, messages: Sequence[Any]) -> Any:
    """Invoke a structured agent and recover once from a truncated response.

    DeepSeek-compatible gateways occasionally ignore the requested completion
    budget and stop exactly at their token ceiling, which raises
    ``LengthFinishReasonError`` before the JSON parser can run.  A compact
    second request gives the agent a chance to emit the required object/code
    without changing the caller's control flow or repair budget.
    """
    try:
        return llm.invoke(messages)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}".lower()
        if "length" not in message or "finish" not in message and "limit" not in message:
            raise

        compact: list[Any] = []
        for item in messages:
            content = str(getattr(item, "content", ""))
            if len(content) > 24000:
                content = content[:24000] + "\n[long context elided]"
            if isinstance(item, SystemMessage):
                compact.append(SystemMessage(content=content))
            else:
                compact.append(HumanMessage(content=content))
        compact.append(
            HumanMessage(
                content=(
                    "Recovery request: emit the smallest complete valid JSON object now. "
                    "Keep only executable code and essential fields; omit explanations, "
                    "long docstrings, duplicated helpers, and markdown. Close the JSON "
                    "object before the response limit."
                )
            )
        )
        # The compact request still has to carry a complete CATG or code
        # object.  A 2800-token ceiling routinely truncates decompositions
        # for long APPS statements; keep the normal DeepSeek budget for the
        # recovery call while asking the model to omit prose.
        retry = get_llm(temperature=0.0, json_mode=True, max_tokens=5500)
        return retry.invoke(compact)
