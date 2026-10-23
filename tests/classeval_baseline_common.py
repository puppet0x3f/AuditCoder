#!/usr/bin/env python3
"""Shared helpers for the ClassEval baseline runners.

The three baseline entry points deliberately keep model generation separate
from the benchmark oracle.  This module only contains the common data loading,
API-compatible chat client, code extraction, and ClassEval execution helpers.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT
CLASSEVAL_DATA = REPO_ROOT / "data" / "ClassEval_data.json"
NAACL_RUNNER = REPO_ROOT / "tests" / "run_classeval_auditcoder.py"

_FENCED_CODE = re.compile(
    r"```(?:python|python3|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL
)
_EVAL_MODULE: Any | None = None


class APIRequestError(RuntimeError):
    """Raised when an OpenAI-compatible request cannot be completed."""


class OutputTruncatedError(APIRequestError):
    """Raised when the provider ends a response at its token limit."""


def preserve_or_wrap_cdata(text: str, tag: str) -> str:
    """Wrap XML element text without duplicating an existing CDATA section."""

    pattern = re.compile(rf"(<{re.escape(tag)}>)(.*?)(</{re.escape(tag)}>)", re.DOTALL)

    def replace(match: re.Match[str]) -> str:
        body = match.group(2)
        if re.match(r"\s*<!\[CDATA\[", body) and re.search(r"\]\]>\s*$", body):
            return match.group(0)
        return f"{match.group(1)}<![CDATA[{body}]]>{match.group(3)}"

    return pattern.sub(replace, text)


def read_env_file(path: Path) -> None:
    """Load simple ``KEY=value`` entries without overwriting the shell."""

    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_classeval(path: Path = CLASSEVAL_DATA) -> list[dict[str, Any]]:
    """Read and minimally validate the official ClassEval JSON file."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ClassEval data file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("ClassEval data must be a JSON list")
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict) or not item.get("task_id"):
            raise ValueError(f"ClassEval row {index} has no task_id")
        task_id = str(item["task_id"])
        if task_id in seen:
            raise ValueError(f"duplicate ClassEval task_id: {task_id}")
        if not item.get("class_name") or not item.get("test"):
            raise ValueError(f"ClassEval row {task_id} is missing class_name/test")
        seen.add(task_id)
        rows.append(item)
    return rows


def _task_sort_key(item: Mapping[str, Any]) -> tuple[int, str]:
    match = re.search(r"(\d+)$", str(item.get("task_id", "")))
    return (int(match.group(1)) if match else -1, str(item.get("task_id", "")))


def select_entries(
    entries: list[dict[str, Any]],
    *,
    start: int = 0,
    end: int | None = None,
    count: int | None = None,
    ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Select deterministic task IDs for smoke, sharded, and resume runs."""

    ordered = sorted(entries, key=_task_sort_key)
    if ids is not None:
        by_id = {str(item["task_id"]): item for item in ordered}
        return [by_id[task_id] for task_id in ids if task_id in by_id]
    if start < 0 or (count is not None and count < 0):
        raise ValueError("start/count must be nonnegative")
    selected = ordered[start : end if end is not None else None]
    return selected if count is None else selected[:count]


def _method_names(entry: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    for method in entry.get("methods_info") or []:
        if isinstance(method, Mapping) and method.get("method_name"):
            names.append(str(method["method_name"]))
    return names


def format_class_prompt(
    entry: Mapping[str, Any],
    *,
    include_tests: bool = True,
    protocol: str = "ClassEval",
    test_limit: int = 24000,
    output_instruction: bool = True,
) -> str:
    """Render a complete class task for an LLM baseline.

    ClassEval's tests are public benchmark specification.  They are included
    for the programmer/coder agents, while the independent AgentCoder test
    designer can request ``include_tests=False``.
    """

    class_name = str(entry.get("class_name", ""))
    methods = entry.get("methods_info") or []
    lines = [
        f"{protocol} Python class implementation task.",
        "Implement the requested class exactly as specified by the skeleton, method descriptions, and trusted unittest suite.",
        f"Class name: {class_name}",
        "",
        "Class description:",
        str(entry.get("class_description", "")).strip(),
        "",
        "Required imports:",
        "\n".join(str(value) for value in (entry.get("import_statement") or [])) or "(none)",
        "",
        "Fields:",
        "\n".join(f"- {value}" for value in (entry.get("fields") or [])) or "(none)",
        "",
        "Constructor contract:",
        str(entry.get("class_constructor", "")).strip() or "Use the constructor in the skeleton.",
        "",
        "Complete skeleton:",
        "```python",
        str(entry.get("skeleton", "")).strip(),
        "```",
        "",
        "Required methods:",
    ]
    for method in methods:
        if not isinstance(method, Mapping):
            continue
        name = str(method.get("method_name", ""))
        if not name:
            continue
        lines.extend(
            [
                f"\n### {name}",
                str(method.get("method_description", "")).strip(),
                f"Dependencies: {json.dumps(method.get('dependencies', {}), ensure_ascii=False)}",
            ]
        )
    if include_tests:
        test_code = str(entry.get("test", ""))
        if test_code:
            lines.extend(["", "Trusted ClassEval unittest suite:", "```python", test_code[:test_limit], "```"])
            if output_instruction:
                lines.extend(
                    [
                        "",
                        "Return only the complete class source. Do not include tests, a main block, or prose.",
                    ]
                )
    return "\n".join(lines)


def extract_code(value: Any) -> str:
    """Extract Python source from fenced, JSON-wrapped, or plain output."""

    if isinstance(value, Mapping):
        for key in ("final_code", "class_code", "code", "completion", "raw"):
            if value.get(key):
                return extract_code(value[key])
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, Mapping):
        for key in ("final_code", "class_code", "code", "completion"):
            if parsed.get(key):
                return extract_code(parsed[key])
    blocks = _FENCED_CODE.findall(text)
    if blocks:
        return blocks[-1].strip()
    return text


def validate_class_code(code: str, class_name: str) -> tuple[bool, str]:
    """Return whether a candidate parses and defines the requested class."""

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"
    if not any(isinstance(node, ast.ClassDef) and node.name == class_name for node in tree.body):
        return False, f"generated code does not define class {class_name!r}"
    return True, ""


def _load_evaluator() -> Any:
    """Load the hardened ClassEval executor lazily.

    Keeping this import lazy makes ``--dry-run`` useful even when only the
    dataset is installed.  The evaluator is the same trusted-unittest harness
    used by the NAACL AuditCoder ClassEval runner.
    """

    global _EVAL_MODULE
    if _EVAL_MODULE is not None:
        return _EVAL_MODULE
    if not NAACL_RUNNER.is_file():
        raise FileNotFoundError(f"ClassEval evaluator not found: {NAACL_RUNNER}")
    implementation_root = NAACL_RUNNER.parents[1]
    if str(implementation_root) not in sys.path:
        sys.path.insert(0, str(implementation_root))
    runner_dir = str(NAACL_RUNNER.parent)
    if runner_dir not in sys.path:
        sys.path.insert(0, runner_dir)
    import importlib.util

    module_name = "_auditcoder_naacl_classeval_runner"
    spec = importlib.util.spec_from_file_location(module_name, NAACL_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import ClassEval evaluator: {NAACL_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _EVAL_MODULE = module
    return module


def evaluate_class(
    entry: Mapping[str, Any],
    code: str,
    *,
    timeout: float = 30.0,
    memory_limit_mb: int = 1024,
) -> dict[str, Any]:
    """Run the official ClassEval suite and per-test-class checks."""

    valid, reason = validate_class_code(code, str(entry.get("class_name", "")))
    if not valid:
        return {
            "all_passed": False,
            "test_count": 0,
            "pass_count": 0,
            "method_pass_rates": {},
            "trusted_failures": [],
            "error_type": "syntax_error",
            "stdout": "",
            "stderr": reason,
        }
    runner = _load_evaluator()
    method_to_id = {
        name: f"M{index + 1}"
        for index, name in enumerate(_method_names(entry))
    }
    return runner.run_class_tests(
        dict(entry),
        code,
        method_to_id=method_to_id,
        timeout=timeout,
        memory_limit_mb=memory_limit_mb,
    )


def failure_summary(result: Mapping[str, Any], *, limit: int = 8000) -> str:
    """Compact, model-friendly summary of a ClassEval failure."""

    rates = result.get("method_pass_rates") or {}
    failed_methods = [name for name, passed in rates.items() if not passed]
    lines = [
        f"ClassEval result: {result.get('pass_count', 0)}/{len(rates)} method groups passed.",
        "Failed methods: " + (", ".join(failed_methods) if failed_methods else "(none)"),
    ]
    failures = result.get("trusted_failures") or []
    if failures:
        lines.append("Trusted failing test classes:")
        lines.extend(json.dumps(item, ensure_ascii=False) for item in failures[:20])
    for key in ("stderr", "stdout"):
        value = str(result.get(key) or "").strip()
        if value:
            lines.append(f"{key}:\n{value[-4000:]}")
    return "\n".join(lines)[:limit]


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
        handle.flush()


def load_jsonl_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if item.get("task_id"):
                    records[str(item["task_id"])] = item
    return records


def aggregate_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(records)
    completed = [item for item in values if item.get("status") == "completed"]
    passed = [
        item
        for item in completed
        if (item.get("final_tests") or item.get("classeval_test_results") or {}).get("all_passed")
    ]
    method_total = 0
    method_passed = 0
    for item in completed:
        tests = item.get("final_tests") or item.get("classeval_test_results") or {}
        rates = tests.get("method_pass_rates") or {}
        method_total += len(rates)
        method_passed += sum(bool(value) for value in rates.values())
    return {
        "task_count": len(values),
        "completed": len(completed),
        "problem_passed": len(passed),
        "problem_accuracy": (len(passed) / len(values)) if values else 0.0,
        "method_passed": method_passed,
        "method_total": method_total,
        "method_accuracy": (method_passed / method_total) if method_total else 0.0,
    }


@dataclass
class ProviderConfig:
    provider: str
    api_key: str
    base_url: str
    model: str


def configure_provider(env_path: Path, provider: str = "auto") -> ProviderConfig:
    """Resolve a DeepSeek/Qwen OpenAI-compatible endpoint from ``.env``."""

    read_env_file(env_path.expanduser().resolve())
    requested = provider.lower()
    candidates = [requested] if requested != "auto" else ["deepseek", "qwen"]
    for name in candidates:
        upper = name.upper()
        api_key = (os.getenv(f"{upper}_API") or os.getenv(f"{upper}_API_KEY") or "").strip()
        base_url = (os.getenv(f"{upper}_BASE_URL") or os.getenv(f"{upper}_API_BASE") or "").strip()
        model = (os.getenv(f"{upper}_MODEL") or "").strip()
        if api_key and base_url and model:
            if name == "qwen" and model.lower() == "qwen-3.6-plus" and "dashscope.aliyuncs.com" in base_url.lower():
                model = "qwen3.6-plus"
            return ProviderConfig(name, api_key, base_url, model)
    names = ", ".join(candidates)
    raise ValueError(f"missing API/model/base URL for provider(s): {names}")


class CompatibleChatModel:
    """Small model client matching the upstream MapCoder protocol."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        max_tokens: int = 32768,
        timeout: int = 300,
        api_attempts: int = 2,
        thinking: str = "enabled",
    ) -> None:
        self.config = config
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_attempts = api_attempts
        self.thinking = thinking
        self.calls = 0
        self.requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.served_models: set[str] = set()

    @property
    def model(self) -> str:
        return self.config.model

    def _url(self) -> str:
        url = self.config.base_url.rstrip("/")
        return url if url.endswith("/chat/completions") else url + "/chat/completions"

    def prompt(self, processed_input: list[dict[str, str]]) -> tuple[str, int, int]:
        if self.max_tokens <= 0 or self.api_attempts <= 0:
            raise ValueError("max_tokens and api_attempts must be positive")
        if self.config.provider == "qwen":
            body: dict[str, Any] = {
                "model": self.config.model,
                "messages": processed_input,
                "temperature": 0,
                "top_p": 0.95,
                "max_completion_tokens": self.max_tokens,
                "enable_thinking": self.thinking == "enabled",
                "stream": False,
            }
        else:
            body = {
                "model": self.config.model,
                "messages": processed_input,
                "temperature": 0,
                "top_p": 0.95,
                "max_tokens": self.max_tokens,
                "thinking": {"type": self.thinking},
                "stream": False,
            }
        payload_bytes = json.dumps(body).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.api_attempts):
            self.requests += 1
            request = urllib.request.Request(
                self._url(),
                data=payload_bytes,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_payload = json.load(response)
                last_error = None
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read(1000).decode("utf-8", errors="replace")
                last_error = APIRequestError(f"{self.config.provider} HTTP {exc.code}: {detail}")
                if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt + 1 >= self.api_attempts:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = APIRequestError(f"{self.config.provider} connection failed: {exc}")
                if attempt + 1 >= self.api_attempts:
                    raise last_error from exc
            time.sleep(min(2 ** attempt, 30))
        if last_error is not None:
            raise last_error

        choice = (response_payload.get("choices") or [{}])[0]
        usage = response_payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        self.calls += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        if response_payload.get("model"):
            self.served_models.add(str(response_payload["model"]))
        if choice.get("finish_reason") == "length":
            raise OutputTruncatedError(
                f"{self.config.provider} response reached max tokens {self.max_tokens}"
            )
        message = choice.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise APIRequestError("provider returned empty content")
        return content, prompt_tokens, completion_tokens


class CodeSimChatModel:
    """Adapt :class:`CompatibleChatModel` to CodeSIM's ``(text, details)`` API."""

    def __init__(self, raw: CompatibleChatModel) -> None:
        self.raw = raw

    @property
    def calls(self) -> int:
        return self.raw.calls

    @property
    def prompt_tokens(self) -> int:
        return self.raw.prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self.raw.completion_tokens

    @property
    def served_models(self) -> set[str]:
        return self.raw.served_models

    def prompt(
        self,
        processed_input: list[dict[str, str]],
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
    ) -> tuple[str, dict[str, Any]]:
        del frequency_penalty, presence_penalty
        started = time.monotonic()
        before = self.raw.calls
        content, prompt_tokens, completion_tokens = self.raw.prompt(processed_input)
        return content, {
            "api_calls": self.raw.calls - before,
            "taken_time": time.monotonic() - started,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "details": [],
        }
