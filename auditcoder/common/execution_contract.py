"""Explicit execution-contract helpers shared by planning, assembly, and judging.

The contract is embedded in raw problem text by a caller that owns the runtime
workspace.  AuditCoder defaults to its historical stdin/stdout behavior when no
contract is present, so existing APPS entrypoints remain unchanged.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


CONTRACT_START = "<AUDITCODER_EXECUTION_CONTRACT>"
CONTRACT_END = "</AUDITCODER_EXECUTION_CONTRACT>"
_CONTRACT_PATTERN = re.compile(
    rf"{re.escape(CONTRACT_START)}\s*(\{{.*?\}})\s*{re.escape(CONTRACT_END)}",
    re.DOTALL,
)
SUPPORTED_IO_MODES = {"stdin_stdout", "file"}


def _normalize_relative_paths(value: Any, *, field_name: str) -> list[str]:
    normalized: list[str] = []
    for item in value or []:
        relative = Path(str(item))
        if relative.is_absolute() or ".." in relative.parts or str(relative) in {
            "",
            ".",
        }:
            raise ValueError(f"unsafe {field_name} path: {item!r}")
        rendered = relative.as_posix()
        if rendered not in normalized:
            normalized.append(rendered)
    return normalized


def normalize_execution_contract(value: Any) -> dict[str, Any]:
    """Validate and normalize a serialized execution contract.

    An absent contract deliberately resolves to the legacy stdin/stdout mode.
    File outputs must be relative to one absolute application directory; this
    prevents a prompt-supplied path from widening cleanup or validation scope.
    """
    if not isinstance(value, dict) or not value:
        return {
            "schema_version": "1.0",
            "io_mode": "stdin_stdout",
            "explicit": False,
        }

    io_mode = str(value.get("io_mode", "stdin_stdout")).strip()
    if io_mode not in SUPPORTED_IO_MODES:
        raise ValueError(f"unsupported execution io_mode: {io_mode!r}")

    normalized: dict[str, Any] = {
        "schema_version": str(value.get("schema_version", "1.0")),
        "io_mode": io_mode,
        "explicit": bool(value.get("explicit", True)),
    }
    if io_mode == "stdin_stdout":
        return normalized

    raw_app_dir = str(value.get("app_dir", "")).strip()
    app_dir = Path(raw_app_dir)
    if not raw_app_dir or not app_dir.is_absolute():
        raise ValueError("file execution contract requires an absolute app_dir")
    normalized["app_dir"] = str(app_dir)
    normalized["stdin"] = "closed_eof"

    required_inputs = _normalize_relative_paths(
        value.get("required_inputs", []),
        field_name="required input",
    )
    if required_inputs:
        normalized["required_inputs"] = required_inputs

    required_outputs = _normalize_relative_paths(
        value.get("required_outputs", []),
        field_name="required output",
    )
    if not required_outputs:
        raise ValueError("file execution contract requires required_outputs")
    normalized["required_outputs"] = required_outputs
    return normalized


def parse_execution_contract(problem_raw: str) -> dict[str, Any]:
    """Parse a machine-readable contract embedded in raw problem text."""
    match = _CONTRACT_PATTERN.search(problem_raw or "")
    if not match:
        return normalize_execution_contract(None)
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid AuditCoder execution contract JSON: {exc}") from exc
    return normalize_execution_contract(value)


def render_execution_contract(value: dict[str, Any]) -> str:
    """Render a normalized contract for inclusion in a model-visible prompt."""
    normalized = normalize_execution_contract(value)
    payload = json.dumps(normalized, ensure_ascii=False, indent=2)
    return f"{CONTRACT_START}\n{payload}\n{CONTRACT_END}"


def resolve_required_output_paths(
    contract: dict[str, Any],
) -> list[tuple[str, Path]]:
    """Resolve validated file outputs beneath the contract application root."""
    normalized = normalize_execution_contract(contract)
    if normalized["io_mode"] != "file":
        return []
    app_dir = Path(normalized["app_dir"]).resolve()
    resolved: list[tuple[str, Path]] = []
    for relative in normalized["required_outputs"]:
        path = (app_dir / relative).resolve()
        if not path.is_relative_to(app_dir):
            raise ValueError(f"required output escapes app_dir: {relative!r}")
        resolved.append((relative, path))
    return resolved


def resolve_required_input_paths(
    contract: dict[str, Any],
) -> list[tuple[str, Path]]:
    """Resolve validated file inputs beneath the contract application root."""
    normalized = normalize_execution_contract(contract)
    if normalized["io_mode"] != "file":
        return []
    app_dir = Path(normalized["app_dir"]).resolve()
    resolved: list[tuple[str, Path]] = []
    for relative in normalized.get("required_inputs", []):
        path = (app_dir / relative).resolve()
        if not path.is_relative_to(app_dir):
            raise ValueError(f"required input escapes app_dir: {relative!r}")
        resolved.append((relative, path))
    return resolved
