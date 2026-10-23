"""Adapters for APPS function-style (for example LeetCode) test cases.

The benchmark mixes stdin/stdout problems with records containing ``fn_name``
and argument arrays.  Treating the latter as one nested stdin value makes a
correct ``Solution.method`` look like a malformed output.  These helpers keep
the adapter deterministic and let both the final verifier and the APPS runner
execute the same callable contract.
"""

from __future__ import annotations

import ast
import json
from typing import Any


def _decode_nested(value: Any) -> Any:
    """Decode JSON/Python string wrappers used by APPS function inputs."""
    if isinstance(value, list):
        return [_decode_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_decode_nested(item) for item in value)
    if not isinstance(value, str):
        return value
    text = value.strip()
    if len(text) < 2:
        return value
    for parser in (json.loads, ast.literal_eval):
        try:
            decoded = parser(text)
        except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if decoded != value:
            # A quoted scalar such as ``"1.01"`` is already the intended
            # string argument.  Do not parse that resulting string again as a
            # number; version-comparison tasks depend on preserving it.
            if isinstance(decoded, str):
                return decoded
            return _decode_nested(decoded)
    return value


def callable_arguments(raw_input: Any) -> list[Any]:
    """Return positional arguments from one APPS function-style input."""
    decoded = _decode_nested(raw_input)
    if isinstance(decoded, list):
        return decoded
    if isinstance(decoded, tuple):
        return list(decoded)
    return [decoded]


def callable_stdin(arguments: list[Any]) -> str:
    """Encode arguments for a generated stdin adapter."""
    return " ".join(json.dumps(argument, ensure_ascii=False) for argument in arguments)


def build_callable_driver(
    final_code: str,
    function_name: str,
    arguments: list[Any],
) -> str:
    """Build a sandbox driver that calls ``Solution.function_name``.

    The candidate is executed under a non-main module name so an optional
    stdin adapter in the candidate cannot run before the method call.
    """
    source_literal = repr(str(final_code or ""))
    args_literal = repr(arguments)
    return f"""import json

_candidate_source = {source_literal}
_candidate_ns = {{"__name__": "__auditcoder_candidate__"}}
exec(compile(_candidate_source, "candidate.py", "exec"), _candidate_ns)
_solution_cls = _candidate_ns.get("Solution")
if _solution_cls is None:
    raise RuntimeError("function-style contract requires a Solution class")
_solution = _solution_cls()
_result = getattr(_solution, {function_name!r})(*{args_literal})
print(json.dumps(_result, ensure_ascii=False, separators=(",", ":"), default=str))
"""


__all__ = ["build_callable_driver", "callable_arguments", "callable_stdin"]
