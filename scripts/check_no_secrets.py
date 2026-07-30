#!/usr/bin/env python3
"""Basic fail-fast scan for secrets in tracked and unignored source files."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


TOKEN_PATTERNS = {
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "OpenAI-style token": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
}
PRIVATE_KEY_MARKER = "-----BEGIN " + "PRIVATE KEY-----"
QUOTED_ASSIGNMENT = re.compile(
    r"""(?ix)
    (?<![A-Za-z0-9_])["']?
    (?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password)
    ["']?(?![A-Za-z0-9_])[ \t]*[:=][ \t]*["']([^"'\r\n]{8,})["']
    """
)
ENV_ASSIGNMENT = re.compile(
    r"(?m)^[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD)[ \t]*=[ \t]*([^\s#]+)"
)
PLACEHOLDER_FRAGMENTS = {
    "changeme",
    "dummy",
    "example",
    "fake",
    "placeholder",
    "test",
    "your_",
    "<",
    "${",
}


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().strip("\"'").lower()
    return not normalized or any(
        fragment in normalized for fragment in PLACEHOLDER_FRAGMENTS
    )


def _candidate_files(root: Path) -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=root,
        capture_output=True,
        check=True,
    )
    return [
        root / relative.decode("utf-8")
        for relative in result.stdout.split(b"\0")
        if relative
    ]


def scan(root: Path) -> list[str]:
    findings: list[str] = []
    for path in _candidate_files(root):
        if not path.is_file():
            continue
        payload = path.read_bytes()
        if b"\0" in payload:
            continue
        text = payload.decode("utf-8", errors="replace")
        relative = path.relative_to(root)

        if PRIVATE_KEY_MARKER in text:
            findings.append(f"{relative}: private key block")
        for label, pattern in TOKEN_PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{relative}:{line}: {label}")
        for pattern in (QUOTED_ASSIGNMENT, ENV_ASSIGNMENT):
            for match in pattern.finditer(text):
                if _is_placeholder(match.group(1)):
                    continue
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{relative}:{line}: non-placeholder credential assignment")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    root = args.root.resolve()
    findings = scan(root)
    if findings:
        print("\n".join(findings))
        return 1
    print("No high-confidence secrets found in tracked or unignored files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
