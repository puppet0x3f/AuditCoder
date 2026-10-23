#!/usr/bin/env python3
"""Check distributable files for local identity paths and common secret formats.

Ignored local configuration, outputs, environments, and Git metadata are outside
this content check. Account identity and Git history must be checked separately.
Matched secret values are never printed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}
PATTERNS = {
    "local user path": re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/|[A-Za-z]:[\\/](?:Users|AuditCoder)[\\/]"),
    "API token": re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "author metadata": re.compile(r"(?m)^\s*(?:__author__|author_email|maintainer_email)\s*="),
}


def main() -> int:
    issues = []
    checked = 0
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if any(part in SKIP for part in rel.parts) or not path.is_file():
            continue
        if rel.parts[0] in {"results", "result"} or rel.parts[:2] == ("data", "APPS"):
            continue
        if path.name.startswith(".env") and path.name != ".env.example":
            continue
        if path.name in {".DS_Store", "cookies.json"} or path.suffix in {".pem", ".key"}:
            issues.append(f"{rel}: private or machine-specific file")
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeError:
            issues.append(f"{rel}: unexpected binary file; inspect before distributing")
            continue
        checked += 1
        for label, pattern in PATTERNS.items():
            match = pattern.search(source)
            if match:
                line = source.count("\n", 0, match.start()) + 1
                issues.append(f"{rel}:{line}: {label}")
    if issues:
        print("\n".join(issues))
        return 1
    print(f"Content check passed: {checked} files; no matched identity paths or secret formats.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
