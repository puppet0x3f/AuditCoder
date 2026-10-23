#!/usr/bin/env python3
"""Run the fixed 200-problem APPS batch with one repair and one retest.

This is a small entry point around ``run_apps_200_auditcoder.py``.  It fixes
the two budgets required for the one-retest experiment:

* one bounded AuditCoder repair attempt per pipeline run;
* at most one additional full-pipeline run for an unsuccessful problem.

All other options supported by the underlying runner (provider, case limit,
checkpointing, artifacts, and verbosity) can be passed through unchanged.
"""

from __future__ import annotations

import sys

from run_apps_200_auditcoder import DEFAULT_OUTPUT, main as run_apps_main


def main() -> None:
    user_args = sys.argv[1:]
    if "-h" in user_args or "--help" in user_args:
        print(__doc__.strip())
        print(
            "\nPass-through options: --input, --output, --summary, --env, "
            "--provider, --max-cases, --case-timeout, --resume, "
            "--save-artifacts, --dry-run, and --quiet."
        )
        return

    forbidden = {
        "--count",
        "--offset",
        "--ids",
        "--difficulty",
        "--max-repair-attempts",
        "--problem-retries",
        "--retry-failed-once",
    }
    supplied = {
        argument.split("=", 1)[0]
        for argument in user_args
        if argument.startswith("--")
    }
    conflicts = sorted(supplied & forbidden)
    if conflicts:
        joined = ", ".join(conflicts)
        raise SystemExit(f"fixed one-retry runner does not accept: {joined}")

    if "--output" not in supplied:
        user_args = [
            "--output",
            str(DEFAULT_OUTPUT.with_name("apps_200_auditcoder_one_retry.jsonl")),
            *user_args,
        ]

    sys.argv = [
        sys.argv[0],
        "--count",
        "200",
        "--max-repair-attempts",
        "1",
        "--retry-failed-once",
        *user_args,
    ]
    run_apps_main()


if __name__ == "__main__":
    main()
