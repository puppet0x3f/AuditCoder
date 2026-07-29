#!/usr/bin/env python3
"""Prepare clean base-commit repositories for portable Micro-5 evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], cwd: Path | None = None) -> None:
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {command}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "dataset_manifest.json").read_text())
    args.cache_root.mkdir(parents=True, exist_ok=True)

    prepared = {}
    for task in manifest["tasks"]:
        repo = task["repo"]
        slug = repo.replace("/", "__")
        destination = args.cache_root / slug
        if not (destination / ".git").is_dir():
            run(["git", "init", "-q", str(destination)])
            run(
                ["git", "remote", "add", "origin", f"https://github.com/{repo}.git"],
                cwd=destination,
            )
        run(
            [
                "git",
                "fetch",
                "-q",
                "--depth",
                "1",
                "origin",
                task["base_commit"],
            ],
            cwd=destination,
        )
        run(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], cwd=destination)
        run(["git", "reset", "-q", "--hard", task["base_commit"]], cwd=destination)
        run(["git", "clean", "-q", "-fdx"], cwd=destination)
        prepared[repo] = str(destination.resolve())
        print(f"{task['task_id']}: {destination}", flush=True)

    output = args.cache_root / "repo_cache.json"
    output.write_text(json.dumps(prepared, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
