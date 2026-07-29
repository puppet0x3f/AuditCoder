#!/usr/bin/env python3
"""Build the frozen SWE-bench Verified Micro-5 dataset.

The public prompt contains the issue and a small oracle-localized source region.
Gold patches, test patches, and official test identifiers remain under
``tasks/*/private`` and are never passed to a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_REVISION = "91aa3ed51b709be6457e12d00300a6a596d4c6a3"
HARNESS_REVISION = "f7bbbb2ccdf479001d6467c9e34af59e44a840f9"
PARQUET_SHA256 = "43ed5a3d1d98da36472c1ade65ddd2085d7b4ff694fcaf6a023a07c5c1f32f21"

TASKS: list[dict[str, Any]] = [
    {
        "task_id": "django__django-11206",
        "repo_key": "django",
        "scenario": "Web framework numeric formatting",
        "oracle_files": ["django/utils/numberformat.py"],
        "context_ranges": {"django/utils/numberformat.py": [1, 79]},
        "test_kind": "django",
        "test_file": "tests/utils_tests/test_numberformat.py",
    },
    {
        "task_id": "psf__requests-6028",
        "repo_key": "requests",
        "scenario": "HTTP proxy URL authentication",
        "oracle_files": ["requests/utils.py"],
        "context_ranges": {"requests/utils.py": [930, 1005]},
        "test_kind": "pytest",
        "test_file": "tests/test_utils.py",
    },
    {
        "task_id": "pytest-dev__pytest-10051",
        "repo_key": "pytest",
        "scenario": "Testing framework log capture state",
        "oracle_files": ["src/_pytest/logging.py"],
        "context_ranges": {"src/_pytest/logging.py": [300, 470]},
        "test_kind": "pytest",
        "test_file": "testing/logging/test_fixture.py",
    },
    {
        "task_id": "sphinx-doc__sphinx-11445",
        "repo_key": "sphinx",
        "scenario": "Documentation parser section roles",
        "oracle_files": ["sphinx/util/rst.py"],
        "context_ranges": {"sphinx/util/rst.py": [1, 110]},
        "test_kind": "pytest",
        "test_file": "tests/test_util_rst.py",
    },
    {
        "task_id": "sympy__sympy-20154",
        "repo_key": "sympy",
        "scenario": "Symbolic-math partition iterator ownership",
        "oracle_files": ["sympy/utilities/iterables.py"],
        "context_ranges": {"sympy/utilities/iterables.py": [1700, 1860]},
        "test_kind": "pytest-bare-names",
        "test_file": "sympy/utilities/tests/test_iterables.py",
    },
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def git_show(repo: Path, commit: str, relative: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "show", f"{commit}:{relative}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"git show failed for {commit}:{relative}: {completed.stderr}"
        )
    return completed.stdout


def extract_context(source: str, start: int, end: int) -> str:
    lines = source.splitlines()
    selected = lines[start - 1 : end]
    width = len(str(end))
    return "\n".join(
        f"{line_no:>{width}} | {line}"
        for line_no, line in enumerate(selected, start=start)
    )


def code_patch_stats(patch: str) -> dict[str, Any]:
    files: list[str] = []
    implementation_lines = 0
    current = ""
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            current = line.split(" b/", 1)[-1]
            files.append(current)
        elif (
            current
            and line.startswith(("+", "-"))
            and not line.startswith(("+++", "---"))
            and "/test" not in f"/{current.lower()}"
            and not Path(current).name.lower().startswith("test")
        ):
            implementation_lines += 1
    return {
        "files": list(dict.fromkeys(files)),
        "implementation_changed_lines": implementation_lines,
    }


def public_problem(
    *,
    row: dict[str, Any],
    task: dict[str, Any],
    contexts: dict[str, str],
) -> str:
    rendered_contexts = []
    for relative in task["oracle_files"]:
        start, end = task["context_ranges"][relative]
        rendered_contexts.append(
            f"### `{relative}` (base lines {start}-{end})\n\n"
            f"```python\n{contexts[relative]}\n```"
        )
    return (
        "# SWE-bench Verified Micro-5 repository patch task\n\n"
        f"- Instance: `{task['task_id']}`\n"
        f"- Repository: `{row['repo']}`\n"
        f"- Base commit: `{row['base_commit']}`\n"
        f"- Scenario: {task['scenario']}\n"
        f"- Allowed implementation files: "
        + ", ".join(f"`{item}`" for item in task["oracle_files"])
        + "\n\n"
        "## Output contract\n\n"
        "Return one complete Python patcher program. At evaluation time the "
        "environment variable `REPO_ROOT` points to a clean checkout at the "
        "base commit. The program must edit only the allowed implementation "
        "files under that directory, then exit successfully. Use normal "
        "`pathlib` file reads and deterministic source transformations. Do "
        "not invoke git, access the network, modify tests, or embed a unified "
        "diff. The evaluator captures the resulting repository diff and then "
        "runs hidden SWE-bench tests. Return only the patcher source in a "
        "single Python code block.\n\n"
        "This is an oracle-region code-generation setting: the relevant source "
        "region is supplied, so the experiment evaluates patch synthesis, not "
        "repository-wide fault localization.\n\n"
        "## Upstream issue\n\n"
        f"{row['problem_statement'].strip()}\n\n"
        "## Base source context\n\n"
        + "\n\n".join(rendered_contexts)
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help="Base repository checkout used to extract frozen source context.",
    )
    args = parser.parse_args()

    if sha256_file(args.parquet) != PARQUET_SHA256:
        raise SystemExit("pinned parquet SHA-256 mismatch")
    repo_map: dict[str, Path] = {}
    for item in args.repo:
        key, separator, value = item.partition("=")
        if not separator:
            raise SystemExit(f"invalid --repo value: {item}")
        repo_map[key] = Path(value).resolve()
    missing = {task["repo_key"] for task in TASKS} - set(repo_map)
    if missing:
        raise SystemExit(f"missing repository checkouts: {sorted(missing)}")

    frame = pd.read_parquet(args.parquet).set_index("instance_id")
    manifest_tasks = []
    public_hashes: dict[str, str] = {}
    for task in TASKS:
        task_id = task["task_id"]
        if task_id not in frame.index:
            raise RuntimeError(f"missing pinned task: {task_id}")
        row = frame.loc[task_id].to_dict()
        repo = repo_map[task["repo_key"]]
        actual_commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout.strip()
        if actual_commit != row["base_commit"]:
            raise RuntimeError(
                f"{task_id}: checkout {actual_commit} != {row['base_commit']}"
            )

        contexts: dict[str, str] = {}
        base_file_hashes: dict[str, str] = {}
        for relative in task["oracle_files"]:
            source = git_show(repo, row["base_commit"], relative)
            start, end = task["context_ranges"][relative]
            contexts[relative] = extract_context(source, start, end)
            base_file_hashes[relative] = sha256_bytes(source.encode())

        public_dir = ROOT / "tasks" / task_id / "public"
        private_dir = ROOT / "tasks" / task_id / "private"
        problem = public_problem(row=row, task=task, contexts=contexts)
        public_dir.mkdir(parents=True, exist_ok=True)
        (public_dir / "problem.md").write_text(problem)
        public_task = {
            "task_id": task_id,
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "version": row["version"],
            "difficulty": row["difficulty"],
            "scenario": task["scenario"],
            "oracle_context": True,
            "oracle_files": task["oracle_files"],
            "context_ranges": task["context_ranges"],
            "base_file_sha256": base_file_hashes,
        }
        write_json(public_dir / "task.json", public_task)

        fail_to_pass = json.loads(row["FAIL_TO_PASS"])
        pass_to_pass = json.loads(row["PASS_TO_PASS"])
        private_task = {
            **public_task,
            "test_kind": task["test_kind"],
            "test_file": task["test_file"],
            "fail_to_pass": fail_to_pass,
            "pass_to_pass": pass_to_pass,
            "test_patch": row["test_patch"],
            "gold_patch": row["patch"],
            "environment_setup_commit": row["environment_setup_commit"],
            "gold_patch_stats": code_patch_stats(row["patch"]),
        }
        write_json(private_dir / "upstream.json", private_task)

        for path in (public_dir / "problem.md", public_dir / "task.json"):
            public_hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        manifest_tasks.append(
            {
                **public_task,
                "fail_to_pass_count": len(fail_to_pass),
                "pass_to_pass_count": len(pass_to_pass),
                "official_test_count": len(fail_to_pass) + len(pass_to_pass),
                "gold_implementation_changed_lines": private_task[
                    "gold_patch_stats"
                ]["implementation_changed_lines"],
            }
        )

    manifest = {
        "dataset_name": "SWE-bench Verified Micro-5 Oracle-Region",
        "dataset_version": "1.0.0",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "SWE-bench/SWE-bench_Verified",
            "dataset_revision": DATASET_REVISION,
            "dataset_parquet_sha256": PARQUET_SHA256,
            "harness_repository": "SWE-bench/SWE-bench",
            "harness_revision": HARNESS_REVISION,
        },
        "protocol": {
            "setting": "oracle-region patch synthesis",
            "candidate_output": "Python patcher program",
            "hidden_test_feedback": False,
            "task_resolved": "all FAIL_TO_PASS and PASS_TO_PASS tests non-failing",
            "primary_metric": "Resolved@1",
            "official_docker_status": "pending",
            "portable_environment": "conda swe-micro5, Python 3.9",
        },
        "tasks": manifest_tasks,
        "public_artifact_sha256": public_hashes,
    }
    write_json(ROOT / "dataset_manifest.json", manifest)
    print(
        json.dumps(
            {
                "dataset": manifest["dataset_name"],
                "tasks": len(manifest_tasks),
                "official_tests": sum(
                    task["official_test_count"] for task in manifest_tasks
                ),
                "public_artifacts": len(public_hashes),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
