#!/usr/bin/env python3
"""Build the frozen SciCode Blind-Small-8 dataset from pinned upstream files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py


DATASET_ROOT = Path(__file__).resolve().parents[1]
HF_REVISION = "4510f6a6aa27c43fad7b43da2c59602a86e88480"
CODE_REVISION = "e3158ea011d4235245a547460d3688d7ccbf9900"
SELECTION = {
    "13.1": {
        "domain": "computational_electromagnetics",
        "scenario": "Maxwell solver finite-difference operator",
        "selection_note": "independent first substep; electromagnetic simulation primitive",
    },
    "14.1": {
        "domain": "biophysics",
        "scenario": "Brownian motion in optical tweezers",
        "selection_note": "independent first substep; stochastic dynamics",
    },
    "25.1": {
        "domain": "ecology",
        "scenario": "Chemostat species growth",
        "selection_note": "independent first substep; ecological dynamics",
    },
    "28.1": {
        "domain": "optics",
        "scenario": "Gaussian beam propagation",
        "selection_note": "independent first substep; Fourier optics",
    },
    "35.1": {
        "domain": "condensed_matter",
        "scenario": "Quantum-dot absorption wavelength",
        "selection_note": "independent first substep; nanophotonics",
    },
    "37.1": {
        "domain": "optical_engineering",
        "scenario": "Paraxial rays through a doublet lens",
        "selection_note": "independent first substep; optical design",
    },
    "45.1": {
        "domain": "thermal_engineering",
        "scenario": "Finite-difference heat-equation grid",
        "selection_note": "independent first substep; heat-transfer simulation",
    },
    "77.1": {
        "domain": "molecular_dynamics",
        "scenario": "Periodic-boundary coordinate wrapping",
        "selection_note": "independent first substep; molecular simulation",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def public_record(problem: dict[str, Any], substep: dict[str, Any]) -> dict[str, Any]:
    return {
        "suite": "SciCode Blind-Small-8",
        "split": "test",
        "source_revision": HF_REVISION,
        "task_id": substep["step_number"],
        "problem_id": problem["problem_id"],
        "problem_name": problem["problem_name"],
        "domain": SELECTION[substep["step_number"]]["domain"],
        "scenario": SELECTION[substep["step_number"]]["scenario"],
        "required_dependencies": problem["required_dependencies"],
        "step_description_prompt": substep["step_description_prompt"],
        "function_header": substep["function_header"],
        "return_line": substep["return_line"],
        "step_background": substep["step_background"],
        "entry_point": substep["function_header"].split("def ", 1)[1].split("(", 1)[0],
        "test_count": len(substep["test_cases"]),
    }


def render_problem(task: dict[str, Any]) -> str:
    return f"""# SciCode Blind-Small-8 task {task['task_id']}

Source split: `SciCode test`  
Domain: `{task['domain']}`  
Scenario: {task['scenario']}

Write a complete Python module that implements the requested function. Keep the
function name and signature exactly as specified. Include all necessary imports.
Do not read stdin, write files, access the network, or include demonstration
calls. Return only executable Python source.

## Required dependencies

```python
{task['required_dependencies']}
```

## Scientific task

{task['step_description_prompt']}

## Required interface

```python
{task['function_header']}

{task['return_line']}
```

## Scientist-provided background

{task['step_background'] or 'No additional background was supplied upstream.'}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems-test", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    args = parser.parse_args()

    problems = read_jsonl(args.problems_test)
    selected: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for problem in problems:
        for substep in problem["sub_steps"]:
            step_id = substep["step_number"]
            if step_id in SELECTION:
                selected[step_id] = (problem, substep)
    missing = set(SELECTION) - set(selected)
    if missing:
        raise SystemExit(f"selected steps missing from source: {sorted(missing)}")

    tasks_root = DATASET_ROOT / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)
    manifest_tasks = []
    for task_id in SELECTION:
        problem, substep = selected[task_id]
        public = public_record(problem, substep)
        task_root = tasks_root / task_id
        public_dir = task_root / "public"
        private_dir = task_root / "private"
        public_dir.mkdir(parents=True, exist_ok=True)
        private_dir.mkdir(parents=True, exist_ok=True)
        (public_dir / "task.json").write_text(
            json.dumps(public, ensure_ascii=False, indent=2) + "\n"
        )
        (public_dir / "problem.md").write_text(render_problem(public))
        private = {
            "task_id": task_id,
            "entry_point": public["entry_point"],
            "required_dependencies": problem["required_dependencies"],
            "official_test_cases": substep["test_cases"],
            "official_test_count": len(substep["test_cases"]),
            "target_group": task_id,
            "comparator": "official SciCode assertions; numpy.allclose defaults",
        }
        (private_dir / "task_spec.json").write_text(
            json.dumps(private, ensure_ascii=False, indent=2) + "\n"
        )
        manifest_tasks.append(
            {
                **public,
                "outputs": [public["entry_point"]],
                "selection_note": SELECTION[task_id]["selection_note"],
                "public_problem": f"tasks/{task_id}/public/problem.md",
                "public_metadata": f"tasks/{task_id}/public/task.json",
                "private_spec": f"tasks/{task_id}/private/task_spec.json",
                "target_group": task_id,
            }
        )

    private_root = DATASET_ROOT / "private"
    private_root.mkdir(parents=True, exist_ok=True)
    target_path = private_root / "targets.h5"
    with h5py.File(args.test_data, "r") as source, h5py.File(target_path, "w") as dest:
        for task_id in SELECTION:
            if task_id not in source:
                raise SystemExit(f"target group {task_id} missing from upstream HDF5")
            source.copy(task_id, dest)
        dest.attrs["source"] = "SciCode official numeric test_data.h5"
        dest.attrs["source_revision"] = HF_REVISION
        dest.attrs["selected_groups"] = json.dumps(list(SELECTION))

    license_source = args.upstream_repo / "LICENSE"
    shutil.copy2(license_source, DATASET_ROOT / "LICENSE-UPSTREAM")
    manifest = {
        "schema_version": "1.0",
        "dataset_name": "SciCode Blind-Small-8",
        "dataset_version": "1.0.0",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "SciCode1/SciCode",
            "dataset_url": "https://huggingface.co/datasets/SciCode1/SciCode",
            "dataset_revision": HF_REVISION,
            "revision": HF_REVISION,
            "code_url": "https://github.com/scicode-bench/SciCode",
            "code_revision": CODE_REVISION,
            "license": "Apache-2.0",
            "split": "test",
        },
        "protocol": {
            "prompt_setting": "with scientist-provided substep background",
            "generation_attempts_per_task": 1,
            "hidden_tests_visible_to_generator": False,
            "hidden_test_feedback": False,
            "resolved_definition": "all three official numeric tests pass",
            "primary_metric": "Resolved@1",
            "secondary_metrics": [
                "official_tests_passed",
                "build_import_rate",
                "timeout_rate",
                "environment_failure_rate",
            ],
        },
        "selection_frozen_before_runs": True,
        "task_count": len(manifest_tasks),
        "official_test_count": sum(task["test_count"] for task in manifest_tasks),
        "tasks": manifest_tasks,
        "artifacts": {
            "official_target_subset": "private/targets.h5",
            "upstream_target_sha256": sha256_file(args.test_data),
            "target_subset_sha256": sha256_file(target_path),
            "problems_test_sha256": sha256_file(args.problems_test),
        },
    }
    (DATASET_ROOT / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )

    upstream_hashes = {
        "hf/problems_test.jsonl": sha256_file(args.problems_test),
        "google_drive/test_data.h5": sha256_file(args.test_data),
        "github/LICENSE": sha256_file(license_source),
        "github/src/scicode/parse/parse.py": sha256_file(
            args.upstream_repo / "src/scicode/parse/parse.py"
        ),
        "github/src/scicode/compare/cmp.py": sha256_file(
            args.upstream_repo / "src/scicode/compare/cmp.py"
        ),
    }
    (DATASET_ROOT / "UPSTREAM_SHA256SUMS").write_text(
        "\n".join(f"{digest}  {name}" for name, digest in upstream_hashes.items())
        + "\n"
    )
    print(
        json.dumps(
            {
                "dataset_root": str(DATASET_ROOT),
                "tasks": len(manifest_tasks),
                "tests": manifest["official_test_count"],
                "target_subset_bytes": target_path.stat().st_size,
                "target_subset_sha256": manifest["artifacts"]["target_subset_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
