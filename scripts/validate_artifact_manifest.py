#!/usr/bin/env python3
"""Validate a paper-artifact manifest and every retained file checksum."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


REQUIRED_FILE_ROLES = {
    "task_ids",
    "run_manifest",
    "raw_output",
    "audit_record",
    "repair_record",
    "aggregate",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_manifest(manifest_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read manifest: {exc}"]
    if not isinstance(manifest, dict):
        return ["manifest root must be a JSON object"]

    if manifest.get("schema_version") != "1.0":
        errors.append("schema_version must be '1.0'")
    if not _non_empty_string(manifest.get("artifact_name")):
        errors.append("artifact_name must be a non-empty string")

    source = manifest.get("source")
    if not isinstance(source, dict):
        errors.append("source must be an object")
    else:
        commit = source.get("commit")
        tree_sha256 = source.get("tree_sha256")
        if not (
            isinstance(commit, str)
            and len(commit) == 40
            and all(char in "0123456789abcdef" for char in commit.lower())
        ):
            errors.append("source.commit must be a 40-character hexadecimal hash")
        if not (
            isinstance(tree_sha256, str)
            and len(tree_sha256) == 64
            and all(char in "0123456789abcdef" for char in tree_sha256.lower())
        ):
            errors.append(
                "source.tree_sha256 must be a 64-character hexadecimal hash"
            )

    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        errors.append("dataset must be an object")
    else:
        for field in ("name", "revision"):
            if not _non_empty_string(dataset.get(field)):
                errors.append(f"dataset.{field} must be a non-empty string")
        task_ids = dataset.get("fixed_task_ids")
        if not (
            isinstance(task_ids, list)
            and task_ids
            and all(_non_empty_string(task_id) for task_id in task_ids)
        ):
            errors.append("dataset.fixed_task_ids must be a non-empty string list")
        elif len(task_ids) != len(set(task_ids)):
            errors.append("dataset.fixed_task_ids must not contain duplicates")

    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        errors.append("models must be a non-empty list")
    else:
        for index, model in enumerate(models):
            if not isinstance(model, dict):
                errors.append(f"models[{index}] must be an object")
                continue
            for field in ("provider", "provider_visible_name"):
                if not _non_empty_string(model.get(field)):
                    errors.append(
                        f"models[{index}].{field} must be a non-empty string"
                    )
            if not isinstance(model.get("decoding"), dict):
                errors.append(f"models[{index}].decoding must be an object")

    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        errors.append("runs must be a non-empty list")
    else:
        run_ids: list[str] = []
        for index, run in enumerate(runs):
            if not isinstance(run, dict):
                errors.append(f"runs[{index}] must be an object")
                continue
            run_id = run.get("id")
            if not _non_empty_string(run_id):
                errors.append(f"runs[{index}].id must be a non-empty string")
            else:
                run_ids.append(run_id)
            run_task_ids = run.get("task_ids")
            if not (
                isinstance(run_task_ids, list)
                and run_task_ids
                and all(_non_empty_string(task_id) for task_id in run_task_ids)
            ):
                errors.append(f"runs[{index}].task_ids must be a non-empty string list")
        if len(run_ids) != len(set(run_ids)):
            errors.append("run IDs must be unique")

    files = manifest.get("files")
    observed_roles: set[str] = set()
    observed_paths: set[str] = set()
    base = manifest_path.parent.resolve()
    if not isinstance(files, list) or not files:
        errors.append("files must be a non-empty list")
    else:
        for index, entry in enumerate(files):
            if not isinstance(entry, dict):
                errors.append(f"files[{index}] must be an object")
                continue
            relative = entry.get("path")
            role = entry.get("role")
            expected_hash = entry.get("sha256")
            expected_bytes = entry.get("bytes")

            if not _non_empty_string(role):
                errors.append(f"files[{index}].role must be a non-empty string")
            else:
                observed_roles.add(role)
            if not _non_empty_string(relative):
                errors.append(f"files[{index}].path must be a non-empty string")
                continue
            if relative in observed_paths:
                errors.append(f"duplicate file path: {relative}")
                continue
            observed_paths.add(relative)

            pure_path = PurePosixPath(relative)
            if pure_path.is_absolute() or ".." in pure_path.parts:
                errors.append(f"unsafe relative file path: {relative}")
                continue
            candidate = manifest_path.parent.joinpath(*pure_path.parts)
            try:
                resolved = candidate.resolve(strict=True)
            except OSError as exc:
                errors.append(f"missing artifact file {relative}: {exc}")
                continue
            if not resolved.is_relative_to(base):
                errors.append(f"artifact path escapes manifest directory: {relative}")
                continue
            if candidate.is_symlink() or not resolved.is_file():
                errors.append(f"artifact path must be a regular non-symlink file: {relative}")
                continue
            if not (
                isinstance(expected_hash, str)
                and len(expected_hash) == 64
                and all(
                    char in "0123456789abcdef"
                    for char in expected_hash.lower()
                )
            ):
                errors.append(f"files[{index}].sha256 must be a hexadecimal SHA-256")
            elif _sha256(resolved) != expected_hash.lower():
                errors.append(f"checksum mismatch: {relative}")
            if not isinstance(expected_bytes, int) or expected_bytes < 0:
                errors.append(f"files[{index}].bytes must be a non-negative integer")
            elif resolved.stat().st_size != expected_bytes:
                errors.append(f"size mismatch: {relative}")

    missing_roles = REQUIRED_FILE_ROLES - observed_roles
    if missing_roles:
        errors.append(
            "files are missing required roles: " + ", ".join(sorted(missing_roles))
        )
    return errors


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="auditcoder-manifest-") as temp_dir:
        root = Path(temp_dir)
        entries = []
        for role in sorted(REQUIRED_FILE_ROLES):
            path = root / f"{role}.json"
            path.write_text(json.dumps({"role": role}) + "\n", encoding="utf-8")
            entries.append(
                {
                    "path": path.name,
                    "role": role,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
        manifest = {
            "schema_version": "1.0",
            "artifact_name": "AuditCoder validator self-test",
            "source": {"commit": "0" * 40, "tree_sha256": "1" * 64},
            "dataset": {
                "name": "self-test",
                "revision": "v1",
                "fixed_task_ids": ["task-1"],
            },
            "models": [
                {
                    "provider": "offline",
                    "provider_visible_name": "self-test",
                    "decoding": {},
                }
            ],
            "runs": [{"id": "run-1", "task_ids": ["task-1"]}],
            "files": entries,
        }
        manifest_path = root / "artifact_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        if errors := validate_manifest(manifest_path):
            raise AssertionError(f"valid self-test manifest failed: {errors}")

        (root / "raw_output.json").write_text("tampered\n", encoding="utf-8")
        errors = validate_manifest(manifest_path)
        if not any("checksum mismatch: raw_output.json" in error for error in errors):
            raise AssertionError("tampered artifact was not rejected")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print(json.dumps({"self_test": "passed"}))
        return 0
    if args.manifest is None:
        parser.error("provide a manifest path or --self-test")

    errors = validate_manifest(args.manifest.resolve())
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "valid": not errors,
                "errors": errors,
            },
            indent=2,
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
