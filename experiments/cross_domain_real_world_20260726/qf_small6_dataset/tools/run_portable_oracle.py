#!/usr/bin/env python3
"""Run QF-Small6 oracle and hidden tests in temporary host sandboxes.

This tool does not import AuditCoder. It preserves the test assertions and rewrites
only the absolute `/app` sandbox prefix in temporary copies.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_with_app_path(source: Path, destination: Path, app: Path) -> None:
    text = source.read_text().replace("/app", str(app))
    destination.write_text(text)
    destination.chmod(source.stat().st_mode)


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
) -> dict:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": completed.returncode,
            "duration_seconds": round(time.monotonic() - started, 6),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 6),
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timeout": True,
        }


def parse_junit(path: Path) -> dict:
    if not path.exists():
        return {
            "collected": 0,
            "passed": 0,
            "failed": 0,
            "errors": 1,
            "skipped": 0,
            "junit_present": False,
        }
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    collected = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failed = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    return {
        "collected": collected,
        "passed": collected - failed - errors - skipped,
        "failed": failed,
        "errors": errors,
        "skipped": skipped,
        "junit_present": True,
    }


def run_pytest(
    python: Path,
    test_file: Path,
    junit_path: Path,
    *,
    cwd: Path,
    env: dict[str, str],
) -> dict:
    command = [
        str(python),
        "-m",
        "pytest",
        "-q",
        str(test_file),
        f"--junitxml={junit_path}",
        "--disable-warnings",
    ]
    result = run_command(command, cwd=cwd, env=env, timeout=300)
    result.update(parse_junit(junit_path))
    result["test_sha256"] = sha256(test_file)
    return result


def apply_semantic_tamper(task_id: str, app: Path) -> dict:
    result_path = app / "output/results.json"
    result = json.loads(result_path.read_text())
    if task_id == "earnings-surprise-calculator":
        result["MSFT"]["summary"]["beat_rate"] = -1.0
        mutation = "MSFT.summary.beat_rate=-1"
    elif task_id == "corporate-action-adjustment":
        result["cumulative_price_adjustment_factor"] = 1.0
        mutation = "cumulative_price_adjustment_factor=1"
    elif task_id == "historical-var-data-prep":
        result["var_95_portfolio"] = 0.0
        mutation = "var_95_portfolio=0"
    elif task_id == "interest-rate-cap-floor":
        result["cap_price"] = 0.0
        mutation = "cap_price=0"
    elif task_id == "zero-coupon-bootstrapping":
        first_key = sorted(result["discount_factors"], key=float)[0]
        result["discount_factors"][first_key] = 2.0
        mutation = f"discount_factors[{first_key}]=2"
    elif task_id == "pca-factor-portfolio":
        result["pca_explained_variance_ratio"][0] = 2.0
        mutation = "pca_explained_variance_ratio[0]=2"
    else:
        raise ValueError(f"no negative control for {task_id}")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return {"mutation": mutation, "output_sha256_after": sha256(result_path)}


def run_task(task: dict, python: Path, mode: str) -> dict:
    task_id = task["task_id"]
    base = ROOT / "tasks" / task_id
    with tempfile.TemporaryDirectory(prefix=f"qf-small6-{task_id}-") as temporary:
        sandbox = Path(temporary)
        app = sandbox / "app"
        app.mkdir()
        mount = app / task["data_mount"]
        mount.mkdir(parents=True, exist_ok=True)
        shutil.copytree(base / "public/data", mount, dirs_exist_ok=True)

        bin_dir = sandbox / "bin"
        bin_dir.mkdir()
        (bin_dir / "python3").symlink_to(python)
        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
        env["PYTHONHASHSEED"] = "0"

        oracle = sandbox / "oracle.sh"
        official_test = sandbox / "test_outputs.py"
        addition_test = sandbox / "test_contract_additions.py"
        materialize_with_app_path(base / "private/oracle/solve.sh", oracle, app)
        materialize_with_app_path(
            base / "private/tests/test_outputs.py", official_test, app
        )
        materialize_with_app_path(
            base / "private/tests/test_contract_additions.py", addition_test, app
        )

        oracle_result = run_command(
            ["bash", str(oracle)], cwd=app, env=env, timeout=300
        )
        tamper = None
        if mode == "negative-control" and oracle_result["returncode"] == 0:
            tamper = apply_semantic_tamper(task_id, app)

        official_result = run_pytest(
            python,
            official_test,
            sandbox / "official-junit.xml",
            cwd=app,
            env=env,
        )
        addition_result = run_pytest(
            python,
            addition_test,
            sandbox / "addition-junit.xml",
            cwd=app,
            env=env,
        )
        failures = (
            official_result["failed"]
            + official_result["errors"]
            + addition_result["failed"]
            + addition_result["errors"]
        )
        oracle_pass = (
            oracle_result["returncode"] == 0
            and official_result["returncode"] == 0
            and addition_result["returncode"] == 0
            and failures == 0
        )
        negative_detected = (
            oracle_result["returncode"] == 0
            and tamper is not None
            and failures > 0
        )
        expected_success = oracle_pass if mode == "oracle" else negative_detected
        output_hashes = {}
        for relative in task["outputs"]:
            path = app / relative
            output_hashes[relative] = sha256(path) if path.exists() else None
        return {
            "task_id": task_id,
            "mode": mode,
            "oracle": oracle_result,
            "upstream_tests": official_result,
            "contract_addition_tests": addition_result,
            "tamper": tamper,
            "oracle_pass": oracle_pass,
            "negative_control_detected": negative_detected,
            "expected_success": expected_success,
            "output_hashes": output_hashes,
            "path_rewrite": {
                "from": "/app",
                "to": "<temporary-sandbox>/app",
                "assertions_changed": False,
            },
        }


def package_versions() -> dict[str, str | None]:
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "pytest"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["oracle", "negative-control"], default="oracle"
    )
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    python = args.python.resolve()
    if not python.is_file():
        raise SystemExit(f"python not found: {python}")
    manifest = json.loads((ROOT / "dataset_manifest.json").read_text())
    selected = [
        task
        for task in manifest["tasks"]
        if not args.task or task["task_id"] in set(args.task)
    ]
    missing = set(args.task) - {task["task_id"] for task in selected}
    if missing:
        raise SystemExit(f"unknown task(s): {sorted(missing)}")

    results = []
    for task in selected:
        result = run_task(task, python, args.mode)
        results.append(result)
        print(
            f"{task['task_id']}: expected_success={result['expected_success']} "
            f"official={result['upstream_tests']['passed']}/"
            f"{result['upstream_tests']['collected']} "
            f"additions={result['contract_addition_tests']['passed']}/"
            f"{result['contract_addition_tests']['collected']}"
        )

    report = {
        "dataset": manifest["dataset_name"],
        "source_revision": manifest["source"]["revision"],
        "mode": args.mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runner": "portable-host-path-rewrite",
        "official_harbor_docker": False,
        "python": str(python),
        "packages": package_versions(),
        "tasks_run": len(results),
        "expected_successes": sum(result["expected_success"] for result in results),
        "all_expected_success": all(result["expected_success"] for result in results),
        "results": results,
    }
    output = args.report if args.report.is_absolute() else ROOT / args.report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        f"summary: {report['expected_successes']}/{report['tasks_run']} "
        f"expected successes; report={output}"
    )
    return 0 if report["all_expected_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
