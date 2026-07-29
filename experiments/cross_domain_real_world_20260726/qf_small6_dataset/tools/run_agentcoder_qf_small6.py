#!/usr/bin/env python3
"""Run an AgentCoder-style baseline on QF-Small6.

This is an interface adapter for the bundled code-only snapshot of the public
AgentCoder implementation under ``baselines/agentcoder/upstream``. It preserves
the HumanEval pipeline's central protocol:

* five independently sampled programmer candidates per pool;
* ten independently sampled tester programs per pool;
* a test executor selects the first candidate with the maximum tester score;
* score >= 3 freezes the candidate;
* otherwise a new pool is sampled, for at most six pools (initial pool plus
  the five executor epochs in the upstream script).

The original implementation only supports function-level HumanEval/MBPP tasks.
Here, the tester programs are adapted to QF-Bench's file input/output contract:
the executor runs a candidate in a temporary copy of the public task, then runs
the independently generated tester against the produced public outputs.

Only public problem text and public data paths are visible to either LLM agent.
Frozen private tests are invoked once, after final candidate selection.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


DATASET_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DATASET_ROOT.parents[2]
MANIFEST_PATH = DATASET_ROOT / "dataset_manifest.json"
BUNDLED_AGENTCODER_ROOT = REPO_ROOT / "baselines" / "agentcoder" / "upstream"
AGENTCODER_ROOT = Path(
    os.getenv("AGENTCODER_ROOT", str(BUNDLED_AGENTCODER_ROOT))
).resolve()
DEFAULT_RUN_ID = "agentcoder_deepseek_v4_flash_nothinking_20260728"
PROGRAMMER_SAMPLES = 5
TESTER_SAMPLES = 10
FREEZE_THRESHOLD = 3
MAX_POOLS = 6
EXECUTOR_WORKERS = 5


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
    )


def truncate(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n... [truncated {len(value) - limit} chars]"


def source_snapshot(root: Path) -> dict[str, Any]:
    include = []
    for relative in ("README.md", "requirements.txt", "UPSTREAM_COMMIT"):
        path = root / relative
        if path.is_file():
            include.append(path)
    for folder, pattern in (("src", "*.py"), ("prompts", "*.txt")):
        if (root / folder).is_dir():
            include.extend(sorted((root / folder).glob(pattern)))
    include = sorted(set(include))
    digest = hashlib.sha256()
    file_hashes: dict[str, str] = {}
    for path in include:
        relative = path.relative_to(root).as_posix()
        file_hash = sha256_file(path)
        file_hashes[relative] = file_hash
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return {
        "root": str(root),
        "git_repository": (root / ".git").exists(),
        "source_tree_sha256": digest.hexdigest(),
        "source_file_count": len(include),
        "source_file_hashes": file_hashes,
    }


def git_snapshot(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return completed.stdout.strip()

    status = git("status", "--porcelain=v1")
    return {
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_status_line_count": len(status.splitlines()) if status else 0,
        "git_status_sha256": sha256_text(status),
    }


def reset_public_app(task: dict[str, Any], app_path: Path) -> None:
    if app_path.exists():
        shutil.rmtree(app_path)
    app_path.mkdir(parents=True)
    source = DATASET_ROOT / "tasks" / task["task_id"] / "public/data"
    mount = app_path / task["data_mount"]
    mount.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, mount, dirs_exist_ok=True)


def build_public_problem(public_problem: str, app_path: Path) -> str:
    adapted = public_problem.replace("/app", str(app_path))
    protocol = f"""
# QF-Small6 executable contract

Write one complete Python program for this file-processing task.
The program is executed once with stdin closed. It must read the public input
files from the absolute paths in the task, create every required output file,
and exit successfully. Do not print an answer instead of creating the files.
Do not embed the input data or expected output values in the source.

The application directory is:
`{app_path}`

Include all imports and an executable entry point. Return only the complete
program inside one ```python code block.
""".strip()
    return f"{protocol}\n\n{adapted.strip()}\n"


def relocate_recorded_app_path(
    code: str,
    task_run_dir: Path,
    task_id: str,
) -> tuple[str, dict[str, Any]]:
    """Relocate a retained QF candidate without changing its source artifact.

    QF candidates are generated against an absolute per-run application path.
    When a retained run directory is copied to another machine or checkout,
    that path is stale. The original path is recovered from the retained
    public prompt and replaced only in the executable evaluation copy.
    """

    current_app_path = (task_run_dir / "app").resolve()
    relocation: dict[str, Any] = {
        "applied": False,
        "source": "prompt.txt",
        "recorded_app_path": None,
        "current_app_path": str(current_app_path),
        "replacement_count": 0,
        "original_candidate_sha256": sha256_text(code),
        "evaluated_candidate_sha256": sha256_text(code),
    }
    prompt_path = task_run_dir / "prompt.txt"
    if not prompt_path.is_file():
        relocation["reason"] = "retained prompt.txt is missing"
        return code, relocation

    prompt = prompt_path.read_text()
    match = re.search(
        r"The application directory is:\s*`([^`\r\n]+)`",
        prompt,
    )
    if not match:
        relocation["reason"] = "application directory not found in prompt.txt"
        return code, relocation

    recorded = Path(match.group(1))
    relocation["recorded_app_path"] = str(recorded)
    expected_tail = ("tasks", task_id, "app")
    if not recorded.is_absolute() or tuple(recorded.parts[-3:]) != expected_tail:
        relocation["reason"] = "recorded application path failed scope check"
        return code, relocation
    if recorded == current_app_path:
        relocation["reason"] = "recorded application path is already current"
        return code, relocation

    replacement_count = code.count(str(recorded))
    relocation["replacement_count"] = replacement_count
    if replacement_count == 0:
        relocation["reason"] = "candidate does not contain the recorded path"
        return code, relocation

    relocated = code.replace(str(recorded), str(current_app_path))
    relocation["applied"] = True
    relocation["reason"] = "retained absolute application path relocated"
    relocation["evaluated_candidate_sha256"] = sha256_text(relocated)
    return relocated, relocation


PROGRAMMER_PREAMBLE = """
**Role**: You are a software programmer.

**Task**: Complete the requested Python program. Reason about the algorithm and
the file contract, then return only the complete source code in one
```python ... ``` block.
""".strip()


TESTER_PREAMBLE = """
**Role**: You are an independent software tester.

Create ONE self-contained Python black-box test program for the task below.
The AgentCoder executor first runs a candidate solution, then runs your test in
the same application directory with environment variable AGENTCODER_APP set to
that directory. Your test must:

1. inspect one or more required candidate output files;
2. derive its expectation only from the public task statement and public input
   data available under AGENTCODER_APP;
3. contain at least one Python `assert` that checks a meaningful functional or
   numerical property;
4. not run the candidate itself, use the network, inspect private/oracle/test
   files, or merely assert a tautology;
5. be returned as one complete ```python ... ``` code block.

This tester is generated independently and cannot see any candidate program.
""".strip()


def build_programmer_prompt(problem: str) -> str:
    return f"{PROGRAMMER_PREAMBLE}\n\n**Task specification**:\n{problem}"


def build_tester_prompt(problem: str, outputs: list[str]) -> str:
    rendered_outputs = "\n".join(f"- `{value}`" for value in outputs)
    return (
        f"{TESTER_PREAMBLE}\n\n"
        f"Required candidate outputs include:\n{rendered_outputs}\n\n"
        f"**Task specification**:\n{problem}"
    )


def extract_python_block(response: str) -> str:
    match = re.search(r"```(?:python|py)\s*\n(.*?)```", response, re.S | re.I)
    if match:
        return match.group(1).strip() + "\n"
    generic = re.search(r"```\s*\n(.*?)```", response, re.S)
    if generic:
        return generic.group(1).strip() + "\n"
    candidate = response.strip()
    try:
        ast.parse(candidate)
    except SyntaxError:
        return ""
    return candidate + ("\n" if candidate else "")


def leakage_scan(text: str) -> dict[str, Any]:
    patterns = {
        "private_path": r"(?:^|[/'\"]+)private(?:[/'\"]+)",
        "upstream_test_file": r"test_outputs\.py",
        "contract_test_file": r"test_contract_additions\.py",
        "oracle_path": r"(?:^|[/'\"]+)oracle(?:[/'\"]+)",
    }
    matches = {
        name: bool(re.search(pattern, text, re.IGNORECASE | re.MULTILINE))
        for name, pattern in patterns.items()
    }
    return {"violation": any(matches.values()), "patterns": matches}


def tester_static_check(code: str, outputs: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "syntax_valid": False,
        "assert_count": 0,
        "mentions_required_output": False,
        "leakage_scan": leakage_scan(code),
        "valid": False,
    }
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        result["syntax_error"] = str(exc)
        return result
    result["syntax_valid"] = True
    result["assert_count"] = sum(
        isinstance(node, ast.Assert) for node in ast.walk(tree)
    )
    output_names = {Path(value).name for value in outputs}
    result["mentions_required_output"] = any(
        name in code for name in output_names
    )
    result["valid"] = bool(
        result["assert_count"] > 0
        and result["mentions_required_output"]
        and not result["leakage_scan"]["violation"]
    )
    return result


class LLMClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        max_tokens: int,
        temperature: float,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=180.0,
            max_retries=0,
        )
        self._lock = threading.Lock()
        self.records: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        task_id: str,
        pool: int,
        stage: str,
        sample_index: int,
        system: str,
        prompt: str,
    ) -> dict[str, Any]:
        started = time.monotonic()
        error = None
        response_text = ""
        usage_record: dict[str, Any] = {}
        for attempt in range(1, 5):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                response_text = response.choices[0].message.content or ""
                if response.usage is not None:
                    usage_record = {
                        "prompt_tokens": response.usage.prompt_tokens or 0,
                        "completion_tokens": response.usage.completion_tokens or 0,
                        "total_tokens": response.usage.total_tokens or 0,
                    }
                error = None
                break
            except Exception as exc:  # API transport/provider errors
                error = f"{type(exc).__name__}: {exc}"
                if attempt < 4:
                    time.sleep(min(2 ** attempt, 10))
        record = {
            "timestamp": utc_now(),
            "task_id": task_id,
            "pool": pool,
            "stage": stage,
            "sample_index": sample_index,
            "model": self.model,
            "attempts": attempt,
            "duration_seconds": round(time.monotonic() - started, 6),
            "prompt_sha256": sha256_text(prompt),
            "response_sha256": sha256_text(response_text),
            "usage": usage_record,
            "error": error,
        }
        with self._lock:
            self.records.append(record)
        return {
            "response": response_text,
            "record": record,
        }

    def usage_summary(self, task_id: str | None = None) -> dict[str, int]:
        with self._lock:
            records = list(self.records)
        if task_id is not None:
            records = [item for item in records if item["task_id"] == task_id]
        return {
            "calls": len(records),
            "failed_calls": sum(bool(item["error"]) for item in records),
            "prompt_tokens": sum(
                item["usage"].get("prompt_tokens", 0) for item in records
            ),
            "completion_tokens": sum(
                item["usage"].get("completion_tokens", 0) for item in records
            ),
            "total_tokens": sum(
                item["usage"].get("total_tokens", 0) for item in records
            ),
        }


def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout: float,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(input="", timeout=timeout)
        return {
            "returncode": process.returncode,
            "timeout": False,
            "duration_seconds": round(time.monotonic() - started, 6),
            "stdout": truncate(stdout),
            "stderr": truncate(stderr),
        }
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        return {
            "returncode": None,
            "timeout": True,
            "duration_seconds": round(time.monotonic() - started, 6),
            "stdout": truncate(stdout),
            "stderr": truncate(stderr),
        }


def execute_pair(
    task: dict[str, Any],
    *,
    visible_app_path: Path,
    candidate_index: int,
    tester_index: int,
    candidate_code: str,
    tester_code: str,
    tester_check: dict[str, Any],
    program_timeout: float,
    test_timeout: float,
) -> dict[str, Any]:
    if not candidate_code.strip():
        return {
            "candidate_index": candidate_index,
            "tester_index": tester_index,
            "passed": False,
            "reason": "empty_candidate",
        }
    if not tester_check["valid"]:
        return {
            "candidate_index": candidate_index,
            "tester_index": tester_index,
            "passed": False,
            "reason": "invalid_generated_test",
            "tester_static_check": tester_check,
        }
    try:
        ast.parse(candidate_code)
    except SyntaxError as exc:
        return {
            "candidate_index": candidate_index,
            "tester_index": tester_index,
            "passed": False,
            "reason": "candidate_syntax_error",
            "syntax_error": str(exc),
        }

    with tempfile.TemporaryDirectory(prefix="agentcoder-qf-pair-") as temp:
        app_path = Path(temp) / "app"
        reset_public_app(task, app_path)
        adapted_candidate = candidate_code.replace(
            str(visible_app_path), str(app_path)
        )
        adapted_test = tester_code.replace(str(visible_app_path), str(app_path))
        candidate_path = app_path / "solution.py"
        tester_path = app_path / "agent_test.py"
        candidate_path.write_text(adapted_candidate)
        tester_path.write_text(adapted_test)
        runtime_env = os.environ.copy()
        runtime_env["AGENTCODER_APP"] = str(app_path)
        runtime_env["PYTHONDONTWRITEBYTECODE"] = "1"
        candidate_run = run_command(
            [sys.executable, str(candidate_path)],
            cwd=app_path,
            timeout=program_timeout,
            env=runtime_env,
        )
        if candidate_run["returncode"] != 0 or candidate_run["timeout"]:
            return {
                "candidate_index": candidate_index,
                "tester_index": tester_index,
                "passed": False,
                "reason": "candidate_execution_failure",
                "candidate_execution": candidate_run,
            }
        tester_run = run_command(
            [sys.executable, str(tester_path)],
            cwd=app_path,
            timeout=test_timeout,
            env=runtime_env,
        )
        return {
            "candidate_index": candidate_index,
            "tester_index": tester_index,
            "passed": (
                tester_run["returncode"] == 0 and not tester_run["timeout"]
            ),
            "reason": (
                "generated_test_passed"
                if tester_run["returncode"] == 0 and not tester_run["timeout"]
                else "generated_test_failed"
            ),
            "candidate_execution": candidate_run,
            "tester_execution": tester_run,
        }


def generate_pool(
    task: dict[str, Any],
    *,
    task_run_dir: Path,
    public_problem: str,
    pool_index: int,
    llm: LLMClient,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    pool_dir = task_run_dir / "pools" / f"pool_{pool_index:02d}"
    candidate_dir = pool_dir / "candidates"
    tester_dir = pool_dir / "tests"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    tester_dir.mkdir(parents=True, exist_ok=True)
    programmer_prompt = build_programmer_prompt(public_problem)
    tester_prompt = build_tester_prompt(public_problem, task["outputs"])
    (pool_dir / "programmer_prompt.txt").write_text(programmer_prompt)
    (pool_dir / "tester_prompt.txt").write_text(tester_prompt)

    jobs = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=EXECUTOR_WORKERS
    ) as executor:
        for index in range(PROGRAMMER_SAMPLES):
            jobs.append(
                (
                    "programmer",
                    index,
                    executor.submit(
                        llm.complete,
                        task_id=task["task_id"],
                        pool=pool_index,
                        stage="programmer",
                        sample_index=index,
                        system="You are a software programmer.",
                        prompt=programmer_prompt,
                    ),
                )
            )
        for index in range(TESTER_SAMPLES):
            jobs.append(
                (
                    "tester",
                    index,
                    executor.submit(
                        llm.complete,
                        task_id=task["task_id"],
                        pool=pool_index,
                        stage="tester",
                        sample_index=index,
                        system="You are an independent code tester.",
                        prompt=tester_prompt,
                    ),
                )
            )
        generated = [
            (stage, index, future.result())
            for stage, index, future in jobs
        ]

    repaired_generated = []
    for stage, index, generated_item in generated:
        retry_cycles = 0
        while not generated_item["response"].strip():
            retry_cycles += 1
            if retry_cycles > 5:
                raise RuntimeError(
                    f"{task['task_id']} pool {pool_index} {stage} "
                    f"sample {index}: provider returned no completion after "
                    "transport retries"
                )
            time.sleep(min(2 ** retry_cycles, 10))
            generated_item = llm.complete(
                task_id=task["task_id"],
                pool=pool_index,
                stage=stage,
                sample_index=index,
                system=(
                    "You are a software programmer."
                    if stage == "programmer"
                    else "You are an independent code tester."
                ),
                prompt=(
                    programmer_prompt if stage == "programmer" else tester_prompt
                ),
            )
        repaired_generated.append((stage, index, generated_item))

    candidates = [""] * PROGRAMMER_SAMPLES
    tests = [""] * TESTER_SAMPLES
    checks: list[dict[str, Any]] = [{} for _ in range(TESTER_SAMPLES)]
    for stage, index, generated_item in repaired_generated:
        response = generated_item["response"]
        code = extract_python_block(response)
        if stage == "programmer":
            candidates[index] = code
            (candidate_dir / f"candidate_{index:02d}.response.txt").write_text(
                response
            )
            (candidate_dir / f"candidate_{index:02d}.py").write_text(code)
            write_json(
                candidate_dir / f"candidate_{index:02d}.call.json",
                generated_item["record"],
            )
        else:
            tests[index] = code
            checks[index] = tester_static_check(code, task["outputs"])
            (tester_dir / f"test_{index:02d}.response.txt").write_text(response)
            (tester_dir / f"test_{index:02d}.py").write_text(code)
            write_json(
                tester_dir / f"test_{index:02d}.call.json",
                generated_item["record"],
            )
            write_json(
                tester_dir / f"test_{index:02d}.static.json",
                checks[index],
            )
    return candidates, tests, checks


def score_pool(
    task: dict[str, Any],
    *,
    task_run_dir: Path,
    visible_app_path: Path,
    pool_index: int,
    candidates: list[str],
    tests: list[str],
    checks: list[dict[str, Any]],
    executor_workers: int,
    program_timeout: float,
    test_timeout: float,
) -> dict[str, Any]:
    started = time.monotonic()
    pairs = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=executor_workers
    ) as executor:
        futures = [
            executor.submit(
                execute_pair,
                task,
                visible_app_path=visible_app_path,
                candidate_index=candidate_index,
                tester_index=tester_index,
                candidate_code=candidate,
                tester_code=test,
                tester_check=checks[tester_index],
                program_timeout=program_timeout,
                test_timeout=test_timeout,
            )
            for candidate_index, candidate in enumerate(candidates)
            for tester_index, test in enumerate(tests)
        ]
        for future in concurrent.futures.as_completed(futures):
            pairs.append(future.result())
    pairs.sort(key=lambda item: (item["candidate_index"], item["tester_index"]))
    scores = [
        sum(
            pair["passed"]
            for pair in pairs
            if pair["candidate_index"] == candidate_index
        )
        for candidate_index in range(len(candidates))
    ]
    max_score = max(scores) if scores else 0
    selected_index = scores.index(max_score) if scores else 0
    result = {
        "pool": pool_index,
        "started_at": utc_now(),
        "duration_seconds": round(time.monotonic() - started, 6),
        "candidate_scores": scores,
        "max_score": max_score,
        "selected_index": selected_index,
        "freeze_threshold": FREEZE_THRESHOLD,
        "frozen": max_score >= FREEZE_THRESHOLD,
        "valid_generated_tests": sum(check["valid"] for check in checks),
        "pairs": pairs,
    }
    write_json(
        task_run_dir
        / "pools"
        / f"pool_{pool_index:02d}"
        / "executor_matrix.json",
        result,
    )
    return result


def parse_junit(path: Path) -> dict[str, Any]:
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


def materialize_test(source: Path, destination: Path, app_path: Path) -> None:
    destination.write_text(source.read_text().replace("/app", str(app_path)))


def run_pytest(
    test_file: Path,
    junit_path: Path,
    *,
    cwd: Path,
) -> dict[str, Any]:
    result = run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(test_file),
            f"--junitxml={junit_path}",
            "--disable-warnings",
        ],
        cwd=cwd,
        timeout=300,
        env=os.environ.copy(),
    )
    result.update(parse_junit(junit_path))
    result["test_sha256"] = sha256_file(test_file)
    return result


def evaluate_candidate(
    task: dict[str, Any],
    task_run_dir: Path,
    code: str,
) -> dict[str, Any]:
    app_path = task_run_dir / "app"
    reset_public_app(task, app_path)
    evaluation_dir = task_run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    executable_code, relocation = relocate_recorded_app_path(
        code,
        task_run_dir,
        task["task_id"],
    )
    final_code_path = task_run_dir / "final_code.py"
    if not final_code_path.is_file():
        final_code_path.write_text(code)
    elif final_code_path.read_text() != code:
        raise ValueError("final_code.py differs from the candidate being evaluated")
    if relocation["applied"]:
        code_path = evaluation_dir / "relocated_candidate.py"
        code_path.write_text(executable_code)
    else:
        code_path = final_code_path
    execution = run_command(
        [sys.executable, str(code_path)],
        cwd=app_path,
        timeout=300,
        env=os.environ.copy(),
    )

    task_source = DATASET_ROOT / "tasks" / task["task_id"] / "private/tests"
    official_test = evaluation_dir / "test_outputs.py"
    addition_test = evaluation_dir / "test_contract_additions.py"
    materialize_test(task_source / "test_outputs.py", official_test, app_path)
    materialize_test(
        task_source / "test_contract_additions.py", addition_test, app_path
    )
    official = run_pytest(
        official_test,
        evaluation_dir / "official-junit.xml",
        cwd=app_path,
    )
    additions = run_pytest(
        addition_test,
        evaluation_dir / "additions-junit.xml",
        cwd=app_path,
    )
    leakage = leakage_scan(code)
    required_outputs = {
        relative: (
            sha256_file(app_path / relative)
            if (app_path / relative).is_file()
            else None
        )
        for relative in task["outputs"]
    }
    resolved = (
        execution["returncode"] == 0
        and not execution["timeout"]
        and official["returncode"] == 0
        and additions["returncode"] == 0
        and not leakage["violation"]
    )
    return {
        "candidate_execution": execution,
        "candidate_path_relocation": relocation,
        "upstream_tests": official,
        "contract_addition_tests": additions,
        "required_output_hashes": required_outputs,
        "leakage_scan": leakage,
        "task_resolved": resolved,
    }


def run_one_task(
    task: dict[str, Any],
    run_dir: Path,
    *,
    llm: LLMClient,
    max_pools: int,
    executor_workers: int,
    program_timeout: float,
    test_timeout: float,
) -> dict[str, Any]:
    task_id = task["task_id"]
    task_run_dir = run_dir / "tasks" / task_id
    task_run_dir.mkdir(parents=True, exist_ok=True)
    visible_app_path = task_run_dir / "app"
    reset_public_app(task, visible_app_path)
    upstream_problem = (
        DATASET_ROOT / "tasks" / task_id / "public/problem.md"
    ).read_text()
    public_problem = build_public_problem(upstream_problem, visible_app_path)
    (task_run_dir / "prompt.txt").write_text(public_problem)

    started = time.monotonic()
    pool_results = []
    selected_code = ""
    selected_pool = None
    selected_index = None
    frozen = False
    for pool_index in range(max_pools):
        print(
            f"{task_id}: AgentCoder pool {pool_index + 1}/{max_pools} generation",
            flush=True,
        )
        candidates, tests, checks = generate_pool(
            task,
            task_run_dir=task_run_dir,
            public_problem=public_problem,
            pool_index=pool_index,
            llm=llm,
        )
        print(
            f"{task_id}: AgentCoder pool {pool_index + 1}/{max_pools} executor",
            flush=True,
        )
        pool_result = score_pool(
            task,
            task_run_dir=task_run_dir,
            visible_app_path=visible_app_path,
            pool_index=pool_index,
            candidates=candidates,
            tests=tests,
            checks=checks,
            executor_workers=executor_workers,
            program_timeout=program_timeout,
            test_timeout=test_timeout,
        )
        pool_results.append(pool_result)
        selected_index = pool_result["selected_index"]
        selected_pool = pool_index
        selected_code = candidates[selected_index]
        frozen = pool_result["frozen"]
        state = {
            "task_id": task_id,
            "updated_at": utc_now(),
            "selected_pool": selected_pool,
            "selected_index": selected_index,
            "selected_score": pool_result["max_score"],
            "frozen": frozen,
            "pool_results": [
                {
                    key: value
                    for key, value in item.items()
                    if key != "pairs"
                }
                for item in pool_results
            ],
            "token_usage": llm.usage_summary(task_id),
        }
        write_json(task_run_dir / "state.json", state)
        (task_run_dir / "selected_code.py").write_text(selected_code)
        print(
            f"{task_id}: score={pool_result['max_score']}/"
            f"{TESTER_SAMPLES}, valid_tests={pool_result['valid_generated_tests']}, "
            f"frozen={frozen}",
            flush=True,
        )
        if frozen:
            break

    (task_run_dir / "final_code.py").write_text(selected_code)
    external_evaluation = evaluate_candidate(task, task_run_dir, selected_code)
    return {
        "task_id": task_id,
        "timestamp": utc_now(),
        "dataset_revision": json.loads(MANIFEST_PATH.read_text())["source"][
            "revision"
        ],
        "prompt_sha256": sha256_text(public_problem),
        "final_code_sha256": sha256_text(selected_code),
        "final_code_nonblank_loc": sum(
            bool(line.strip()) for line in selected_code.splitlines()
        ),
        "agentcoder": {
            "variant": "HumanEval selection protocol with file-I/O adapter",
            "programmer_samples_per_pool": PROGRAMMER_SAMPLES,
            "tester_samples_per_pool": TESTER_SAMPLES,
            "freeze_threshold": FREEZE_THRESHOLD,
            "max_pools": max_pools,
            "pools_executed": len(pool_results),
            "selected_pool": selected_pool,
            "selected_index": selected_index,
            "selected_generated_test_score": (
                pool_results[-1]["max_score"] if pool_results else 0
            ),
            "frozen": frozen,
            "pool_results": [
                {
                    key: value
                    for key, value in item.items()
                    if key != "pairs"
                }
                for item in pool_results
            ],
            "hidden_test_feedback": False,
            "programmer_tester_independence": True,
        },
        "pipeline_seconds": round(time.monotonic() - started, 6),
        "token_usage": llm.usage_summary(task_id),
        "external_evaluation": external_evaluation,
    }


def result_row(result: dict[str, Any]) -> str:
    if result.get("error"):
        return (
            f"| `{result['task_id']}` | ERROR | 0/0 | 0/0 | "
            f"- | - | `{result['error']['type']}` |"
        )
    evaluation = result["external_evaluation"]
    upstream = evaluation["upstream_tests"]
    additions = evaluation["contract_addition_tests"]
    protocol = result["agentcoder"]
    return (
        f"| `{result['task_id']}` | "
        f"{'PASS' if evaluation['task_resolved'] else 'FAIL'} | "
        f"{upstream['passed']}/{upstream['collected']} | "
        f"{additions['passed']}/{additions['collected']} | "
        f"{protocol['selected_generated_test_score']}/10 | "
        f"{result['token_usage']['total_tokens']} |  |"
    )


def build_summary(
    run_dir: Path,
    config: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    completed = [result for result in results if not result.get("error")]
    resolved = sum(
        bool(result["external_evaluation"]["task_resolved"])
        for result in completed
    )
    pipeline_errors = sum(bool(result.get("error")) for result in results)
    upstream_passed = sum(
        result["external_evaluation"]["upstream_tests"]["passed"]
        for result in completed
    )
    upstream_collected = sum(
        result["external_evaluation"]["upstream_tests"]["collected"]
        for result in completed
    )
    additions_passed = sum(
        result["external_evaluation"]["contract_addition_tests"]["passed"]
        for result in completed
    )
    additions_collected = sum(
        result["external_evaluation"]["contract_addition_tests"]["collected"]
        for result in completed
    )
    total_calls = sum(result["token_usage"]["calls"] for result in completed)
    failed_calls = sum(
        result["token_usage"]["failed_calls"] for result in completed
    )
    total_tokens = sum(
        result["token_usage"]["total_tokens"] for result in completed
    )
    total_seconds = sum(result["pipeline_seconds"] for result in completed)
    summary = {
        "run_id": config["run_id"],
        "generated_at": utc_now(),
        "method": "AgentCoder",
        "dataset": config["dataset"],
        "agentcoder_source_snapshot": config["agentcoder_source_snapshot"],
        "model": config["model"],
        "tasks_requested": len(config["task_ids"]),
        "tasks_completed": len(completed),
        "tasks_with_pipeline_result": len(completed),
        "pipeline_errors": pipeline_errors,
        "resolved_tasks": resolved,
        "resolved_rate": (
            resolved / len(config["task_ids"]) if config["task_ids"] else None
        ),
        "upstream_tests": {
            "passed": upstream_passed,
            "collected": upstream_collected,
        },
        "contract_addition_tests": {
            "passed": additions_passed,
            "collected": additions_collected,
        },
        "official_tests_passed": upstream_passed + additions_passed,
        "official_tests_total": upstream_collected + additions_collected,
        "token_usage": {
            "calls": total_calls,
            "failed_calls": failed_calls,
            "total_tokens": total_tokens,
        },
        "total_pipeline_seconds": round(total_seconds, 6),
        "results": results,
    }
    write_json(run_dir / "summary.json", summary)

    lines = [
        f"# QF-Small6 AgentCoder Run: {config['run_id']}",
        "",
        f"- Model: `{config['model']['name']}`",
        f"- Dataset revision: `{config['dataset']['revision']}`",
        f"- AgentCoder source SHA-256: "
        f"`{config['agentcoder_source_snapshot']['source_tree_sha256']}`",
        f"- Resolved@1: `{resolved}/{len(config['task_ids'])}` "
        f"({(100 * resolved / len(config['task_ids'])) if config['task_ids'] else 0:.2f}%)",
        f"- Pipeline errors: `{pipeline_errors}`",
        f"- Upstream tests: `{upstream_passed}/{upstream_collected}`",
        f"- Contract additions: `{additions_passed}/{additions_collected}`",
        f"- LLM calls / failed calls / tokens: "
        f"`{total_calls}` / `{failed_calls}` / `{total_tokens}`",
        f"- Total pipeline time: `{total_seconds:.3f}s`",
        "",
        "| Task | External verdict | Upstream | Additions | AgentCoder test score | Tokens | Error |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    lines.extend(result_row(result) for result in results)
    lines.extend(
        [
            "",
            "The frozen external tests determine functional correctness. "
            "AgentCoder-generated test scores are selection signals only.",
            "",
            "This run adapts the local AgentCoder HumanEval protocol from "
            "function assertions to independent file-output tester programs. "
            "It does not expose private QF tests to the programmer or tester.",
            "",
        ]
    )
    (run_dir / "SUMMARY.md").write_text("\n".join(lines))
    return summary


def write_checksums(run_dir: Path) -> None:
    checksum_path = run_dir / "ARTIFACT_SHA256SUMS"
    paths = sorted(
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path != checksum_path
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(run_dir).as_posix()}"
        for path in paths
    ]
    checksum_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task id to run; repeat for multiple tasks. Default: all six.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Re-evaluate existing final code without invoking AgentCoder/LLM.",
    )
    parser.add_argument("--max-pools", type=int, default=MAX_POOLS)
    parser.add_argument("--executor-workers", type=int, default=EXECUTOR_WORKERS)
    parser.add_argument("--program-timeout", type=float, default=180.0)
    parser.add_argument("--test-timeout", type=float, default=60.0)
    args = parser.parse_args()

    if not 1 <= args.max_pools <= MAX_POOLS:
        raise SystemExit(f"--max-pools must be in [1, {MAX_POOLS}]")
    if args.executor_workers < 1:
        raise SystemExit("--executor-workers must be positive")
    if not AGENTCODER_ROOT.is_dir():
        raise SystemExit(f"AgentCoder root not found: {AGENTCODER_ROOT}")

    load_dotenv(REPO_ROOT / ".env", override=True)
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key and not args.evaluate_only:
        raise SystemExit("DEEPSEEK_API_KEY is not configured")
    model_name = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    max_tokens = int(
        os.getenv("DEEPSEEK_MAX_TOKENS") or os.getenv("MAX_TOKENS") or "4096"
    )

    manifest = json.loads(MANIFEST_PATH.read_text())
    task_map = {task["task_id"]: task for task in manifest["tasks"]}
    selected_ids = args.task or [task["task_id"] for task in manifest["tasks"]]
    unknown = set(selected_ids) - set(task_map)
    if unknown:
        raise SystemExit(f"unknown task(s): {sorted(unknown)}")

    run_dir = DATASET_ROOT / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run_id": args.run_id,
        "created_at": utc_now(),
        "dataset": {
            "name": manifest["dataset_name"],
            "revision": manifest["source"]["revision"],
        },
        "task_ids": selected_ids,
        "model": {
            "provider": "deepseek-compatible-openai-api",
            "name": model_name,
            "base_url": base_url,
            "max_tokens": max_tokens,
            "temperature": 1.0,
            "thinking": {"type": "disabled"},
            "api_key_configured": bool(api_key),
        },
        "runtime": {
            "python": sys.executable,
            "python_version": sys.version,
            "host_subprocess_not_security_sandbox": True,
            "official_harbor_docker": False,
        },
        "protocol": {
            "upstream_variant": "src/test_executor_humaneval.py",
            "provider_reasoning_mode": (
                "disabled because upstream AgentCoder consumes only "
                "ChatCompletion message.content"
            ),
            "interface_adaptation": (
                "function assertions -> independent file-output tester programs"
            ),
            "programmer_samples_per_pool": PROGRAMMER_SAMPLES,
            "tester_samples_per_pool": TESTER_SAMPLES,
            "freeze_threshold": FREEZE_THRESHOLD,
            "max_pools": args.max_pools,
            "executor_workers": args.executor_workers,
            "hidden_test_feedback": False,
            "candidate_stdin": "closed EOF",
            "public_prompt_path_rewrite": "/app -> task-specific absolute app path",
            "functional_judge": "upstream tests AND contract-addition tests",
        },
        "agentcoder_source_snapshot": source_snapshot(AGENTCODER_ROOT),
        "adapter": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "auditcoder_repository_snapshot": git_snapshot(REPO_ROOT),
    }
    config_path = run_dir / "run_config.json"
    if args.evaluate_only and config_path.is_file():
        original_config = json.loads(config_path.read_text())
        original_config["last_evaluated_at"] = utc_now()
        config = original_config
    write_json(config_path, config)

    llm = LLMClient(
        model=model_name,
        base_url=base_url,
        api_key=api_key or "evaluate-only",
        max_tokens=max_tokens,
        temperature=1.0,
    )
    existing_calls_path = run_dir / "llm_calls.json"
    if existing_calls_path.is_file():
        existing_calls = json.loads(existing_calls_path.read_text())
        if isinstance(existing_calls, list):
            if args.force:
                existing_calls = [
                    item
                    for item in existing_calls
                    if item.get("task_id") not in selected_ids
                ]
            llm.records.extend(existing_calls)
    results = []
    for task_id in selected_ids:
        result_path = run_dir / "tasks" / task_id / "result.json"
        task_run_dir = run_dir / "tasks" / task_id
        if args.evaluate_only:
            code_path = task_run_dir / "final_code.py"
            if not result_path.is_file() or not code_path.is_file():
                raise SystemExit(f"missing existing result/code for {task_id}")
            result = json.loads(result_path.read_text())
            result["external_evaluation"] = evaluate_candidate(
                task_map[task_id], task_run_dir, code_path.read_text()
            )
            result["external_evaluation_revision"] = {
                "evaluated_at": utc_now(),
                "llm_or_agentcoder_rerun": False,
            }
            write_json(result_path, result)
            results.append(result)
            print(
                f"{task_id}: re-evaluated "
                f"resolved={result['external_evaluation']['task_resolved']}",
                flush=True,
            )
            continue
        if result_path.exists() and not args.force:
            result = json.loads(result_path.read_text())
            print(f"{task_id}: resume existing result", flush=True)
            results.append(result)
            continue
        if args.force and task_run_dir.exists():
            shutil.rmtree(task_run_dir)
        print(f"{task_id}: AgentCoder start", flush=True)
        try:
            result = run_one_task(
                task_map[task_id],
                run_dir,
                llm=llm,
                max_pools=args.max_pools,
                executor_workers=args.executor_workers,
                program_timeout=args.program_timeout,
                test_timeout=args.test_timeout,
            )
            evaluation = result["external_evaluation"]
            print(
                f"{task_id}: resolved={evaluation['task_resolved']} "
                f"upstream={evaluation['upstream_tests']['passed']}/"
                f"{evaluation['upstream_tests']['collected']} "
                f"additions={evaluation['contract_addition_tests']['passed']}/"
                f"{evaluation['contract_addition_tests']['collected']}",
                flush=True,
            )
        except Exception as exc:
            result = {
                "task_id": task_id,
                "timestamp": utc_now(),
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
            print(
                f"{task_id}: ERROR {type(exc).__name__}: {exc}",
                flush=True,
            )
        write_json(result_path, result)
        write_json(run_dir / "llm_calls.json", llm.records)
        results.append(result)
        build_summary(run_dir, config, results)
        write_checksums(run_dir)

    summary = build_summary(run_dir, config, results)
    write_json(run_dir / "llm_calls.json", llm.records)
    write_checksums(run_dir)
    print(
        f"summary: resolved={summary['resolved_tasks']}/"
        f"{summary['tasks_requested']} pipeline_errors={summary['pipeline_errors']} "
        f"report={run_dir / 'SUMMARY.md'}",
        flush=True,
    )
    return 0 if summary["pipeline_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
