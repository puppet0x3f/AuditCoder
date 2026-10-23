"""Run the vendored mapcoder strategy on the fixed 200-task APPS sample.

From the repository root::

    python baselines/mapcoder/run_apps_200_qwen.py --check
    python baselines/mapcoder/run_apps_200_qwen.py --limit 1
    python baselines/mapcoder/run_apps_200_qwen.py --max-cases 10

The first two APPS pairs serve as public examples. The final judge checks ten
pairs by default; --max-cases 0 checks all stored pairs. Reusing the same results
path resumes saved task IDs. See docs/TESTING.md for protocol and output details.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT
DEFAULT_INPUT = REPO_ROOT / "data" / "apps_sampled_200.jsonl"
DEFAULT_RESULTS = REPO_ROOT / "results" / "mapcoder_apps_200_qwen.jsonl"
EXPECTED_COUNTS = {"introductory": 50, "interview": 100, "competition": 50}


def read_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def normalize_qwen_model(model: str, base_url: str) -> str:
    """Accept the local alias while using DashScope's official model id."""
    if model.strip().lower() == "qwen-3.6-plus" and "dashscope.aliyuncs.com" in base_url.lower():
        return "qwen3.6-plus"
    return model.strip()


def load_sample(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = Counter(row.get("difficulty") for row in rows)
    if len(rows) != 200 or dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"Expected 50/100/50 APPS test problems, found {dict(counts)}")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(pid, int) for pid in ids) or len(set(ids)) != len(ids):
        raise ValueError("APPS sample must have 200 distinct integer IDs")
    for row in rows:
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError(f"Problem {row['id']} has no question")
        io = row.get("input_output")
        io = json.loads(io) if isinstance(io, str) else io
        if not isinstance(io, dict) or not io.get("inputs") or len(io["inputs"]) != len(io.get("outputs", [])):
            raise ValueError(f"Problem {row['id']} has invalid input_output")
        row["_io"] = io
    return rows


def code_from_response(response: str) -> str:
    """Correct upstream MapCoder's fenced-code parser, which duplicates a character."""
    python_blocks = re.findall(r"```(?:python3?|py)\s*(.*?)```", response, flags=re.I | re.S)
    if python_blocks:
        return python_blocks[-1].strip()
    blocks = re.findall(r"```\s*(.*?)```", response, flags=re.S)
    return blocks[-1].strip() if blocks else response.strip()


def preserve_or_wrap_cdata(text: str, tag: str) -> str:
    """Wrap plain XML element text without duplicating existing CDATA sections."""
    pattern = re.compile(rf"(<{tag}>)(.*?)(</{tag}>)", re.S)

    def replace(match: re.Match[str]) -> str:
        body = match.group(2)
        if re.match(r"\s*<!\[CDATA\[", body) and re.search(r"\]\]>\s*$", body):
            return match.group(0)
        return f"{match.group(1)}<![CDATA[{body}]]>{match.group(3)}"

    return pattern.sub(replace, text)


class APIRequestError(RuntimeError):
    """An API problem that should stop the run without scoring the task."""


class OutputTruncatedError(APIRequestError):
    """This task exhausted the configured output budget without producing a complete reply."""


def console_log(message: str) -> None:
    """Print request progress outside the per-task trace redirection."""
    stream = getattr(sys, "__stdout__", None) or sys.stdout
    print(message, file=stream, flush=True)


class QwenModel:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        thinking: str,
        max_tokens: int,
        timeout: int,
        attempts: int = 2,
    ):
        self.api_key = api_key
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.attempts = attempts
        self.calls = 0
        self.requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.served_models: set[str] = set()

    def prompt(self, processed_input: list[dict[str, str]]) -> tuple[str, int, int]:
        token_limits = [self.max_tokens]
        if self.max_tokens < 65536:
            token_limits.append(min(self.max_tokens * 2, 65536))
        for token_limit in token_limits:
            body = json.dumps({
                "model": self.model,
                "messages": processed_input,
                "temperature": 0,
                "top_p": 0.95,
                # DashScope's OpenAI-compatible API uses these top-level
                # fields for Qwen3.6 hybrid thinking models.
                "max_completion_tokens": token_limit,
                "enable_thinking": self.thinking == "enabled",
                "stream": False,
            }).encode("utf-8")
            request = urllib.request.Request(
                self.url,
                data=body,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            for attempt in range(self.attempts):
                self.requests += 1
                request_started = time.monotonic()
                console_log(
                    f"[Qwen] request={self.requests} model={self.model} "
                    f"max_completion_tokens={token_limit} "
                    f"attempt={attempt + 1}/{self.attempts}"
                )
                try:
                    with urllib.request.urlopen(request, timeout=self.timeout) as response:
                        payload = json.load(response)
                    console_log(
                        f"[Qwen] response received in {time.monotonic() - request_started:.1f}s"
                    )
                    break
                except urllib.error.HTTPError as exc:
                    detail = exc.read(500).decode("utf-8", errors="replace")
                    if exc.code not in {429, 500, 502, 503, 504} or attempt == self.attempts - 1:
                        raise APIRequestError(f"Qwen HTTP {exc.code}: {detail}") from exc
                    console_log(f"[Qwen] HTTP {exc.code}; retrying after backoff")
                except (urllib.error.URLError, TimeoutError) as exc:
                    if attempt == self.attempts - 1:
                        raise APIRequestError(f"Qwen connection failed: {exc}") from exc
                    console_log(f"[Qwen] connection timeout/error: {exc}; retrying after backoff")
                time.sleep(2 ** attempt)

            choice = payload["choices"][0]
            usage = payload.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            self.calls += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            if payload.get("model"):
                self.served_models.add(str(payload["model"]))
            if choice.get("finish_reason") == "length":
                print(f"Qwen response truncated at max_completion_tokens={token_limit}; retrying if possible.")
                console_log(f"[Qwen] response truncated at {token_limit}; trying the next output budget")
                continue
            content = choice["message"].get("content")
            if not isinstance(content, str) or not content.strip():
                raise APIRequestError(f"Qwen returned empty content (finish_reason={choice.get('finish_reason')})")
            return content, prompt_tokens, completion_tokens
        raise OutputTruncatedError(f"Qwen response remained truncated at max_completion_tokens={token_limits[-1]}")


FUNCTION_DRIVER = """\
import json
import sys
from pathlib import Path

namespace = {'__name__': 'mapcoder_solution'}
source = Path('solution.py').read_text(encoding='utf-8')
exec(compile(source, 'solution.py', 'exec'), namespace)
name = sys.argv[1]
function = namespace.get(name)
if function is None and 'Solution' in namespace:
    function = getattr(namespace['Solution'](), name)
if function is None:
    raise AttributeError(f'Function {name!r} was not defined')
arguments = json.loads(sys.stdin.read())
print(json.dumps(function(*arguments), ensure_ascii=False))
"""


class LocalAPPSJudge:
    def __init__(self, *, timeout: int, max_cases: int):
        self.timeout = timeout
        self.max_cases = max_cases
        self.last_result: dict[str, Any] = {}

    @staticmethod
    def _child_env() -> dict[str, str]:
        allowed = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "COMSPEC"}
        env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
        env["PYTHONUTF8"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    def check(self, code: str, cases: list[dict[str, Any]], fn_name: str | None, *, sample: bool) -> tuple[bool, str]:
        selected = cases if sample or self.max_cases == 0 else cases[:self.max_cases]
        with tempfile.TemporaryDirectory(prefix="mapcoder-apps-") as temp:
            folder = Path(temp)
            program = folder / "solution.py"
            program.write_text(code, encoding="utf-8")
            if fn_name:
                (folder / "driver.py").write_text(FUNCTION_DRIVER, encoding="utf-8")
                command = [sys.executable, "-I", "-B", "-X", "utf8", str(folder / "driver.py"), fn_name]
            else:
                command = [sys.executable, "-I", "-B", "-X", "utf8", str(program)]

            for index, case in enumerate(selected):
                stdin = json.dumps(case["args"], ensure_ascii=False) if fn_name else case["input"]
                try:
                    result = subprocess.run(
                        command,
                        input=stdin,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        capture_output=True,
                        timeout=self.timeout,
                        cwd=folder,
                        env=self._child_env(),
                    )
                    actual = result.stdout.rstrip("\n")
                    if fn_name:
                        try:
                            passed = result.returncode == 0 and json.loads(actual) == case["expected"]
                        except json.JSONDecodeError:
                            passed = False
                    else:
                        passed = result.returncode == 0 and actual == str(case["expected"]).rstrip("\n")
                    error = result.stderr.strip() if result.returncode else ""
                except subprocess.TimeoutExpired:
                    actual, passed, error = "", False, f"Timed out after {self.timeout}s"
                if not passed:
                    self.last_result = {"passed": False, "cases_run": index + 1, "failed_case": index, "error": error[:1000]}
                    feedback = (
                        f"Input:\n{stdin[:1000]}\nExpected output:\n{str(case['expected'])[:1000]}\n"
                        f"Your output:\n{actual[:1000]}\nError:\n{error[:1000]}"
                    )
                    return False, feedback

        self.last_result = {"passed": True, "cases_run": len(selected), "failed_case": None, "error": ""}
        return True, ""


def make_dataset_class(upstream_apps_class: type):
    class SampledAPPSDataset(upstream_apps_class):
        def __init__(self, rows: list[dict[str, Any]], judge: LocalAPPSJudge):
            self.id_key = "id"
            self.judge = judge
            self.data = []
            for row in rows:
                io = row["_io"]
                fn_name = io.get("fn_name")
                cases = []
                for raw_input, expected in zip(io["inputs"], io["outputs"]):
                    cases.append({
                        "input": json.dumps(raw_input, ensure_ascii=False) if fn_name else raw_input,
                        "args": raw_input if fn_name else None,
                        "expected": expected,
                        "output": [expected],
                    })
                self.data.append({
                    "id": row["id"],
                    "description": row["question"],
                    "difficulty": row["difficulty"],
                    "fn_name": fn_name,
                    "starter_code": row.get("starter_code", ""),
                    "sample_io": cases[:2],
                    "test_list": cases,
                })

        @staticmethod
        def get_prompt(item: dict[str, Any]) -> str:
            if item["fn_name"]:
                return (
                    f"{item['description']}\n\nImplement the Python function {item['fn_name']} "
                    "(either at top level or as a method of class Solution). "
                    "The evaluator calls it with positional arguments and compares its return value. "
                    "Do not read stdin or print the answer."
                )
            return upstream_apps_class.get_prompt(item)

        def evaluate_sample_io(self, item: dict[str, Any], cur_imp: str, language: str):
            return self.judge.check(cur_imp, item["sample_io"], item["fn_name"], sample=True)

        def evaluate(self, item: dict[str, Any], cur_imp: str, language: str):
            passed, _ = self.judge.check(cur_imp, item["test_list"], item["fn_name"], sample=False)
            return passed

    return SampledAPPSDataset


def load_previous(path: Path, settings: dict[str, Any]) -> dict[int, dict[str, Any]]:
    previous = {}
    if not path.exists():
        return previous
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("settings") != settings:
                raise ValueError(f"Existing results use different settings: {path}")
            previous[record["id"]] = record
    return previous


def append_result(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_summary(records: dict[int, dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    completed = sum(record.get("status") == "completed" for record in records.values())
    print(f"Processed: {len(records)}/{len(rows)}; completed: {completed}; output truncated: {sum(record.get('status') == 'output_truncated' for record in records.values())}")
    for difficulty in EXPECTED_COUNTS:
        subset = [record for record in records.values() if record["difficulty"] == difficulty]
        passed = sum(bool(record.get("is_solved")) for record in subset)
        target = EXPECTED_COUNTS[difficulty]
        score = f"{100 * passed / target:.1f}%" if len(subset) == target else "incomplete"
        print(f"{difficulty}: {passed}/{target} solved, {len(subset)}/{target} processed, score={score}")
    solved = sum(bool(record.get("is_solved")) for record in records.values())
    score = f"{100 * solved / len(rows):.1f}%" if len(records) == len(rows) else "incomplete"
    print(f"Total: {solved}/{len(rows)} solved, {len(records)}/{len(rows)} processed, score={score}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--env", type=Path, default=WORKSPACE_ROOT / ".env")
    parser.add_argument("--mapcoder-root", type=Path, default=REPO_ROOT / "baselines" / "mapcoder" / "upstream")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N sample items; 0 means all")
    parser.add_argument("--max-cases", type=int, default=10, help="Final tests per problem; 0 means all")
    parser.add_argument("--case-timeout", type=int, default=10)
    parser.add_argument("--api-timeout", type=int, default=120, help="Timeout for one Qwen HTTP request in seconds")
    parser.add_argument("--api-attempts", type=int, default=2, help="Attempts per Qwen request")
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--check", action="store_true", help="Validate files without sending API calls")
    args = parser.parse_args()
    if args.limit < 0 or args.max_cases < 0:
        parser.error("limit/max-cases must be nonnegative")
    if min(args.case_timeout, args.api_timeout, args.api_attempts, args.max_tokens) <= 0:
        parser.error("timeouts, api-attempts, and max-tokens must be positive")

    input_path = args.input.resolve(strict=True)
    results_path = args.results.resolve()
    mapcoder_root = args.mapcoder_root.resolve(strict=True)
    rows = load_sample(input_path)
    args.env = args.env.expanduser().resolve()
    sys.path.insert(0, str(mapcoder_root / "src"))
    os.chdir(mapcoder_root)
    from datasets.APPSDataset import APPSDataset
    from promptings.MapCoder import MapCoder

    if args.check:
        dataset = make_dataset_class(APPSDataset)(
            rows, LocalAPPSJudge(timeout=args.case_timeout, max_cases=args.max_cases)
        )
        print(f"MapCoder imports OK; dataset: {len(dataset)} tasks; no API calls.")
        return

    read_env_file(args.env)
    requested_model = os.getenv("QWEN_MODEL", "").strip()
    base_url = os.getenv("QWEN_BASE_URL", "").strip()
    api_key = (os.getenv("QWEN_API_KEY") or os.getenv("QWEN_API") or "").strip()
    model_name = normalize_qwen_model(requested_model, base_url)
    if not model_name or not base_url or not api_key:
        raise ValueError("Configure QWEN_API_KEY, QWEN_MODEL, and QWEN_BASE_URL")
    print(f"Dataset: {len(rows)} tasks; model ID: {model_name}; final cases: {args.max_cases or 'all'}")

    settings = {
        "input": str(input_path), "model": model_name, "thinking": args.thinking,
        "max_tokens": args.max_tokens, "max_cases": args.max_cases,
        "case_timeout": args.case_timeout, "api_attempts": args.api_attempts,
        "mapcoder_root": str(mapcoder_root), "k": 3, "t": 5,
    }
    previous = load_previous(results_path, settings)
    class AdaptedMapCoder(MapCoder):
        parse_code = staticmethod(code_from_response)
        replace_tag = staticmethod(preserve_or_wrap_cdata)

    judge = LocalAPPSJudge(timeout=args.case_timeout, max_cases=args.max_cases)
    dataset = make_dataset_class(APPSDataset)(rows, judge)
    target = dataset.data[:args.limit] if args.limit else dataset.data
    for index, item in enumerate(target, 1):
        pid = item["id"]
        if pid in previous:
            print(f"[{index}/{len(target)}] {pid}: already saved")
            continue
        model = QwenModel(
            api_key=api_key, base_url=base_url, model=model_name,
            thinking=args.thinking, max_tokens=args.max_tokens, timeout=args.api_timeout,
            attempts=args.api_attempts,
        )
        strategy = AdaptedMapCoder(model=model, data=dataset, language="Python3", pass_at_k=1,
                                   results=None, verbose=False, k=3, t=5)
        trace_path = results_path.parent / (results_path.stem + "_traces") / f"{pid:04d}.log"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        record: dict[str, Any] = {"id": pid, "difficulty": item["difficulty"], "settings": settings}
        with trace_path.open("w", encoding="utf-8") as trace, contextlib.redirect_stdout(trace):
            try:
                code, _, _ = strategy.run_single_pass(item)
                code = code_from_response(code)
                solved = dataset.evaluate(item, code, "Python3")
                record.update({"status": "completed", "is_solved": solved,
                               "code": code, "evaluation": judge.last_result})
            except OutputTruncatedError as exc:
                print(f"Output truncated: {exc}")
                record.update({"status": "output_truncated", "is_solved": False,
                               "error": str(exc)})
            except APIRequestError as exc:
                print(f"API error: {exc}")
                raise
            except Exception as exc:
                print(f"Method error: {type(exc).__name__}: {exc}")
                record.update({"status": "method_error", "is_solved": False,
                               "error": f"{type(exc).__name__}: {exc}"})
        record.update({
            "api_calls": model.calls,
            "prompt_tokens": model.prompt_tokens,
            "completion_tokens": model.completion_tokens,
            "served_models": sorted(model.served_models),
            "api_requests": model.requests,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "trace": str(trace_path),
        })
        append_result(results_path, record)
        previous[pid] = record
        print(f"[{index}/{len(target)}] {pid} {item['difficulty']}: {record['status']}, "
              f"solved={record['is_solved']}, calls={model.calls}", flush=True)
    print_summary(previous, rows)
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
