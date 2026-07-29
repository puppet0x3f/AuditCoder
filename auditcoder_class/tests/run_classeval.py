"""ClassEval dataset batch test runner.

Run the auditcoder_class pipeline on ClassEval problems and save per-problem JSON results.

Usage:
  # Run first 5 problems (index 0-4)
  python auditcoder_class/tests/run_classeval.py --start 0 --count 5

  # Run problems at specific indices
  python auditcoder_class/tests/run_classeval.py --ids ClassEval_0,ClassEval_5

  # Run index range 0-19
  python auditcoder_class/tests/run_classeval.py --start 0 --end 19

  # Custom output directory
  python auditcoder_class/tests/run_classeval.py --start 0 --count 3 --output data/my_results
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)
os.environ["AUDITCODER_LLM_PROVIDER"] = "qwen"

from auditcoder_class.class_plan_agent.graph import build_class_plan_agent_graph
from auditcoder_class.method_agents.dispatcher import dispatch_method_agents
from auditcoder_class.class_assembler_agent.agent import build_class_assembler_graph
from auditcoder_class.class_final_verifier_agent.graph import build_class_final_verifier_graph
from auditcoder_class.class_final_verifier_agent.nodes.error_locator import error_locator
from auditcoder_class.class_final_verifier_agent.nodes.method_retrier import method_retrier

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLASSEVAL_DATA_PATH = PROJECT_ROOT / "data" / "ClassEval_data.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "classeval_results"

MAX_OUTER_RETRIES = 2


def load_classeval(path: Path) -> dict[str, dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {item["task_id"]: item for item in data}


def select_problems(
    problems: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    all_ids = sorted(problems.keys())

    if args.ids:
        selected_ids = [s.strip() for s in args.ids.split(",")]
        return [problems[i] for i in selected_ids if i in problems]

    start_idx = args.start or 0
    end_idx = args.end if args.end is not None else len(all_ids)
    selected = all_ids[start_idx:end_idx]

    if args.count:
        selected = selected[: args.count]

    return [problems[i] for i in selected]


def run_pipeline(entry: dict[str, Any]) -> dict[str, Any]:
    """Run the full auditcoder_class pipeline on one ClassEval entry."""
    # Step 1: class plan agent
    plan_result = build_class_plan_agent_graph().invoke({"raw_entry": entry})
    ir = plan_result["class_plan_ir"]

    # Step 2: method agents (with per-method verification)
    method_bodies, per_method_reports = dispatch_method_agents(ir)

    # Step 3: class assembler (rule-based skeleton injection + structural smoke check)
    assembler_result = build_class_assembler_graph().invoke({
        "class_plan_ir": ir,
        "method_bodies": method_bodies,
    })
    final_class_code = assembler_result["final_class_code"]
    structural_violations = assembler_result.get("structural_violations", [])

    # Step 4: final verifier (full test suite + optional retry)
    final_result = build_class_final_verifier_graph().invoke({
        "final_class_code": final_class_code,
        "class_plan_ir": ir,
        "method_bodies": method_bodies,
        "per_method_reports": per_method_reports,
        "structural_violations": structural_violations,
        "retried_methods": [],
    })

    return {
        "ir": ir,
        "method_bodies": method_bodies,
        "per_method_reports": per_method_reports,
        "final_class_code": final_result.get("final_class_code", final_class_code),
        "final_report": final_result.get("final_report", {}),
        "structural_violations": structural_violations,
    }


def run_classeval_evaluation(
    final_class_code: str,
    entry: dict[str, Any],
) -> dict[str, Any]:
    """Run the full test suite and per-method tests directly for result metrics."""
    from auditcoder.verifier_agent.sandbox.executor import SandboxExecutor

    executor = SandboxExecutor(default_timeout_s=30.0)
    imports = entry.get("import_statement", [])
    test_code = entry.get("test", "")
    test_classes = entry.get("test_classes", [])
    methods_info = entry.get("methods_info", [])

    if not test_code:
        return {"all_passed": True, "test_count": 0, "pass_count": 0, "method_pass_rates": {}}

    # Full test suite
    script_parts = []
    if imports:
        script_parts.append("\n".join(imports))
    script_parts.append(final_class_code)
    script_parts.append("import unittest")
    script_parts.append(test_code)
    script_parts.append(f"""
if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls_name in [{', '.join(repr(c) for c in test_classes)}]:
        try:
            cls = globals()[cls_name]
            suite.addTests(loader.loadTestsFromTestCase(cls))
        except KeyError:
            pass
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    import sys
    sys.exit(0 if result.wasSuccessful() else 1)
""")
    script = "\n\n".join(script_parts)
    result = executor.run(script)
    all_passed = result.exit_code == 0 and not result.timed_out

    # Count total tests from output
    ran_match = re.search(r'Ran (\d+) test', result.stdout)
    test_count = int(ran_match.group(1)) if ran_match else 0

    # Per-method pass rates by running each TestCase individually
    method_pass_rates: dict[str, bool] = {}
    tc_to_method = {m.get("test_class", ""): m["method_name"] for m in methods_info}

    for tc_name in test_classes:
        method_name = tc_to_method.get(tc_name, tc_name)
        per_script_parts = []
        if imports:
            per_script_parts.append("\n".join(imports))
        per_script_parts.append(final_class_code)
        per_script_parts.append("import unittest")
        per_script_parts.append(test_code)
        per_script_parts.append(f"""
if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase({tc_name})
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    import sys
    sys.exit(0 if result.wasSuccessful() else 1)
""")
        per_script = "\n\n".join(per_script_parts)
        per_result = executor.run(per_script)
        method_pass_rates[method_name] = per_result.exit_code == 0 and not per_result.timed_out

    pass_count = sum(1 for v in method_pass_rates.values() if v)

    return {
        "all_passed": all_passed,
        "test_count": test_count,
        "pass_count": pass_count,
        "method_pass_rates": method_pass_rates,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def run_single_problem(
    entry: dict[str, Any],
    args: argparse.Namespace,
    max_retries: int = MAX_OUTER_RETRIES,
) -> dict[str, Any]:
    task_id = entry["task_id"]
    if not args.quiet:
        print(f"\n{'='*60}")
        print(f"[{task_id}] {entry.get('class_name', '')}")
        print(f"{'='*60}")

    result: dict[str, Any] = {"task_id": task_id, "class_name": entry.get("class_name", "")}

    try:
        pipeline = run_pipeline(entry)
        final_class_code = pipeline["final_class_code"]
        ir = pipeline["ir"]
        method_bodies = pipeline["method_bodies"]
        per_method_reports = pipeline["per_method_reports"]

        eval_result = run_classeval_evaluation(final_class_code, entry)

        # Outer retry loop — mirrors the replan loop in run_apps_global_replan
        outer_retry_count = 0
        retried_methods: list[str] = []
        structural_violations: list[dict[str, Any]] = pipeline.get("structural_violations", [])

        while not eval_result.get("all_passed", False) and outer_retry_count < max_retries:
            fv_state: dict[str, Any] = {
                "final_class_code": final_class_code,
                "class_plan_ir": ir,
                "method_bodies": method_bodies,
                "per_method_reports": per_method_reports,
                "structural_violations": structural_violations,
                "execution_result": {
                    "all_passed": False,
                    "stdout": eval_result.get("stdout", ""),
                    "stderr": eval_result.get("stderr", ""),
                    "exit_code": 1,
                    "timed_out": False,
                },
                "retried_methods": retried_methods,
            }

            locator_result = error_locator(fv_state)
            error_loc = locator_result.get("error_location")
            if not error_loc:
                break

            # Skip methods already retried (same as graph-level guard)
            if error_loc.get("method_name") in retried_methods:
                break

            fv_state["error_location"] = error_loc

            retrier_result = method_retrier(fv_state)
            if not retrier_result:
                break

            method_bodies = retrier_result.get("method_bodies", method_bodies)
            retried_methods = retrier_result.get("retried_methods", retried_methods)

            assembler_result = build_class_assembler_graph().invoke({
                "class_plan_ir": ir,
                "method_bodies": method_bodies,
            })
            final_class_code = assembler_result["final_class_code"]
            # Update structural violations for the re-assembled code
            structural_violations = assembler_result.get("structural_violations", [])

            eval_result = run_classeval_evaluation(final_class_code, entry)
            outer_retry_count += 1

            if not args.quiet:
                retried_name = error_loc.get("method_name", "?")
                n_pass = eval_result.get("pass_count", 0)
                n_methods = len(eval_result.get("method_pass_rates", {}))
                print(f"  [Retry {outer_retry_count}] retried '{retried_name}' → {n_pass}/{n_methods} passed")

        result["classeval_test_results"] = eval_result
        result["final_class_code"] = final_class_code
        result["timestamp"] = datetime.now().isoformat()
        result["agent_outputs"] = {
            "class_plan_agent": {"class_plan_ir": ir},
            "method_agents": per_method_reports,
            "class_assembler_agent": {"final_class_code": final_class_code},
            "class_final_verifier_agent": pipeline["final_report"],
            "retry_info": {
                "outer_retry_count": outer_retry_count,
                "retried_methods": retried_methods,
            },
        }

        passed = eval_result.get("all_passed", False)
        n_methods = len(eval_result.get("method_pass_rates", {}))
        n_pass = eval_result.get("pass_count", 0)
        status = "PASS" if passed else "FAIL"
        if not args.quiet:
            print(f"Result: {status}  ({n_pass}/{n_methods} methods passed)")

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        result["classeval_test_results"] = {"all_passed": False}
        if not args.quiet:
            print(f"ERROR: {e}")

    return result


def save_result(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    task_id = result["task_id"]
    safe_id = task_id.replace("/", "_")
    out_path = output_dir / f"{safe_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run auditcoder_class pipeline on ClassEval dataset")
    parser.add_argument("--start", type=int, default=None, help="Start index (0-based)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--count", type=int, default=None, help="Max number of problems")
    parser.add_argument("--ids", type=str, default=None, help="Specific task IDs, comma-separated")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--max-retries", type=int, default=MAX_OUTER_RETRIES, help="Max outer retry attempts per problem (default: %(default)s)")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR

    problems = load_classeval(CLASSEVAL_DATA_PATH)
    selected = select_problems(problems, args)

    if not selected:
        print("No problems selected.")
        return

    print(f"[Qwen 测试] 模型：{os.getenv('QWEN_MODEL', 'qwen-plus')}")
    print(f"Running {len(selected)} problem(s) → {output_dir}")

    passed_count = 0
    for entry in selected:
        result = run_single_problem(entry, args, max_retries=args.max_retries)
        save_result(result, output_dir)
        if result.get("classeval_test_results", {}).get("all_passed", False):
            passed_count += 1

    print(f"\nDone. {passed_count}/{len(selected)} problems passed all tests.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
