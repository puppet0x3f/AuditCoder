from __future__ import annotations

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = (
    REPO_ROOT / "experiments" / "cross_domain_real_world_20260726"
)
AGENTCODER_ROOT = REPO_ROOT / "baselines" / "agentcoder" / "upstream"
QF_AGENTCODER_RUNNER = (
    EXPERIMENT_ROOT
    / "qf_small6_dataset"
    / "tools"
    / "run_agentcoder_qf_small6.py"
)

SUITES = {
    "qf_small6_dataset": {
        "run_auditcoder_qf_small6.py",
        "run_agentcoder_qf_small6.py",
        "run_portable_oracle.py",
        "validate_dataset.py",
    },
    "scicode_blind8_dataset": {
        "build_dataset.py",
        "evaluate_candidate.py",
        "reverify_run.py",
        "run_auditcoder_scicode_blind8.py",
        "run_agentcoder_scicode_blind8.py",
        "run_evaluator_smoke.py",
        "scicode_compat.py",
        "validate_dataset.py",
    },
    "aerocomm_small8_dataset": {
        "evaluate_candidate.py",
        "reverify_run.py",
        "run_auditcoder_aerocomm.py",
        "run_agentcoder_aerocomm.py",
        "run_negative_control.py",
        "run_oracle.py",
        "validate_dataset.py",
    },
}


def test_all_three_cross_domain_code_packages_are_present() -> None:
    for suite, expected_tools in SUITES.items():
        suite_root = EXPERIMENT_ROOT / suite
        assert suite_root.is_dir(), suite
        actual_tools = {path.name for path in (suite_root / "tools").glob("*.py")}
        assert expected_tools <= actual_tools, (
            suite,
            sorted(expected_tools - actual_tools),
        )

    assert (
        EXPERIMENT_ROOT
        / "aerocomm_small8_dataset"
        / "private"
        / "build_dataset.py"
    ).is_file()


def test_agentcoder_upstream_snapshot_is_code_only() -> None:
    commit = (AGENTCODER_ROOT / "UPSTREAM_COMMIT").read_text().strip()
    assert commit == "d8538d63675c2b805855e6deb7ddc39fefb5723d"

    assert len(list((AGENTCODER_ROOT / "src").glob("*.py"))) == 6
    assert len(list((AGENTCODER_ROOT / "prompts").glob("*.txt"))) == 16
    assert not (AGENTCODER_ROOT / "dataset").exists()
    assert not (AGENTCODER_ROOT / "images").exists()
    assert not (AGENTCODER_ROOT / "CodeGeeX").exists()

    forbidden_suffixes = {".csv", ".h5", ".json", ".jsonl", ".sqlite", ".xlsx"}
    assert not [
        path
        for path in AGENTCODER_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]


def test_agentcoder_adapters_use_the_bundled_snapshot() -> None:
    adapter_paths = [
        QF_AGENTCODER_RUNNER,
        EXPERIMENT_ROOT
        / "scicode_blind8_dataset"
        / "tools"
        / "run_agentcoder_scicode_blind8.py",
        EXPERIMENT_ROOT
        / "aerocomm_small8_dataset"
        / "tools"
        / "run_agentcoder_aerocomm.py",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in adapter_paths)
    assert "/Users/" not in text
    assert "baselines\" / \"agentcoder\" / \"upstream" in text


def test_cross_domain_runners_have_no_machine_specific_interpreter() -> None:
    runner_paths = [
        *EXPERIMENT_ROOT.glob(
            "scicode_blind8_dataset/tools/run_*coder_scicode_blind8.py"
        ),
        *EXPERIMENT_ROOT.glob(
            "aerocomm_small8_dataset/tools/run_*coder_aerocomm.py"
        ),
    ]
    assert len(runner_paths) == 4
    text = "\n".join(
        path.read_text(encoding="utf-8") for path in runner_paths
    )
    assert "/opt/anaconda3" not in text
    assert "Path(sys.executable).resolve()" in text


def test_qf_retained_candidate_path_relocation_is_scoped(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "auditcoder_qf_agentcoder_runner",
        QF_AGENTCODER_RUNNER,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    task_id = "earnings-surprise-calculator"
    task_run_dir = tmp_path / "restored-run" / "tasks" / task_id
    task_run_dir.mkdir(parents=True)
    recorded = Path("/old/checkout/runs/example/tasks") / task_id / "app"
    (task_run_dir / "prompt.txt").write_text(
        f"The application directory is:\n`{recorded}`\n"
    )
    code = f"INPUT = {str(recorded / 'input.csv')!r}\n"

    relocated, evidence = module.relocate_recorded_app_path(
        code,
        task_run_dir,
        task_id,
    )

    assert str(recorded) not in relocated
    assert str((task_run_dir / "app").resolve()) in relocated
    assert evidence["applied"] is True
    assert evidence["replacement_count"] == 1
    assert evidence["original_candidate_sha256"] != (
        evidence["evaluated_candidate_sha256"]
    )
    assert code == f"INPUT = {str(recorded / 'input.csv')!r}\n"


def test_readme_documents_agentcoder_execution_and_reverification() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    required = {
        "## AgentCoder Baseline",
        "run_agentcoder_qf_small6.py",
        "run_agentcoder_scicode_blind8.py",
        "run_agentcoder_aerocomm.py",
        "--evaluate-only",
        "reverify_run.py",
    }
    assert all(value in readme for value in required)
