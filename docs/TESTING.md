# Testing and evaluation

All commands below run from the repository root. Install
`requirements-baselines.txt` for the complete offline test suite and the
CodeSim/MapCoder adapters. `requirements.txt` is sufficient for the AuditCoder
core and its synthetic smoke tests. Model commands require your own `.env` or
exported provider settings.

## 1. Offline tests

```bash
python -m pytest -q
python tests/catg_one_problem.py
python tests/controlled_fault_injection_deepseek.py --self-test
```

The unit suite checks CATG ownership and evidence history, contract types,
scope selection, callable execution, output judging, baseline imports, and
the baseline subprocess judges. The synthetic controlled-fault self-test uses
an oracle repair candidate to test repair mechanics without an LLM.
Its `expected_behavior` field checks the current routing rules, while
`localization_correct` compares against the known mutation target.

```bash
python tests/run_apps_200_auditcoder.py --dry-run
python tests/run_classeval_auditcoder.py --dry-run
python tests/apps_100_repair_benchmark.py --dry-run
python baselines/codesim/run_apps_200_deepseek.py --check
python baselines/mapcoder/run_apps_200_deepseek.py --check
python baselines/codesim/run_apps_200_qwen.py --check
python baselines/mapcoder/run_apps_200_qwen.py --check
```

Each baseline `--check` imports its actual vendored strategy and builds the
200-task dataset adapter before any credential lookup. It does not call the
model. The repair dry run checks the selected corpus files and planned run
count without starting an experiment.

## 2. CodeSim generation on APPS

The runner invokes the vendored `promptings.CodeSIM.CodeSIM` strategy:
planning, simulation, code generation, and debugging. Defaults are five
planning tries and five debug tries, configurable with `--max-plan-try` and
`--max-debug-try`. It makes one upstream strategy run per task (`pass_at_k=1`).

```bash
# One-task connectivity and generation smoke test.
python baselines/codesim/run_apps_200_deepseek.py --limit 1 \
  --results results/codesim_smoke.jsonl

# All 200 tasks, first ten stored test pairs per final evaluation.
python baselines/codesim/run_apps_200_deepseek.py --max-cases 10 \
  --results results/codesim_apps_200_deepseek.jsonl

# Separate run that evaluates every stored pair.
python baselines/codesim/run_apps_200_deepseek.py --max-cases 0 \
  --results results/codesim_apps_200_deepseek_all_cases.jsonl

# Qwen endpoint; configure QWEN_API_KEY, QWEN_MODEL and QWEN_BASE_URL.
python baselines/codesim/run_apps_200_qwen.py --max-cases 10 \
  --results results/codesim_apps_200_qwen.jsonl
```

Reissue the same command to resume. Completed task IDs already recorded in that
results file are skipped. Use a fresh results path to rerun a saved failure or
change the model, case limit, or strategy settings; incompatible saved settings
are rejected. A saved failure is not automatically regenerated.

## 3. MapCoder generation on APPS

The runner invokes the vendored `promptings.MapCoder.MapCoder` strategy with
`k=3` recalled examples and `t=5` improvement iterations. It preserves the
retrieval, planning, generation, and improvement flow. The local subclass fixes
fenced-code extraction and CDATA wrapping; the dataset adapter handles APPS
stdin and callable tasks.

```bash
python baselines/mapcoder/run_apps_200_deepseek.py --limit 1 \
  --results results/mapcoder_smoke.jsonl

python baselines/mapcoder/run_apps_200_deepseek.py --max-cases 10 \
  --results results/mapcoder_apps_200_deepseek_corrected.jsonl

python baselines/mapcoder/run_apps_200_deepseek.py --max-cases 0 \
  --results results/mapcoder_apps_200_deepseek_all_cases.jsonl

python baselines/mapcoder/run_apps_200_qwen.py --max-cases 10 \
  --results results/mapcoder_apps_200_qwen.jsonl
```

Resume behavior is the same as CodeSim: rerun the identical command and keep
the same results path. Final APPS checks run in local Python subprocesses;
the upstream external evaluation server is not needed.

### Baseline inputs, outputs, and scoring

Both adapters use the bundled 200-task sample: 50 introductory, 100 interview,
and 50 competition tasks. The first two APPS pairs are exposed as examples
during generation and debugging. The final judge evaluates the first ten
stored pairs by default, including those examples; `--max-cases 0` uses all
pairs. This is the existing local-sample protocol, not an official APPS
leaderboard evaluation with a separate hidden test split. Reference solutions
are not included in model prompts.

The CodeSim adapter explicitly enables its competitive-programming mode for
the sampled dataset subclass and uses the shared fenced-code parser. Extra
generated-IO checking is disabled for this competitive dataset, as in the
upstream flow. The vendored imports do not require the unrelated xCodeEval
test database; this distribution supports these APPS adapters.

Each results file contains one JSON record per processed problem. Inspect:

| Field | Interpretation |
| --- | --- |
| `id`, `difficulty` | Fixed sample identity and stratum |
| `status` | `completed`, `method_error`, or `output_truncated` |
| `is_solved` | Final candidate passes every selected pair |
| `code`, `evaluation` | Candidate and local judge result for completed tasks |
| `settings` | Requested model ID, budgets, case limit, and run paths |
| `served_models` | Provider-reported model IDs; inspect alongside requested ID |
| `api_calls`, `prompt_tokens`, `completion_tokens` | Recorded model usage |
| `trace` | Local diagnostic log path |

For a completed full run, report solved tasks divided by 200 and the
50/100/50 stratum counts. The console marks incomplete runs explicitly.
API request failures stop the generation runner without adding a scored task;
saved method errors and truncated outputs count as unsolved. Check statuses
before reporting a score. Trace logs live beside the JSONL file in
`<results_stem>_traces/`.

## 4. Shared 100-case controlled repair experiment

The shipped corpus contains the 100 interview tasks from the fixed sample.
Each task has the same mutant for all four methods. Faults are assigned
cyclically: 25 runtime errors, 25 wrong answers, 25 interface exceptions, and
25 timeouts. It includes all question copies, test pairs, mutants, and source
provenance, so no full APPS download is needed to run it.

```bash
# Offline validation for CodeSim and MapCoder only.
python tests/apps_100_repair_benchmark.py --methods codesim,mapcoder --dry-run

# One case, two repair adapters; makes model calls.
python tests/apps_100_repair_benchmark.py --provider deepseek \
  --methods codesim,mapcoder --limit 1 --workers 1 \
  --output-dir results/repair_smoke

# All 100 cases and all four adapters.
python tests/apps_100_repair_benchmark.py --provider deepseek --workers 4 \
  --output-dir results/apps_100_repair_benchmark

# Resume; rerun rows that ended in API/error status.
python tests/apps_100_repair_benchmark.py --provider deepseek --workers 4 \
  --output-dir results/apps_100_repair_benchmark --resume --retry-errors
```

Qwen uses the same commands with `--provider qwen`. Keep the same provider,
model configuration, selection, and budgets when resuming. Use a separate output
directory for a changed protocol. The runner writes `results.jsonl` and
`summary.json`; inspect `repair_success`, `final_all_passed`, `attempt_count`,
and error statuses. Unlike the generation baselines, `--resume` is required to
skip saved repair records.

| Adapter | Repair budget |
| --- | --- |
| AuditCoder | One regeneration using deterministic localization and scope checks |
| CodeSim | Up to five simulation/debug repair prompts |
| MapCoder | Up to five mapping/planning improvement prompts |
| AgentCoder | One programmer repair after an independent test-design call |

These are repair adapters implemented in `tests/apps_100_repair_benchmark.py`;
the CodeSim/MapCoder adapters do not call their upstream strategy classes.
They receive execution feedback from the stored test pairs after each
candidate. This measures repair on a shared injected fault, independently of
initial generation quality. Budgets differ by method and must be reported.

Clean-source provenance is retained in the corpus manifest: 78 APPS references,
18 previous CodeSim candidates, one previous MapCoder candidate, and three
test-set-bounded fallbacks (APPS 0583, 2685, 2911). Those three programs contain
test-input lookup behavior and support only a stored-test diagnostic; report
that fact when using the complete corpus. No claim of general algorithmic
correctness follows from their stored-test success. Clean references are not
sent to the repair model. Historical trace paths were cleared for anonymity;
source code and recorded source hashes are preserved.

## 5. AuditCoder APPS and ClassEval

```bash
python tests/run_apps_200_auditcoder.py --provider deepseek --count 1 \
  --output results/auditcoder_smoke.jsonl
python tests/run_apps_200_auditcoder.py --provider deepseek --max-cases 10 \
  --output results/apps_200_auditcoder.jsonl --resume
python tests/run_apps_200_once_retry.py --provider deepseek --max-cases 10 \
  --output results/apps_200_auditcoder_one_retry.jsonl --resume
```

The ordinary APPS runner and the dedicated one-retry runner have different
whole-task budgets. The latter fixes one bounded repair per pipeline run and
at most one full-task retest. Selected dataset cases are trusted checks inside
the final verifier; generated development checks are advisory. The core also
retains statement-specific templates in `canonical_solutions.py`. These
protocol details must accompany comparisons with the generation baselines,
which use only their first two pairs for model-visible execution feedback.

ClassEval task suites need additional libraries:

```bash
python -m pip install -r requirements-classeval.txt
python tests/run_classeval_auditcoder.py --dry-run
python tests/run_classeval_auditcoder.py --provider deepseek --ids ClassEval_0 \
  --output results/classeval_smoke.jsonl
python tests/run_classeval_auditcoder.py --provider deepseek --count 100 \
  --output results/classeval_auditcoder.jsonl --resume
```

Each class has at most one bounded method repair. The ClassEval unittest suite
is treated as a trusted specification and is available to this runner; it is
not a hidden-test evaluation. ClassEval data is CC BY-NC 4.0. See
[data attribution](../data/README.md).

## 6. Optional corpus rebuilding

The shipped corpus is sufficient for the repair runs above. The older rebuild
and policy-comparison scripts additionally need the full raw APPS test folder
(`0000/`, `0001/`, ...), available through the
[APPS project](https://github.com/hendrycks/apps).

```bash
python tests/build_apps_100_controlled_injections.py --apps-root /path/to/APPS/test \
  --output-dir results/corpus_rebuilt
python tests/apps_controlled_repair_comparison.py --apps-root /path/to/APPS/test \
  --output-dir results/policy_comparison
```

The builder optionally reads native generation records from `results/` using
the default DeepSeek filenames above. Different available candidates can
change clean-source selection and validation timings. Use the shipped corpus
for the frozen 100-case protocol; rebuilding from a raw APPS download alone
does not promise byte-identical historical source selection.

Generated code is executed by local subprocesses. Run model experiments in a
disposable environment. Local results, logs, and `.env` are ignored by Git;
review their contents before sharing them separately.

## 7. AgentCoder-specific tests

The release supports two AgentCoder-style experiments: repair of the shared
APPS mutants, and class generation on ClassEval. Both use the local API client
and dataset judges. They use task-specific role prompts; the designer produces
a text test plan. They do not execute the original HumanEval/MBPP scripts or
select candidates using independently generated executable tests.

### APPS controlled repair

```bash
# Offline: 100 cases, one method, 100 planned runs, zero API calls.
python tests/apps_100_repair_benchmark.py --methods agentcoder --dry-run

# One-case online smoke run.
python tests/apps_100_repair_benchmark.py --provider deepseek \
  --methods agentcoder --limit 1 --workers 1 \
  --output-dir results/agentcoder_repair_smoke

# All 100 cases, including all four fault types.
python tests/apps_100_repair_benchmark.py --provider deepseek \
  --methods agentcoder --workers 4 \
  --output-dir results/agentcoder_repair_deepseek

# Continue the same run and retry API/error records.
python tests/apps_100_repair_benchmark.py --provider deepseek \
  --methods agentcoder --workers 4 \
  --output-dir results/agentcoder_repair_deepseek --resume --retry-errors

# Qwen uses its own configuration and output directory.
python tests/apps_100_repair_benchmark.py --provider qwen \
  --methods agentcoder --workers 4 \
  --output-dir results/agentcoder_repair_qwen
```

Per case, an independent designer receives the task and public examples, then
one programmer repair receives its plan, the faulty candidate, and execution
feedback. The repair budget is fixed at one; API transport retries are separate
from that budget. A failed designer request is recorded and the programmer may
continue using execution feedback. No clean reference is sent to the model.

Inspect `test_designer.used` and `test_designer.api.error`, `attempt_count`
(at most one), `repair_success`, `final_all_passed`, and `status` in
`results.jsonl`. The summary is `summary.json`, with
`by_method.agentcoder.repair_success_rate` and per-fault counts under
`by_method.agentcoder.fault_types`. Verify 100 recorded rows and inspect
`api_errors` before interpreting a complete run. Non-AuditCoder
`localization_correct` values are adapter placeholders, not measured
ground-truth fault-localization accuracy.

### ClassEval generation and bounded repair

```bash
python -m pip install -r requirements-classeval.txt

# Offline data and prompt-provenance check; no credentials required.
python tests/run_classeval_agentcoder.py --dry-run

# One class with the default five candidates and at most one repair.
python tests/run_classeval_agentcoder.py --provider deepseek --ids ClassEval_0 \
  --output results/agentcoder_classeval_smoke.jsonl

python tests/run_classeval_agentcoder.py --provider deepseek --count 100 \
  --candidates 5 --max-retries 1 \
  --output results/agentcoder_classeval_deepseek.jsonl

# Resume skips all saved task IDs, including saved errors.
python tests/run_classeval_agentcoder.py --provider deepseek --count 100 \
  --candidates 5 --max-retries 1 \
  --output results/agentcoder_classeval_deepseek.jsonl --resume

python tests/run_classeval_agentcoder.py --provider qwen --count 100 \
  --candidates 5 --max-retries 1 \
  --output results/agentcoder_classeval_qwen.jsonl
```

The designer does not receive the unittest suite. Programmer prompts do, and
the executor scores each candidate with that suite. The best candidate is
selected; if it fails, one repair is allowed and retained only when its score
strictly improves. The normal model budget is one designer call plus five
candidate calls and, if needed, one repair call. `--candidates` changes the
pool size; `--max-retries 0` disables repair. Only zero or one repair is allowed.

The JSONL records include `candidate_scores`, `selected_candidate`,
`retry_count`, `final_tests.all_passed`, `final_class_code`, usage, and statuses.
The sibling `.summary.json` includes `overall.problem_accuracy` and
`overall.method_accuracy`; report candidate count and repair budget with them.
Logs are in `<output_stem>_traces/`. This is evaluation and selection using
trusted ClassEval tests, so the score is not a hidden-test pass@1 result.

For this ClassEval adapter, omitting `--resume` resets the output file. Keep
the same settings when resuming; use a fresh path to rerun saved errors or
change the protocol. Configuration and output defaults are repository-local,
and the adapter reuses the bundled ClassEval harness. See
[the AgentCoder directory](../baselines/agentcoder/README.md) for provenance.

