# AuditCoder: Process-Auditable Code Generation through Responsibility-Preserving Task Graphs

Anonymous code release accompanying the paper **AuditCoder: Process-Auditable
Code Generation through Responsibility-Preserving Task Graphs**.

## Overview

Plans can guide code generation without recording which code implements each
planned subproblem. This **decision–code responsibility misalignment** leaves
reviewers to reconstruct those connections during handoff, maintenance, and
failure diagnosis.

AuditCoder makes the responsibility record an explicit output of code
generation. It jointly constructs a program and an inspectable trace linking
planning commitments to their implementations, versioned check results, and
repair history through **Contract-Annotated Task Graphs (CATG)**.

The paper makes three contributions:

- **Process-auditable code generation:** a formulation that makes the links
  between planning decisions, delivered code, verification evidence, and later
  changes available for human inspection and automated record audits.
- **Responsibility-preserving construction and bounded repair:** stable IDs and
  explicit ownership connect planning to implementation and support declared
  node or branch repair scopes, recorded expansions, and unchanged code outside
  the final scope.
- **Evaluation beyond final correctness:** benchmark comparisons, a construction
  record audit, and controlled fault injection measure functional performance,
  record completeness, fault-owner containment, and repair decisions.

## Method

![AuditCoder framework: responsibility-preserving generation and bounded repair](docs/assets/auditcoder-framework.svg)

*Figure 2 from the paper. Responsibility-preserving generation and repair:
(a) the audit workflow; (b) decision–code–repair correspondence.*

The paper's workflow connects four stages through the same responsibility IDs:

1. **Plan the CATG.** Before implementation, assign stable IDs to subproblems
   and record goals, contracts, dependencies, input provenance, complexity
   budgets, and local check requirements.
2. **Generate and assemble owned code.** Algorithm-specialized agents implement
   the planned responsibilities. Assembly preserves ownership for registered
   executable units, including helper functions and orchestration code.
3. **Verify and localize.** Attach findings to responsibility IDs and code
   versions, distinguishing execution evidence from verifier judgments.
   Deterministic rules use tracebacks, resource evidence, and compatible local
   findings to identify implicated responsibilities.
4. **Repair within a declared scope.** Regenerate a supported node or replan a
   supported branch, recording any scope expansion before changing additional
   owners. Preserve code outside the final scope, rerun the fixed development
   checks, and accept a candidate only on strict improvement under the recorded
   ordering; otherwise restore the prior state. Abstain when no supported
   boundary can be selected.

## Results reported in the paper

### Functional performance

Table 2 reports **fixed-run pass@1 (%)** with at most one feedback-conditioned
revision per task. APPS uses a fixed 200-task sample (50 introductory, 100
interview, and 50 competition tasks); ClassEval uses 100 records per model–method
run. DS denotes `deepseek-v4-flash`; QW denotes `Qwen3.6-plus`.

| Method | APPS (DS) | APPS (QW) | ClassEval (DS) | ClassEval (QW) |
| --- | ---: | ---: | ---: | ---: |
| AgentCoder | 65.0 | 70.0 | 80.0 | 82.0 |
| MapCoder | 76.5 | 80.0 | 89.0 | 92.0 |
| CodeSim | 87.0 | 88.5 | **91.0** | **95.0** |
| **AuditCoder** | **89.0** | **89.5** | 90.0 | 92.0 |

AuditCoder remains competitive with CodeSim while producing responsibility
records for inspection and scoped repair. In the separate adaptive multi-round
APPS setting, AuditCoder reaches **94.5% / 97.5%** with DS / QW, respectively.

### Process audit and controlled repair

Tables 3 and 4 examine the responsibility records and repair behavior:

- **Record completeness:** in the 200-task DS APPS audit, a task-averaged
  **97.25%** of final leaf records retain nonempty code, contract, provenance,
  and validation fields.
- **Natural failures:** **44/52** initially failing tasks receive a node or
  branch repair scope, and **30/44** repairs pass held-out tests. Scopes average
  **22.1% of graph nodes**. All **44/44** attempts follow the recorded
  accept/rollback policy, and all **14/14** rollbacks restore the audited
  snapshots.
- **Controlled fault injection:** in a separate 100-task study with seven
  abstentions, manual inspection finds the injected fault's owner inside
  **88/93 (94.6%)** selected scopes.

Record completeness measures retained fields. Fault-owner containment measures
localization and is distinct from successful repair. These are paper-reported
results; see [evaluation scope](#evaluation-scope) for the shipped snapshot's
protocols and interpretation.

## Code release

This repository contains the implementation snapshot, runnable evaluation
scripts, the fixed APPS sample, ClassEval data, and the shared controlled-fault
corpus. CodeSim and MapCoder generation baselines include their upstream
strategy code and local APPS adapters. AgentCoder-style adapters support
ClassEval generation and the shared APPS controlled-repair experiment.

## Installation and offline checks

Use Python 3.11 or later. Run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-baselines.txt
python -m pytest -q
python tests/catg_one_problem.py
python tests/controlled_fault_injection_deepseek.py --self-test
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1`.

Validate the datasets and both upstream baseline imports without credentials:

```bash
python tests/run_apps_200_auditcoder.py --dry-run
python tests/run_classeval_auditcoder.py --dry-run
python baselines/codesim/run_apps_200_deepseek.py --check
python baselines/mapcoder/run_apps_200_deepseek.py --check
python tests/apps_100_repair_benchmark.py --methods codesim,mapcoder --dry-run
python tests/apps_100_repair_benchmark.py --methods agentcoder --dry-run
python tests/run_classeval_agentcoder.py --dry-run
```

## Model configuration

```bash
cp .env.example .env
```

Fill in the API key, model ID, and base URL of your provider. The runners accept
`DEEPSEEK_API_KEY` / `QWEN_API_KEY` and the legacy `DEEPSEEK_API` / `QWEN_API`
names. Shell settings take precedence over `.env`. Check and dry-run commands
make no model requests; generation and repair commands use the configured API.

Run a single APPS problem with each generation method:

```bash
python tests/run_apps_200_auditcoder.py --provider deepseek --count 1 \
  --output results/auditcoder_smoke.jsonl
python baselines/codesim/run_apps_200_deepseek.py --limit 1 \
  --results results/codesim_smoke.jsonl
python baselines/mapcoder/run_apps_200_deepseek.py --limit 1 \
  --results results/mapcoder_smoke.jsonl
```

For complete runs, Qwen commands, checkpointing, output interpretation, and the
controlled repair comparison, see [the testing guide](docs/TESTING.md) or
[中文测试说明](docs/TESTING_zh.md).
AgentCoder commands and protocol details are in
[its baseline guide](baselines/agentcoder/README.md).

## Repository layout

```text
auditcoder/                         CATG, planner, algorithm agents, assembler, verifiers
baselines/codesim/                 DeepSeek/Qwen adapters and vendored CodeSim strategy
baselines/mapcoder/                DeepSeek/Qwen adapters and vendored MapCoder strategy
baselines/agentcoder/              Adapter guide and upstream prompt provenance
tests/                            Unit tests and experiment entry points
data/                             Fixed APPS 200-task sample and ClassEval 100-task data
benchmarks/apps_100_controlled_injections/
                                  Shared mutants, source provenance, and stored test pairs
docs/                             Testing and anonymous publication instructions
scripts/check_anonymity.py         File-content check for local identity paths and secrets
results/                          Generated locally; excluded from Git
```

The deterministic repair rules are in
`auditcoder/final_verifier_agent/audit.py`; CATG records are in
`auditcoder/schemas/contract_graph.py`.
This snapshot also includes a low-confidence `GlobalAttribution` fallback
after traceback, resource, and local-evidence localization.

## Evaluation scope

The APPS generation runners and controlled-fault repair adapters answer
different experimental questions. The generation adapters execute the vendored
CodeSim/MapCoder strategies. The controlled experiment starts from identical
mutants and uses method-specific repair prompts and budgets; it does not rerun
those complete generation strategies.

## Anonymous publication and attribution

See [anonymous publication instructions](docs/ANONYMITY.md). This folder has no
Git history, private `.env`, author contact metadata, manuscript, or historical
run logs. Third-party copyright notices remain intact.

AuditCoder code uses the included [MIT license](LICENSE). Vendored baseline
licenses and dataset terms are listed in [third-party notices](THIRD_PARTY_NOTICES.md).
ClassEval data uses CC BY-NC 4.0 and is not covered by the code's MIT license.
