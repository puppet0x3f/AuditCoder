# APPS 100 controlled-injection corpus

This folder contains the 100 medium-difficulty (`interview`) problems from the previous fixed 200-problem APPS sample (`seed=2026`). Every problem receives exactly one deterministic fault type, assigned cyclically across four shared semantic types:

- `runtime_exception` (RE)
- `wrong_answer` (WA)
- `interface_contract` (contract violation)
- `resource_timeout` (TLE)

The cyclic assignment gives 25 cases to each type. All four method records for a case share the same fault marker and semantic mutation, while their JSON metadata identifies the method-native responsibility unit and repair flow.

The same semantic fault and the same clean reference source are materialized for AuditCoder, CodeSIM, MapCoder, and AgentCoder. Each method has separate unit metadata under `cases/*/methods/*.json` describing where that fault belongs in its native workflow: AuditCoder TaskPlanIR ownership, CodeSIM plan/simulation steps, MapCoder planning/mapping units, or AgentCoder programmer candidates and executor feedback.

Each case also includes a local `question.txt` copy and its paired
`input_output.json`, so the folder is directly usable as a shared repair-test
corpus.

`clean.py` is the clean source used for validation. It is taken from `solutions.json` when a full stored-test check succeeds, otherwise from a fully checked CodeSIM/MapCoder generation; the manifest records the provenance. The few APPS records without a usable full reference are explicitly marked as test-set-bounded fallbacks. The four method `.py` files contain the same controlled mutant so the repair comparison changes only the method's localization/retry or bounded-repair process. `manifest.json` is the single machine-readable index.

The bounded fallback cases are APPS 2911, 583, and 2685; they are retained in
the corpus with their provenance exposed rather than silently presenting them
as recovered accepted solutions.

Run the shared repair adapters from the repository root:

```bash
python tests/apps_100_repair_benchmark.py --dry-run
python tests/apps_100_repair_benchmark.py --provider deepseek \
  --methods codesim,mapcoder --limit 1 --workers 1 \
  --output-dir results/repair_smoke
```

The adapters use method-specific repair prompts and budgets; they do not
invoke the complete upstream CodeSim/MapCoder generation strategies. See
[the testing guide](../../docs/TESTING.md) for both experiments and source
provenance. Historical trace paths have been removed from metadata.

Rebuilding additionally requires raw APPS test data and, for some reference
choices, prior baseline candidates. The shipped corpus preserves the fixed
source selection; rebuilding from a different set of candidates can change it.
