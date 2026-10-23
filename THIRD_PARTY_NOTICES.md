# Third-party code and benchmark data

## CodeSim

The APPS strategy subset under `baselines/codesim/upstream/` comes from
[KAGNLP CodeGenerator](https://github.com/kagnlp/CodeGenerator). Its original
MIT copyright notice is retained in that directory's `LICENSE`.

Only modules required by the APPS strategy imports are included. The unrelated
xCodeEval database is optional at import time; the local APPS adapter supplies
the evaluator. Model-provider implementations and unrelated datasets are not
included. Adapter code selects competitive mode for the sampled dataset and
uses the local fenced-code parser.

## MapCoder

The APPS strategy subset under `baselines/mapcoder/upstream/` comes from
[MapCoder](https://github.com/Md-Ashraful-Pramanik/MapCoder). Its original MIT
copyright notice is retained in that directory's `LICENSE`.

The unrelated xCodeEval database is optional at import time. The local adapter
uses a subprocess APPS judge and corrects fenced-code parsing and CDATA wrapping.
The planning/retrieval/improvement strategy is retained in the vendored source.

These are copies of the available local upstream snapshots, without upstream
Git history or a claimed upstream commit ID. The distributed source itself
defines the tested implementation.

## AgentCoder

The prompt provenance subset under `baselines/agentcoder/upstream/` comes from
[AgentCoder](https://github.com/huangd1999/AgentCoder). Its source snapshot is
recorded in `UPSTREAM_COMMIT`; the supplied local license and provenance note
are preserved in `baselines/agentcoder/LICENSE-UPSTREAM`.

`tests/run_classeval_agentcoder.py` is the existing local ClassEval adapter,
relocated to use this repository's configuration and output paths. It uses its
own class prompts, trusted-test candidate selection, and at most one repair.
The shared APPS repair adapter also uses its own prompts. These adapters do not
execute the original upstream HumanEval/MBPP scripts.

## APPS

`data/apps_sampled_200.jsonl`, `data/apps_sampled_200_ids.json`, and the problem
statements/test pairs used in `benchmarks/apps_100_controlled_injections/` are
derived from [APPS](https://github.com/hendrycks/apps), *Measuring Coding
Challenge Competence With APPS*. The fixed sample records seed 2026 and its
50/100/50 difficulty selection. Original problem URLs remain in the records.
The upstream repository's MIT notice is retained in `data/LICENSE-APPS.txt`;
original problem-source terms remain applicable. AuditCoder's root license
does not replace third-party data or solution rights.

The controlled corpus includes APPS community solutions and previously
generated candidates, with provenance in `manifest.json`. It also explicitly
identifies three stored-test lookup fallbacks; see the testing guide.

## ClassEval

`data/ClassEval_data.json` comes from
[ClassEval](https://github.com/FudanSELab/ClassEval). The upstream project
distributes its benchmark data under
[Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/),
as stated in its [license section](https://github.com/FudanSELab/ClassEval#license).
The dataset is retained as supplied in the source workspace. Its data terms
are separate from the MIT licenses on this repository's code.
