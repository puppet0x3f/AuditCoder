# Benchmark inputs

| File | Contents |
| --- | --- |
| `apps_sampled_200_ids.json` | Seed 2026 and the fixed 50 introductory / 100 interview / 50 competition APPS IDs |
| `apps_sampled_200.jsonl` | The same 200 APPS tasks, statements, starter code, and stored input/output pairs |
| `ClassEval_data.json` | The 100 ClassEval class tasks, specifications, reference material, and unittest suites |

These are copies of the existing benchmark files; samples are not redrawn at
runtime. The controlled-fault corpus includes its own question and test copies
under `benchmarks/`, so it does not depend on a full APPS installation.

The optional raw APPS test directory can be placed at `data/APPS/test/` or
passed to rebuild scripts with `--apps-root`. It is excluded from Git.

APPS: [project and citation](https://github.com/hendrycks/apps), with the upstream
repository license reproduced in [LICENSE-APPS.txt](LICENSE-APPS.txt). Original
problem URLs and solution provenance are retained.

ClassEval: [project and citation](https://github.com/FudanSELab/ClassEval), with
benchmark data under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
The data file is retained without content changes. This attribution and
license link apply to that data independently of the root code license.
