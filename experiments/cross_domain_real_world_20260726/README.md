# Cross-domain experiment code

This directory contains the code for the three exploratory suites reported in
the AuditCoder manuscript:

| Suite | Interface | Frozen tests | AuditCoder runner | AgentCoder runner |
| --- | --- | ---: | --- | --- |
| QF-Small6 | File program | 64 | `qf_small6_dataset/tools/run_auditcoder_qf_small6.py` | `qf_small6_dataset/tools/run_agentcoder_qf_small6.py` |
| SciCode Blind-Small-8 | Function | 24 | `scicode_blind8_dataset/tools/run_auditcoder_scicode_blind8.py` | `scicode_blind8_dataset/tools/run_agentcoder_scicode_blind8.py` |
| AeroComm-Small8 | Function | 72 | `aerocomm_small8_dataset/tools/run_auditcoder_aerocomm.py` | `aerocomm_small8_dataset/tools/run_agentcoder_aerocomm.py` |

## What is included

- deterministic builders where the required source material is available;
- dataset validators and external candidate evaluators;
- oracle and negative-control checks;
- AuditCoder runners;
- AgentCoder interface adapters;
- deterministic re-verification tools for retained final candidates.

The AgentCoder adapters use the code-only upstream snapshot under
`baselines/agentcoder/upstream/` for protocol provenance. They run from this
repository without a separate `/Users/.../AgentCoder-main` checkout.

## What is excluded

The repository intentionally excludes generated tasks, private targets,
historical candidates, raw model calls, result bundles, and run directories.
QF-Small6 additionally lacks the artifact-specific subset packager and 16
contract-addition tests, so its exact paper package must be restored before
that suite can run.

See the repository-level `README.md` for pinned downloads, build commands,
one-task smoke tests, full AuditCoder/AgentCoder runs, and result
re-verification.
