# AgentCoder baseline

This directory makes the AgentCoder comparison code used by the cross-domain
experiments self-contained.

## Included source

`upstream/` is a code-only snapshot of
[`huangd1999/AgentCoder`](https://github.com/huangd1999/AgentCoder) at commit
`d8538d63675c2b805855e6deb7ddc39fefb5723d`. It contains the upstream Python
sources, prompts, requirements, and launcher, but excludes:

- benchmark datasets and generated result files;
- the `CodeGeeX` submodule;
- the pipeline image and other non-code assets;
- credentials and local environment files.

The upstream snapshot is retained for protocol provenance. Its original
HumanEval/MBPP scripts use `openai==0.28.0` and the CodeGeeX submodule. Do not
install that requirements file into the main AuditCoder environment.

The pinned upstream README identifies AgentCoder as MIT-licensed. The vendored
snapshot remains governed by its upstream license and attribution. See
`LICENSE-UPSTREAM` and the repository-level `THIRD_PARTY_NOTICES.md`; the
AuditCoder project's own MIT license is provided in the repository root.

## Paper cross-domain adapters

The executable paper adapters are:

- `experiments/cross_domain_real_world_20260726/qf_small6_dataset/tools/run_agentcoder_qf_small6.py`
- `experiments/cross_domain_real_world_20260726/scicode_blind8_dataset/tools/run_agentcoder_scicode_blind8.py`
- `experiments/cross_domain_real_world_20260726/aerocomm_small8_dataset/tools/run_agentcoder_aerocomm.py`

They preserve the upstream HumanEval protocol's central roles and selection
loop while adapting generated tests to each suite's execution interface:

1. sample five programmer candidates;
2. independently sample ten tester programs;
3. score candidates using only generated tests;
4. freeze the first maximum-scoring candidate when the score reaches three;
5. otherwise sample another pool, up to six pools;
6. invoke frozen external tests only after final candidate selection.

The reported cross-domain AgentCoder results use a protocol-preserving adapter,
not the original HumanEval/MBPP runner. The adapters run with the main
AuditCoder environment and OpenAI-compatible DeepSeek configuration. They do
not import or execute the legacy upstream scripts, so the old OpenAI client and
CodeGeeX are not required.

The main repository README contains dataset preparation, smoke-test, full-run,
and re-verification commands. Retained QF runs must keep `prompt.txt` with
`final_code.py`, because the re-verifier uses it to relocate the generation
machine's absolute application path without modifying the retained candidate.
