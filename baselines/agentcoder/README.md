# AgentCoder testing

Two executable adapters are included:

| Task | Entry point | Protocol |
| --- | --- | --- |
| APPS shared controlled repair | `tests/apps_100_repair_benchmark.py --methods agentcoder` | Independent test-plan call, one programmer repair, local stored-test evaluation |
| ClassEval generation | `tests/run_classeval_agentcoder.py` | Independent test-plan call, five programmer candidates by default, trusted-test selection, at most one repair |

From the repository root, validate both selections without API credentials:

```bash
python tests/apps_100_repair_benchmark.py --methods agentcoder --dry-run
python tests/run_classeval_agentcoder.py --dry-run
```

Install `requirements.txt` for the repair adapter and
`requirements-classeval.txt` for ClassEval execution. Configure DeepSeek or Qwen
in the repository-root `.env` using `.env.example`.

Both adapters use their own task-specific prompts and the common local model
client. The independent designer produces a text test plan. ClassEval candidate
selection uses the trusted benchmark unittest suites. These protocols differ
from the original AgentCoder HumanEval/MBPP generated-test selection loop.

`upstream/prompts/humaneval_prompt_update.txt` is retained for provenance from
[AgentCoder](https://github.com/huangd1999/AgentCoder); it is checked as a
provenance file by the ClassEval adapter, rather than inserted into its class
prompts. `upstream/UPSTREAM_COMMIT` records the source snapshot. The legacy
upstream runtime is not required by these adapters. The original local
[license/provenance notice](LICENSE-UPSTREAM) is retained.

Single-task, full-run, DeepSeek/Qwen, resume, budgets, and output instructions:

- [中文：AgentCoder 专项测试](../../docs/TESTING_zh.md#7-agentcoder-专项测试)
- [English: AgentCoder-specific tests](../../docs/TESTING.md#7-agentcoder-specific-tests)
