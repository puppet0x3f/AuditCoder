# Third-party notices

## AgentCoder

This repository contains a code-only snapshot of AgentCoder:

- Upstream: <https://github.com/huangd1999/AgentCoder>
- Commit: `d8538d63675c2b805855e6deb7ddc39fefb5723d`
- Included path: `baselines/agentcoder/upstream/`
- Declared upstream license: MIT
- Local license copy and provenance note:
  `baselines/agentcoder/LICENSE-UPSTREAM`

The snapshot contains selected Python sources, prompts, requirements, and the
upstream launcher. It excludes benchmark datasets, generated results, the
CodeGeeX submodule, images, credentials, and local environment files.

The reported cross-domain AgentCoder results use a protocol-preserving adapter,
not the original HumanEval/MBPP runner. AuditCoder's adapters preserve the
programmer/test-designer/test-executor roles and candidate-selection loop while
adapting generated tests to file-program and scientific-function interfaces.
The adapters are original AuditCoder integration code and remain governed by
the repository-level license.

No endorsement by the AgentCoder authors is implied.
