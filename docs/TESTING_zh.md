# 测试与复现实验说明

以下命令均在仓库根目录执行。目录内已经包含固定的 APPS 200 题样本、
ClassEval 100 个类，以及 100 题受控故障语料。CodeSim、MapCoder 所需的上游
策略代码也已收录，无需依赖另一个本地仓库。

## 1. 安装与不调用 API 的检查

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-baselines.txt
python -m pytest -q
python tests/catg_one_problem.py
python tests/controlled_fault_injection_deepseek.py --self-test
```

Windows PowerShell 使用 `.venv\Scripts\Activate.ps1` 激活环境。
代码要求 Python 3.11 或以上。

```bash
python tests/run_apps_200_auditcoder.py --dry-run
python tests/run_classeval_auditcoder.py --dry-run
python baselines/codesim/run_apps_200_deepseek.py --check
python baselines/codesim/run_apps_200_qwen.py --check
python baselines/mapcoder/run_apps_200_deepseek.py --check
python baselines/mapcoder/run_apps_200_qwen.py --check
python tests/apps_100_repair_benchmark.py --methods codesim,mapcoder --dry-run
```

`--check` 会实际导入上游策略并构造 200 题的数据适配器，无需 `.env` 或密钥。
修复实验的 `--dry-run` 会检查语料文件和任务数量。
这些检查验证程序与数据是否齐全，不产生模型性能结果。
原有故障自测的断言已与当前 `GlobalAttribution` 兜底规则对齐：
`expected_behavior` 检查规则是否按预期执行，`localization_correct` 独立记录
定位是否正确。

## 2. 配置模型

```bash
cp .env.example .env
```

填写自己可访问的模型、API 地址和密钥：

- DeepSeek：`DEEPSEEK_API_KEY`、`DEEPSEEK_MODEL`、`DEEPSEEK_BASE_URL`。
- Qwen：`QWEN_API_KEY`、`QWEN_MODEL`、`QWEN_BASE_URL`。
- AuditCoder 还支持 GLM；按 `.env.example` 配置相应字段。

兼容原来的 `DEEPSEEK_API` / `QWEN_API` 名称；shell 环境变量优先。
下面的生成和修复命令会实际调用模型 API。

## 3. CodeSim：完整生成流程

执行的是 `baselines/codesim/upstream/src/promptings/CodeSIM.py` 中的策略，
包括规划、模拟、生成和调试。默认规划尝试和调试尝试上限均为 5，
可用 `--max-plan-try`、`--max-debug-try` 调整。

```bash
# 先跑一道题，检查模型连通性和完整流程。
python baselines/codesim/run_apps_200_deepseek.py --limit 1 \
  --results results/codesim_smoke.jsonl

# 完整的 200 题，每题最终检查前 10 个数据集测试对。
python baselines/codesim/run_apps_200_deepseek.py --max-cases 10 \
  --results results/codesim_apps_200_deepseek.jsonl

# 使用 Qwen。
python baselines/codesim/run_apps_200_qwen.py --max-cases 10 \
  --results results/codesim_apps_200_qwen.jsonl
```

`--max-cases 0` 表示检查全部测试对。改变这个参数时应换一个结果文件名。

## 4. MapCoder：完整生成流程

执行的是 `baselines/mapcoder/upstream/src/promptings/MapCoder.py` 中的策略，
保留检索、规划、生成和改进流程，参数为 `k=3`、`t=5`。
适配层修正了代码块解析和 XML CDATA 包装，并使用本地 APPS 子进程判题。

```bash
python baselines/mapcoder/run_apps_200_deepseek.py --limit 1 \
  --results results/mapcoder_smoke.jsonl

python baselines/mapcoder/run_apps_200_deepseek.py --max-cases 10 \
  --results results/mapcoder_apps_200_deepseek_corrected.jsonl

python baselines/mapcoder/run_apps_200_qwen.py --max-cases 10 \
  --results results/mapcoder_apps_200_qwen.jsonl
```

CodeSim 和 MapCoder 都通过重新执行同一条命令续跑：已保存的题号会跳过，
不需要 `--resume`。要重跑已保存的失败题，或更换模型、预算、测试对数量，
使用新的结果文件。需要复用原设置的文件会拒绝不兼容的参数。

两个基线共享固定的 50 道 introductory、100 道 interview、50 道 competition 题。
前两个 APPS 测试对作为生成/调试示例；最终检查前十对或全部对，包含这两个示例。
这属于当前本地样本的评测协议，不能当作另有隐藏测试集的官方 APPS 榜单评测。
模型提示中不提供参考答案。

结果保存在 JSONL 中：`status` 表示执行状态，`is_solved` 表示最终代码是否通过
全部选中的测试对，`settings` 记录模型和预算，`served_models` 记录服务端返回的
模型标识。token 和调用次数也会记录。逐题日志位于同级的
`<结果文件名去掉后缀>_traces/`。全量实验应按 200 题及 50/100/50 分层统计；
不要把仅跑一题的 smoke test 当作性能结果。API 请求失败会中止当前生成任务，
已保存的 `method_error`、`output_truncated` 则按未解决计入结果。

## 5. 受控故障修复测试

入口为 `tests/apps_100_repair_benchmark.py`。这项实验从已有的相同故障程序开始，
比较修复行为。CodeSim/MapCoder 在这里使用专门的修复提示和预算，
不会调用上面完整生成实验的上游策略类，因此两类结果要分开报告。

```bash
# 不调用 API，预期为 100 个 case、2 个方法、200 次计划运行。
python tests/apps_100_repair_benchmark.py --methods codesim,mapcoder --dry-run

# 一道题、两个方法的在线 smoke test。
python tests/apps_100_repair_benchmark.py --provider deepseek \
  --methods codesim,mapcoder --limit 1 --workers 1 \
  --output-dir results/repair_smoke

# 100 题、AuditCoder/CodeSim/MapCoder/AgentCoder 四种修复适配器。
python tests/apps_100_repair_benchmark.py --provider deepseek --workers 4 \
  --output-dir results/apps_100_repair_benchmark

# 续跑，同时重试 API/error 状态的记录。
python tests/apps_100_repair_benchmark.py --provider deepseek --workers 4 \
  --output-dir results/apps_100_repair_benchmark --resume --retry-errors
```

Qwen 将 `--provider deepseek` 改为 `--provider qwen`。
续跑时保持模型、方法、case 选择和预算一致；更换协议使用新的输出目录。
这项修复实验必须加 `--resume` 才会跳过已有记录。

每类故障各 25 题：运行异常、错误答案、接口异常、超时。
AuditCoder 的修复预算为 1 次，CodeSim/MapCoder 各最多 5 次，
AgentCoder 在独立测试设计后修复 1 次。每次候选修复都用存储的测试对重新判定，
模型会接收失败反馈。输出为 `results.jsonl` 和 `summary.json`；主要查看
`repair_success`、`final_all_passed`、`attempt_count` 和异常状态。

语料保留了参考程序来源：78 个 APPS 参考答案、18 个既有 CodeSim 输出、
1 个既有 MapCoder 输出，以及 APPS 0583、2685、2911 三个测试集查表兜底。
这三题仅适合存储测试范围内的故障诊断，使用完整语料报告结果时必须说明。
语料构建本身不调用模型，修复模型也不会收到 `clean.py`。

## 6. AuditCoder 与 ClassEval

```bash
python tests/run_apps_200_auditcoder.py --provider deepseek --count 1 \
  --output results/auditcoder_smoke.jsonl
python tests/run_apps_200_auditcoder.py --provider deepseek --max-cases 10 \
  --output results/apps_200_auditcoder.jsonl --resume
python tests/run_apps_200_once_retry.py --provider deepseek --max-cases 10 \
  --output results/apps_200_auditcoder_one_retry.jsonl --resume

python -m pip install -r requirements-classeval.txt
python tests/run_classeval_auditcoder.py --provider deepseek --ids ClassEval_0 \
  --output results/classeval_smoke.jsonl
python tests/run_classeval_auditcoder.py --provider deepseek --count 100 \
  --output results/classeval_auditcoder.jsonl --resume
```

专用 one-retry 入口限定每次流程最多一次有界修复，并允许一次整题重跑。
ClassEval 每个类最多一次方法修复。

当前 AuditCoder APPS 流程会把选中的数据集测试对交给最终验证器，且保留了
`canonical_solutions.py` 中按题目特征匹配的模板解法；ClassEval 流程使用可信
unittest 规格。与基线比较时必须同时说明这些协议和 oracle 访问差异。
整理版本保留原实现行为，未宣称重新生成论文表格。

重新构建语料所需的完整 APPS 原始目录、可选脚本及更详细的字段说明见
[英文测试文档](TESTING.md)。匿名上传说明见 [ANONYMITY.md](ANONYMITY.md)。

## 7. AgentCoder 专项测试

发布包包含两种 AgentCoder 风格适配器：APPS 100 题受控故障修复，以及
ClassEval 类生成与有界修复。两者使用本地任务提示和判题器，测试设计器输出
文字测试计划；与原生 HumanEval/MBPP 中用独立生成的可执行测试选择候选的
流程有所不同。以下命令均在仓库根目录执行。

### 7.1 APPS：相同故障程序上的修复

安装 `requirements.txt`，在根目录 `.env` 中配置 DeepSeek 或 Qwen。
使用的是共享 100 题语料，无需额外下载完整 APPS。

```bash
# 离线检查，无需密钥；预期 100 个 case、1 个方法、100 次计划运行。
python tests/apps_100_repair_benchmark.py --methods agentcoder --dry-run

# 先跑一道题，实际调用模型。
python tests/apps_100_repair_benchmark.py --provider deepseek \
  --methods agentcoder --limit 1 --workers 1 \
  --output-dir results/agentcoder_repair_smoke

# 完整 100 题。
python tests/apps_100_repair_benchmark.py --provider deepseek \
  --methods agentcoder --workers 4 \
  --output-dir results/agentcoder_repair_deepseek

# 续跑，并重试 API/error 状态记录。
python tests/apps_100_repair_benchmark.py --provider deepseek \
  --methods agentcoder --workers 4 \
  --output-dir results/agentcoder_repair_deepseek --resume --retry-errors

# Qwen 单独保存结果。
python tests/apps_100_repair_benchmark.py --provider qwen \
  --methods agentcoder --workers 4 \
  --output-dir results/agentcoder_repair_qwen
```

每题先由独立测试设计器阅读题目和公开示例，再由 programmer 接收测试计划、
故障程序和执行反馈，最多修复 1 次；API 网络重试不等同于修复次数。
测试设计请求失败时会记录异常，programmer 仍可根据执行反馈继续。
最终候选使用该题存储的全部测试对判定，不向模型提供 `clean.py`。

输出目录中的 `results.jsonl` 记录逐题结果，`summary.json` 记录统计。
主要查看：

- `test_designer.used`、`test_designer.api.error`：测试设计器是否启用及是否失败。
- `attempt_count`：programmer 修复次数，最多 1 次。
- `repair_success`、`final_all_passed`、`status`：修复结果与执行状态。
- `summary.json` 中 `by_method.agentcoder.repair_success_rate` 和
  `by_method.agentcoder.fault_types`：总体及四类故障结果。

全量完成后核对 100 条结果及 `api_errors`。AgentCoder 的
`localization_correct` 是适配器占位字段，不代表真实定位准确率。
参考程序中三题查表兜底的来源说明仍适用，见第 5 节。

### 7.2 ClassEval：类生成、候选选择与一次修复

入口 `tests/run_classeval_agentcoder.py` 已收录，并复用本仓库的 ClassEval
执行器、数据与 API 客户端。

```bash
python -m pip install -r requirements-classeval.txt

# 检查全部 100 个类和提示来源文件，不调用 API。
python tests/run_classeval_agentcoder.py --dry-run

# 单个类的在线 smoke test。
python tests/run_classeval_agentcoder.py --provider deepseek --ids ClassEval_0 \
  --output results/agentcoder_classeval_smoke.jsonl

# 100 个类，默认每类 5 个候选，最多一次修复。
python tests/run_classeval_agentcoder.py --provider deepseek --count 100 \
  --candidates 5 --max-retries 1 \
  --output results/agentcoder_classeval_deepseek.jsonl

# 续跑同一组实验。
python tests/run_classeval_agentcoder.py --provider deepseek --count 100 \
  --candidates 5 --max-retries 1 \
  --output results/agentcoder_classeval_deepseek.jsonl --resume

python tests/run_classeval_agentcoder.py --provider qwen --count 100 \
  --candidates 5 --max-retries 1 \
  --output results/agentcoder_classeval_qwen.jsonl
```

流程为一次独立测试计划调用、5 次 programmer 候选生成、执行器选出得分最高的
候选；未全通过时再允许一次修复，仅在得分严格提高时保留。测试设计器不看
unittest，programmer 提示和候选选择会使用可信 ClassEval 测试。
`--candidates` 可调整候选数，`--max-retries 0` 可关闭修复，修复上限只能为 0 或 1。
正常模型调用预算为 1 + 5 + 最多 1 次，传输重试另计。

逐题 JSONL 中查看 `candidate_scores`、`selected_candidate`、`retry_count`、
`final_tests.all_passed`、token 用量和 `status`；同名 `.summary.json` 中查看
`overall.problem_accuracy` 与 `overall.method_accuracy`。
日志位于 `<输出文件名去掉后缀>_traces/`。报告结果时同时写明候选数、修复预算
和可信测试的使用方式，不把它表述成隐藏测试集上的 pass@1。

此入口不加 `--resume` 会重建输出文件；续跑会跳过所有已保存题号，包括错误记录。
要重跑错误题或改变模型、候选数等设置，使用新结果文件。
来源说明见 [AgentCoder 目录说明](../baselines/agentcoder/README.md)。

