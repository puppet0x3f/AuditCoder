# AuditCoder Architecture

## 1. 项目目标

AuditCoder 是一个面向算法题的多 Agent 代码生成与验证流水线。  
核心目标是把原始题目文本逐步转化为可执行 Python 代码，并通过分层验证机制提高正确率。

端到端主流程：

1. `plan_agent` 生成任务树 IR（`TaskPlanIR`）
2. `algo_agents` 按叶子任务生成代码片段并做片段验证
3. `assembler_agent` 组装完整代码
4. `final_verifier_agent` 做整体验证、定位错误并尝试一次重试

入口参考：`tests/run_mbpp.py:657` (`run_pipeline`)

---

## 2. 目录结构与职责

```text
auditcoder/
├─ common/                  # 共享工具（LLM、JSON 提取）
├─ schemas/                 # Pydantic 数据模型
├─ plan_agent/              # 题目解析与任务规划（生成 TaskPlanIR）
├─ algo_agents/             # 12 类算法 Agent + 调度器
├─ verifier_agent/          # 单片段验证图（强/中/弱信号）
├─ assembler_agent/         # 代码片段组装为完整解
└─ final_verifier_agent/    # 全代码验证与责任定位 + 一次重试
```

---

## 3. 端到端时序

```text
problem_raw
  -> plan_agent (task_plan_ir)
  -> algo dispatcher (code_snippets + verification_reports)
  -> assembler_agent (final_code)
  -> final_verifier_agent
       -> full_code_runner
       -> pass: final_report
       -> fail: error_locator -> task_replanner -> algo_retrier -> code_reassembler -> final_report
```

---

## 4. 核心模块说明

## 4.1 `plan_agent`

图定义：`auditcoder/plan_agent/graph.py:23`

固定链路：

1. `problem_parser`：原题文本 -> 结构化题面
2. `constraint_analyzer`：约束 -> 复杂度预算
3. `algorithm_selector`：算法选择与推理链
4. `task_decomposer`：分解为层级任务树（叶子可分配 agent）
5. `interface_designer`：为叶子节点生成严格函数接口
6. `test_case_analyzer`：样例和边界测试推导
7. `ir_assembler`：组装最终 `TaskPlanIR`

状态：`auditcoder/plan_agent/state.py:16` (`PlanState`)

备注：`ir_reviewer` 已实现，但当前未接入主 graph。

## 4.2 `algo_agents`

调度器：`auditcoder/algo_agents/dispatcher.py:56`

职责：

- 遍历 `task_plan_ir.execution_order`
- 仅处理 `is_leaf=true` 节点
- 按 `agent_type` 选择具体算法 agent
- 每个任务做一次片段验证；若 `reject`，注入反馈重试一次

当前支持 12 类算法 Agent：

- `dp`
- `greedy`
- `binary_search`
- `string`
- `segment_tree`
- `monotonic_queue`
- `graph_bfs`
- `graph_dfs`
- `union_find`
- `math`
- `sorting`
- `two_pointer`

每个算法 Agent 结构一致：单节点 LangGraph，输入 `task_node + task_plan_ir (+feedback)`，输出 `code_snippet`。

## 4.3 `verifier_agent`

图定义：`auditcoder/verifier_agent/graph.py:32`

节点顺序（带编译分支路由）：

1. `contract_verifier`
2. `compile_verifier`
3. 编译通过 -> `sample_test_runner -> bruteforce_verifier -> complexity_verifier`
4. 否则直接进入 `counterexample_gen`
5. `weak_signal_eval`
6. `verdict_assembler`

验证信号：

- 强信号：语法、编译、样例、对拍、资源
- 中信号：契约一致性、复杂度预算
- 弱信号：可读性、风格、自洽性

## 4.4 `assembler_agent`

入口：`auditcoder/assembler_agent/agent.py:170`

职责：

- 读取 `task_plan_ir + code_snippets`
- 组织任务片段文本、依赖数据流文本
- 让 LLM 生成完整 `final_code`

## 4.5 `final_verifier_agent`

图定义：`auditcoder/final_verifier_agent/graph.py:41`

职责：

- 运行完整代码并汇总测试结果
- 如果失败，定位责任任务（traceback/timing/report）
- 触发该任务的重规划与重生成
- 重组代码并输出最终报告

注意：当前图在重试后直接进入 `final_report_assembler`，不会再次回到 `full_code_runner` 复测。

---

## 5. 关键数据契约

## 5.1 Schema

- `TaskPlanIR`：`auditcoder/schemas/task_plan_ir.py:55`
- `TaskNode`：`auditcoder/schemas/task_plan_ir.py:33`
- `DataSlot`：`auditcoder/schemas/task_plan_ir.py:21`
- `VerificationReport`：`auditcoder/schemas/verification.py`

## 5.2 各阶段状态

- `PlanState`：`auditcoder/plan_agent/state.py:16`
- `AlgoAgentState`：`auditcoder/algo_agents/state.py:10`
- `AssemblerState`：`auditcoder/assembler_agent/state.py:10`
- `VerifierState`：`auditcoder/verifier_agent/state.py:10`
- `FinalVerifierState`：`auditcoder/final_verifier_agent/state.py:10`

## 5.3 共享能力

- LLM 工厂：`auditcoder/common/llm.py:14` (`get_llm`)
- JSON 提取：`auditcoder/common/utils.py:12` (`extract_json`)
- 生成代码执行：`auditcoder/verifier_agent/sandbox/executor.py`
  - `host`：兼容本地测试的主机子进程，不提供安全隔离
  - `container`：Docker/Podman 后端，限制网络、根文件系统、权限、CPU、内存和 PID

---

## 6. Prompt 层角色

每个 Agent 的行为由 `prompts/` 下模板约束，代码层主要负责：

1. 构建上下文
2. 调用 LLM
3. 解析 JSON
4. 合并到状态

因此该项目是“图编排 + 结构化提示词 + 程序化验证”混合架构，而非纯规则系统。

---

## 7. 已识别风险与修复状态

## 已修复的 P0

1. `task_contract` 结构已经统一

调度器现在构造包含 `task_id`、`agent_type` 和 `interface` 的 verifier 输入，相关节点统一从 `task_contract["interface"]` 读取。

2. 复杂度预算字段已经闭合

字符串预算规范化为 `complexity_target={"time": ...}`；复杂度验证器同时兼容历史 `budget` 字段，并有 dispatcher 到 verifier 的回归测试。

3. 不安全执行边界已经显式化

主机后端保留但明确标记为非隔离；新增可选 Docker/Podman 后端和非 root 镜像配方。容器后端不会静默回退到主机执行。

## P1（建议尽快处理）

1. Final retry 后缺少复测闭环  
问题：重规划 + 重生成后未重新执行全量测试。  
影响：`final_report` 可能基于旧失败结果。  
涉及：`auditcoder/final_verifier_agent/graph.py`

2. `ir_reviewer` 未接入主流程  
问题：IR 一致性审查节点存在但未被 graph 使用。  
影响：TaskPlanIR 质量保障缺失一层防线。  
涉及：`auditcoder/plan_agent/graph.py`、`auditcoder/plan_agent/nodes/ir_reviewer.py`

## P2（可迭代优化）

1. 报告字段标准化  
目标：统一各验证节点输出结构，减少 fallback 与隐式兼容逻辑。

2. 可观测性增强  
目标：增加每阶段耗时、token 消耗、失败原因统计，便于离线分析与调参。

---

## 8. 运行与调试入口

- 单题/批量 MBPP 跑通：`tests/run_mbpp.py`
- 输出结果目录：`data/results`（默认）
- `conftest.py` 中定义了 `integration` 标记（需 LLM API）

---

## 9. 架构结论

当前 AuditCoder 的主干结构清晰，具备“规划-生成-验证-回退”的基本闭环。  
当前 P0 字段契约与执行边界问题已有代码和回归测试；下一步重点是处理 `P1` 的复测闭环与 IR 审查接入。
