"""Pydantic models for the Plan Agent's AnalysisDocument — the single contract
between Plan Agent and all downstream agents (FrameworkAgent, AlgorithmAgents)."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Part 1: Problem Information
# ──────────────────────────────────────────────

class DataRange(BaseModel):
    var: str = Field(description="变量名，如 n, m, k")
    min: int = Field(description="最小值")
    max: int = Field(description="最大值")


class Constraints(BaseModel):
    time_limit: str = Field(default="1s", description="时间限制")
    memory_limit: str = Field(default="256MB", description="内存限制")
    data_ranges: list[DataRange] = Field(default_factory=list, description="各变量的数据范围")


class SampleCase(BaseModel):
    input: str = Field(description="样例输入")
    output: str = Field(description="样例输出")
    explanation: str = Field(default="", description="样例解释")


class ProblemInfo(BaseModel):
    title: str = Field(description="题目标题")
    description: str = Field(description="清洗后的题目描述")
    input_format: str = Field(description="输入格式的精确描述")
    output_format: str = Field(description="输出格式的精确描述")
    constraints: Constraints = Field(default_factory=Constraints)
    sample_cases: list[SampleCase] = Field(default_factory=list)


# ──────────────────────────────────────────────
# Part 2: Analysis & Reasoning
# ──────────────────────────────────────────────

class ComplexityTarget(BaseModel):
    time: str = Field(description="时间复杂度目标，如 O(n log n)")
    space: str = Field(description="空间复杂度目标，如 O(n)")


class AlgorithmChoice(BaseModel):
    algorithm_type: str = Field(description="算法类型，如 dp, greedy, binary_search")
    agent_id: str = Field(description="对应的算法 Agent 标识")
    subtask: str = Field(description="此算法负责解决的子问题描述")
    reason: str = Field(description="选择该算法的推理依据")


class AnalysisInfo(BaseModel):
    problem_category: str = Field(description="问题分类，如 图论/数论/字符串/数据结构")
    key_observations: list[str] = Field(default_factory=list, description="关键性质与观察")
    complexity_target: ComplexityTarget = Field(description="由约束推导的复杂度目标")
    algorithm_selection: list[AlgorithmChoice] = Field(default_factory=list)
    reasoning_chain: list[str] = Field(default_factory=list, description="完整推理链（可审计）")


# ──────────────────────────────────────────────
# Part 3: Architecture Design
# ──────────────────────────────────────────────

class ParamSpec(BaseModel):
    name: str
    type: str
    description: str = ""


class FunctionInterface(BaseModel):
    function_name: str = Field(description="函数名称")
    params: list[ParamSpec] = Field(default_factory=list, description="参数列表")
    return_type: str = Field(description="返回值类型")
    return_description: str = Field(default="", description="返回值描述")
    preconditions: list[str] = Field(default_factory=list, description="前置条件")
    postconditions: list[str] = Field(default_factory=list, description="后置条件")


class AlgorithmTask(BaseModel):
    task_id: str = Field(description="任务唯一标识")
    agent_type: str = Field(description="算法 Agent 类型，如 dp, greedy")
    description: str = Field(description="任务描述")
    interface: FunctionInterface = Field(description="严格的函数接口定义")
    hints: list[str] = Field(default_factory=list, description="给算法 Agent 的提示")
    complexity_expectation: str = Field(default="", description="该部分的复杂度期望")


class DependencyEdge(BaseModel):
    source: str = Field(alias="from", description="前置任务 task_id")
    target: str = Field(alias="to", description="后续任务 task_id")

    model_config = {"populate_by_name": True}


class TaskDependencyGraph(BaseModel):
    edges: list[DependencyEdge] = Field(default_factory=list)


class DataStructureSpec(BaseModel):
    name: str = Field(description="数据结构变量名")
    type: str = Field(description="类型，如 list[int], dict[str, list[int]]")
    purpose: str = Field(description="用途说明")


class ArchitectureInfo(BaseModel):
    solution_overview: str = Field(description="解法概述（自然语言）")
    execution_flow: list[str] = Field(default_factory=list, description="执行流程步骤")
    data_structures: list[DataStructureSpec] = Field(default_factory=list)
    algorithm_tasks: list[AlgorithmTask] = Field(default_factory=list)
    task_dependency_graph: TaskDependencyGraph = Field(default_factory=TaskDependencyGraph)


# ──────────────────────────────────────────────
# Part 4: Validation
# ──────────────────────────────────────────────

class TestCase(BaseModel):
    input: str
    expected_output: str


class EdgeCase(BaseModel):
    description: str
    input: str
    expected_output: str


class ValidationInfo(BaseModel):
    sample_test_cases: list[TestCase] = Field(default_factory=list)
    edge_cases: list[EdgeCase] = Field(default_factory=list)
    invariants: list[str] = Field(default_factory=list, description="解必须满足的不变量")


# ──────────────────────────────────────────────
# Top-level AnalysisDocument
# ──────────────────────────────────────────────

class AnalysisDocument(BaseModel):
    """Plan Agent 输出给下游所有子 Agent 的唯一契约。"""

    problem: ProblemInfo
    analysis: AnalysisInfo
    architecture: ArchitectureInfo
    validation: ValidationInfo
