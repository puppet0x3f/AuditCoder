"""Pydantic models for the Plan Agent's TaskPlanIR — the hierarchical task tree
IR that replaces the flat AnalysisDocument architecture section.

The IR uses a flat list of TaskNode with parent/dependencies fields to express
a tree structure, enabling both hierarchical decomposition and DAG-based
execution scheduling."""

from __future__ import annotations

from pydantic import BaseModel, Field

from auditcoder.schemas.analysis_document import (
    AnalysisInfo,
    ComplexityTarget,
    FunctionInterface,
    ProblemInfo,
    ValidationInfo,
)


class DataSlot(BaseModel):
    """Typed data port on a TaskNode, supporting cross-task references."""

    name: str = Field(description="变量名，如 'a', 'query_max'")
    type: str = Field(description="Python 类型，如 'list[int]', 'Callable[[int,int], int]'")
    description: str = Field(default="", description="说明")
    source: str | None = Field(
        default=None,
        description="数据来源：null 表示外部输入，'S1_1.query' 表示引用任务 S1_1 的 query 输出",
    )


class TaskNode(BaseModel):
    """A single node in the hierarchical task tree.

    Leaf nodes (is_leaf=True) are assigned to algorithm agents and must have
    agent_type and interface defined.  Internal nodes represent composite tasks
    whose children must complete before the parent can proceed."""

    id: str = Field(description="层级 ID：'S0', 'S1', 'S1_1'")
    goal: str = Field(description="任务目标")
    type: str = Field(description="算法类型标识，如 'dp', 'greedy', 'orchestration'")
    is_leaf: bool = Field(description="是否为叶节点（可直接分配 Agent）")
    agent_type: str | None = Field(default=None, description="Agent 类型（仅叶节点非空）")
    parent: str | None = Field(default=None, description="父节点 ID")
    dependencies: list[str] = Field(default_factory=list, description="依赖的子任务 ID（执行前须完成）")
    inputs: list[DataSlot] = Field(default_factory=list, description="输入数据槽")
    outputs: list[DataSlot] = Field(default_factory=list, description="输出数据槽")
    interface: FunctionInterface | None = Field(default=None, description="函数接口（叶节点必须有）")
    hints: list[str] = Field(default_factory=list, description="给算法 Agent 的提示")
    complexity_budget: str = Field(default="", description="分配的复杂度预算")
    context: str = Field(default="", description="上下文说明")


class TaskPlanIR(BaseModel):
    """Plan Agent 输出给下游调度器和算法 Agent 的层级任务树 IR。"""

    problem: ProblemInfo
    analysis: AnalysisInfo
    global_strategy: str = Field(description="整体策略描述")
    complexity_budget: ComplexityTarget = Field(description="总体复杂度预算")
    task_nodes: list[TaskNode] = Field(description="扁平化存储的任务节点，通过 parent/dependencies 表达树")
    root_id: str = Field(description="根任务 ID")
    execution_order: list[str] = Field(description="拓扑排序后的执行顺序（叶节点）")
    validation: ValidationInfo
