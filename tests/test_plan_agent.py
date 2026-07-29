"""Smoke tests for the Plan Agent graph construction and schema validation.

When run directly (python tests/test_plan_agent.py), prints detailed IR
structures and analysis results.  Under pytest, output is captured unless
run with `pytest -s`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auditcoder.plan_agent.graph import build_plan_agent_graph
from auditcoder.plan_agent.state import PlanState
from auditcoder.schemas.analysis_document import (
    AnalysisInfo,
    ComplexityTarget,
    FunctionInterface,
    ProblemInfo,
    ValidationInfo,
)
from auditcoder.schemas.task_plan_ir import DataSlot, TaskNode, TaskPlanIR


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _print_ir(ir: TaskPlanIR):
    """Print a TaskPlanIR in human-readable format."""
    print(f"\n  Problem:          {ir.problem.title}")
    print(f"  Category:         {ir.analysis.problem_category}")
    print(f"  Global Strategy:  {ir.global_strategy}")
    print(f"  Complexity:       time={ir.complexity_budget.time}, space={ir.complexity_budget.space}")
    print(f"  Root ID:          {ir.root_id}")
    print(f"  Execution Order:  {ir.execution_order}")
    print(f"  Total Nodes:      {len(ir.task_nodes)}")

    print("\n  --- Task Tree ---")
    node_map = {n.id: n for n in ir.task_nodes}
    children_map: dict[str | None, list[str]] = {}
    for n in ir.task_nodes:
        children_map.setdefault(n.parent, []).append(n.id)

    def _print_node(nid: str, depth: int = 0):
        node = node_map[nid]
        indent = "    " + "  " * depth
        leaf_tag = "[LEAF]" if node.is_leaf else "[INTERNAL]"
        agent_tag = f" -> agent:{node.agent_type}" if node.agent_type else ""
        print(f"{indent}{node.id} {leaf_tag}{agent_tag}")
        print(f"{indent}  goal: {node.goal}")
        print(f"{indent}  type: {node.type}")
        if node.dependencies:
            print(f"{indent}  deps: {node.dependencies}")
        if node.complexity_budget:
            print(f"{indent}  complexity: {node.complexity_budget}")
        if node.inputs:
            for inp in node.inputs:
                src = f" <- {inp.source}" if inp.source else ""
                print(f"{indent}  IN:  {inp.name}: {inp.type}{src}")
        if node.outputs:
            for out in node.outputs:
                print(f"{indent}  OUT: {out.name}: {out.type}")
        if node.interface:
            iface = node.interface
            params_str = ", ".join(f"{p.name}: {p.type}" for p in iface.params)
            print(f"{indent}  interface: {iface.function_name}({params_str}) -> {iface.return_type}")
            if iface.preconditions:
                print(f"{indent}    pre:  {iface.preconditions}")
            if iface.postconditions:
                print(f"{indent}    post: {iface.postconditions}")
        if node.hints:
            print(f"{indent}  hints: {node.hints}")
        for child_id in children_map.get(nid, []):
            _print_node(child_id, depth + 1)

    _print_node(ir.root_id)

    print("\n  --- Data Flow Graph ---")
    for node in ir.task_nodes:
        for inp in node.inputs:
            if inp.source:
                print(f"    {inp.source}  -->  {node.id}.{inp.name}  ({inp.type})")

    print("\n  --- Validation ---")
    print(f"    Sample tests:  {len(ir.validation.sample_test_cases)}")
    print(f"    Edge cases:    {len(ir.validation.edge_cases)}")
    print(f"    Invariants:    {ir.validation.invariants}")

    print("\n  --- Full IR (JSON) ---")
    print(json.dumps(ir.model_dump(), ensure_ascii=False, indent=2))


def _print_hierarchy(nodes: list[TaskNode]):
    """Print a node list as a hierarchy summary."""
    node_map = {n.id: n for n in nodes}
    children_map: dict[str | None, list[str]] = {}
    for n in nodes:
        children_map.setdefault(n.parent, []).append(n.id)

    def _walk(nid: str, depth: int = 0):
        node = node_map[nid]
        indent = "    " + "  " * depth
        marker = "L" if node.is_leaf else "+"
        agent = f" [{node.agent_type}]" if node.agent_type else ""
        deps = f" deps={node.dependencies}" if node.dependencies else ""
        print(f"{indent}[{marker}] {nid}: {node.goal}{agent}{deps}")
        for cid in children_map.get(nid, []):
            _walk(cid, depth + 1)

    roots = children_map.get(None, [])
    for r in roots:
        _walk(r)


# ---------------------------------------------------------------------------
# Test 1: Graph compilation
# ---------------------------------------------------------------------------

def test_graph_compiles():
    """The Plan Agent graph should compile without error."""
    _separator("Test: Graph Compilation")
    app = build_plan_agent_graph()
    assert app is not None

    print("\n  Graph nodes:")
    for node_name in app.get_graph().nodes:
        print(f"    - {node_name}")

    print("\n  Graph edges:")
    for edge in app.get_graph().edges:
        print(f"    {edge}")

    print("\n  PASSED")


# ---------------------------------------------------------------------------
# Test 2: TaskPlanIR schema (Two Sum example)
# ---------------------------------------------------------------------------

def test_task_plan_ir_schema():
    """TaskPlanIR should accept a minimal valid payload."""
    _separator("Test: TaskPlanIR Schema (Two Sum)")

    ir = TaskPlanIR(
        problem=ProblemInfo(
            title="Two Sum",
            description="Given an array, find two numbers that add up to target.",
            input_format="First line: n and target. Second line: n integers.",
            output_format="Two 0-indexed positions.",
            constraints={
                "time_limit": "1s",
                "memory_limit": "256MB",
                "data_ranges": [{"var": "n", "min": 1, "max": 100000}],
            },
            sample_cases=[
                {"input": "4 9\n2 7 11 15", "output": "0 1", "explanation": "2+7=9"}
            ],
        ),
        analysis=AnalysisInfo(
            problem_category="数据结构",
            key_observations=["可以用哈希表在 O(1) 内查找 complement"],
            complexity_target=ComplexityTarget(time="O(n)", space="O(n)"),
            algorithm_selection=[
                {
                    "algorithm_type": "hash_map",
                    "agent_id": "agent_hash_map",
                    "subtask": "利用哈希表查找 target - nums[i]",
                    "reason": "单次遍历 + O(1) 查找满足 O(n) 复杂度要求",
                }
            ],
            reasoning_chain=[
                "暴力解 O(n^2) 不满足 n<=10^5 的约束",
                "需要 O(n) 解法",
                "哈希表可在 O(1) 内查找 complement",
            ],
        ),
        global_strategy="单次遍历数组，用哈希表查找 complement",
        complexity_budget=ComplexityTarget(time="O(n)", space="O(n)"),
        task_nodes=[
            TaskNode(
                id="S0",
                goal="解决 Two Sum 问题",
                type="orchestration",
                is_leaf=False,
                parent=None,
                dependencies=["S1"],
                context="根节点",
            ),
            TaskNode(
                id="S1",
                goal="哈希表查找 complement 并返回索引对",
                type="hash_map",
                is_leaf=True,
                agent_type="hash_map",
                parent="S0",
                dependencies=[],
                inputs=[
                    DataSlot(name="nums", type="list[int]", description="输入数组"),
                    DataSlot(name="target", type="int", description="目标和"),
                ],
                outputs=[
                    DataSlot(name="result", type="tuple[int, int]", description="索引对"),
                ],
                interface=FunctionInterface(
                    function_name="two_sum_hash",
                    params=[
                        {"name": "nums", "type": "list[int]", "description": "输入数组"},
                        {"name": "target", "type": "int", "description": "目标和"},
                    ],
                    return_type="tuple[int, int]",
                    return_description="满足条件的两个索引",
                    preconditions=["len(nums) >= 2", "恰好有一组解"],
                    postconditions=["nums[i] + nums[j] == target", "i < j"],
                ),
                hints=["单次遍历即可"],
                complexity_budget="O(n)",
                context="唯一的叶节点，直接完成查找",
            ),
        ],
        root_id="S0",
        execution_order=["S1"],
        validation=ValidationInfo(
            sample_test_cases=[
                {"input": "4 9\n2 7 11 15", "expected_output": "0 1"}
            ],
            edge_cases=[
                {
                    "description": "最小输入",
                    "input": "2 3\n1 2",
                    "expected_output": "0 1",
                }
            ],
            invariants=["nums[i] + nums[j] == target"],
        ),
    )

    assert ir.problem.title == "Two Sum"
    assert ir.root_id == "S0"
    assert len(ir.task_nodes) == 2
    assert ir.task_nodes[1].is_leaf is True
    assert ir.task_nodes[1].agent_type == "hash_map"
    assert ir.task_nodes[1].interface is not None

    _print_ir(ir)
    print("\n  PASSED")


# ---------------------------------------------------------------------------
# Test 3: Multi-level task hierarchy (DP + RMQ + Greedy)
# ---------------------------------------------------------------------------

def test_task_node_hierarchy():
    """Verify parent/dependencies/is_leaf relationships in a multi-level tree."""
    _separator("Test: Task Node Hierarchy (DP + RMQ + Greedy)")

    nodes = [
        TaskNode(id="S0", goal="解决完整问题", type="orchestration", is_leaf=False,
                 parent=None, dependencies=["S1"],
                 context="根节点"),
        TaskNode(id="S1", goal="DP 求解 dp[i]=min(dp[j]+max(a[j+1..i]))",
                 type="dp", is_leaf=True, agent_type="dp",
                 parent="S0", dependencies=["S1_1", "S1_2"],
                 inputs=[
                     DataSlot(name="n", type="int", description="数组长度"),
                     DataSlot(name="a", type="list[int]", description="输入数组"),
                     DataSlot(name="query_max", type="Callable[[int,int],int]",
                              description="区间最值查询", source="S1_1.query"),
                     DataSlot(name="valid_left", type="list[int]",
                              description="有效转移左边界", source="S1_2.boundaries"),
                 ],
                 outputs=[DataSlot(name="answer", type="int", description="最终答案")],
                 interface=FunctionInterface(
                     function_name="solve_dp",
                     params=[
                         {"name": "n", "type": "int", "description": "数组长度"},
                         {"name": "a", "type": "list[int]", "description": "输入数组"},
                         {"name": "query_max", "type": "Callable[[int,int],int]", "description": "RMQ"},
                         {"name": "valid_left", "type": "list[int]", "description": "左边界"},
                     ],
                     return_type="int",
                     return_description="dp[n]",
                     preconditions=["query_max 已初始化", "len(valid_left)==n+1"],
                     postconditions=["返回值 >= 0"],
                 ),
                 hints=["dp[0]=0", "转移: dp[i]=min(dp[j]+query_max(j+1,i))"],
                 complexity_budget="O(n log n)"),
        TaskNode(id="S1_1", goal="构建区间最值查询 query(l,r)->max(a[l..r])",
                 type="segment_tree", is_leaf=True, agent_type="segment_tree",
                 parent="S1", dependencies=[],
                 inputs=[DataSlot(name="a", type="list[int]", description="输入数组")],
                 outputs=[DataSlot(name="query", type="Callable[[int,int],int]",
                                   description="查询函数")],
                 interface=FunctionInterface(
                     function_name="build_range_max",
                     params=[{"name": "a", "type": "list[int]", "description": "输入数组"}],
                     return_type="Callable[[int,int],int]",
                     return_description="query(l,r) 返回 max(a[l..r])",
                     preconditions=["len(a)>=1"],
                     postconditions=["query(l,r)==max(a[l:r+1])"],
                 ),
                 hints=["线段树 build O(n), query O(log n)"],
                 complexity_budget="O(n log n) build, O(log n) query"),
        TaskNode(id="S1_2", goal="贪心预处理有效转移左边界",
                 type="greedy", is_leaf=True, agent_type="greedy",
                 parent="S1", dependencies=[],
                 inputs=[
                     DataSlot(name="n", type="int", description="数组长度"),
                     DataSlot(name="a", type="list[int]", description="输入数组"),
                     DataSlot(name="k", type="int", description="种类上限"),
                 ],
                 outputs=[DataSlot(name="boundaries", type="list[int]",
                                   description="valid_left 数组")],
                 interface=FunctionInterface(
                     function_name="compute_valid_boundaries",
                     params=[
                         {"name": "n", "type": "int", "description": "数组长度"},
                         {"name": "a", "type": "list[int]", "description": "输入数组"},
                         {"name": "k", "type": "int", "description": "种类上限"},
                     ],
                     return_type="list[int]",
                     return_description="valid_left 数组",
                     preconditions=["n>=1", "k>=1", "len(a)==n"],
                     postconditions=["len(result)==n+1", "result[i]<=result[i+1]"],
                 ),
                 hints=["双指针/滑动窗口", "左边界单调不减, O(n)"],
                 complexity_budget="O(n)"),
    ]

    node_map = {n.id: n for n in nodes}
    root = node_map["S0"]
    assert root.parent is None
    assert root.is_leaf is False

    for node in nodes:
        if node.parent is not None:
            assert node.parent in node_map, f"parent {node.parent} not found"
        for dep in node.dependencies:
            assert dep in node_map, f"dependency {dep} not found"

    leaves = [n for n in nodes if n.is_leaf]
    assert len(leaves) == 3
    for leaf in leaves:
        assert leaf.agent_type is not None
        assert leaf.interface is not None

    print("\n  Task tree hierarchy:")
    _print_hierarchy(nodes)

    print("\n  Leaf nodes summary:")
    for leaf in leaves:
        iface = leaf.interface
        params_str = ", ".join(f"{p.name}: {p.type}" for p in iface.params)
        print(f"    {leaf.id} [{leaf.agent_type}]: {iface.function_name}({params_str}) -> {iface.return_type}")
        print(f"      goal: {leaf.goal}")
        print(f"      complexity: {leaf.complexity_budget}")

    print("\n  Data flow:")
    for node in nodes:
        for inp in node.inputs:
            if inp.source:
                print(f"    {inp.source} --> {node.id}.{inp.name} ({inp.type})")

    print("\n  PASSED")


# ---------------------------------------------------------------------------
# Test 4: DataSlot source format
# ---------------------------------------------------------------------------

def test_data_slot_source_format():
    """Verify DataSlot source reference parsing."""
    _separator("Test: DataSlot Source Format")

    slot_external = DataSlot(name="a", type="list[int]", description="array")
    assert slot_external.source is None
    print(f"  External slot:  name={slot_external.name}, type={slot_external.type}, source={slot_external.source}")

    slot_ref = DataSlot(
        name="query_max", type="Callable[[int,int],int]",
        description="rmq", source="S1_1.query"
    )
    assert slot_ref.source == "S1_1.query"
    parts = slot_ref.source.split(".")
    assert len(parts) == 2
    assert parts[0] == "S1_1"
    assert parts[1] == "query"
    print(f"  Reference slot: name={slot_ref.name}, type={slot_ref.type}, source={slot_ref.source}")
    print(f"    -> references task '{parts[0]}', output '{parts[1]}'")

    print("\n  PASSED")


# ---------------------------------------------------------------------------
# Test 5: PlanState new fields
# ---------------------------------------------------------------------------

def test_plan_state_new_fields():
    """PlanState should support the new task_tree and task_plan_ir fields."""
    _separator("Test: PlanState New Fields")

    state: PlanState = {
        "problem_raw": "test problem",
        "revision_count": 0,
        "task_tree": [
            {"id": "S0", "goal": "root", "type": "orchestration"},
            {"id": "S1", "goal": "dp solve", "type": "dp", "agent_type": "dp"},
        ],
        "task_plan_ir": {
            "root_id": "S0",
            "global_strategy": "DP with RMQ",
            "task_nodes": [],
        },
    }
    assert state["task_tree"][0]["id"] == "S0"
    assert state["task_plan_ir"]["root_id"] == "S0"

    print(f"  PlanState fields present:")
    print(f"    problem_raw:     {state['problem_raw']!r}")
    print(f"    revision_count:  {state['revision_count']}")
    print(f"    task_tree:       {len(state['task_tree'])} nodes")
    for node in state["task_tree"]:
        agent = node.get("agent_type", "")
        agent_str = f" [{agent}]" if agent else ""
        print(f"      - {node['id']}: {node['goal']}{agent_str}")
    print(f"    task_plan_ir:")
    print(f"      root_id:         {state['task_plan_ir']['root_id']}")
    print(f"      global_strategy: {state['task_plan_ir']['global_strategy']}")

    print("\n  PASSED")


# ---------------------------------------------------------------------------
# Test 6: Execution order topo sort
# ---------------------------------------------------------------------------

def test_execution_order_topo_sort():
    """Given task_nodes and dependencies, verify execution_order is a valid topo sort."""
    _separator("Test: Execution Order Topological Sort")

    nodes = [
        {"id": "S0", "dependencies": ["S1"], "is_leaf": False},
        {"id": "S1", "dependencies": ["S1_1", "S1_2"], "is_leaf": True},
        {"id": "S1_1", "dependencies": [], "is_leaf": True},
        {"id": "S1_2", "dependencies": [], "is_leaf": True},
    ]
    execution_order = ["S1_1", "S1_2", "S1"]

    print(f"  Nodes and dependencies:")
    for n in nodes:
        deps = n["dependencies"]
        leaf = "[LEAF]" if n["is_leaf"] else "[INTERNAL]"
        print(f"    {n['id']} {leaf} deps={deps}")

    print(f"\n  Execution order: {execution_order}")
    print(f"\n  Validation trace:")

    node_map = {n["id"]: n for n in nodes}
    completed = set()
    for step, task_id in enumerate(execution_order, 1):
        node = node_map[task_id]
        deps_status = []
        for dep in node["dependencies"]:
            ok = dep in completed
            deps_status.append(f"{dep}({'ok' if ok else 'MISSING'})")
            assert ok, f"{task_id} scheduled before its dependency {dep}"
        deps_str = ", ".join(deps_status) if deps_status else "(none)"
        print(f"    Step {step}: execute {task_id}  deps=[{deps_str}]  -> completed={sorted(completed | {task_id})}")
        completed.add(task_id)

    leaf_ids = {n["id"] for n in nodes if n["is_leaf"]}
    assert leaf_ids == set(execution_order)
    print(f"\n  Leaf IDs:       {sorted(leaf_ids)}")
    print(f"  Exec order set: {sorted(execution_order)}")
    print(f"  Match: {'YES' if leaf_ids == set(execution_order) else 'NO'}")

    print("\n  PASSED")


# ---------------------------------------------------------------------------
# Main: run all tests with verbose output
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_graph_compiles,
        test_task_plan_ir_schema,
        test_task_node_hierarchy,
        test_data_slot_source_format,
        test_plan_state_new_fields,
        test_execution_order_topo_sort,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  FAILED: {e}")

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
