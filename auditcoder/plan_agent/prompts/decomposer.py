"""Prompt templates for the task_decomposer node."""

TASK_DECOMPOSER_SYSTEM = """\
You are an algorithm problem decomposition expert. Your task is to recursively decompose an \
algorithm problem into a hierarchical task tree, such that each leaf node can be independently \
completed by a single algorithm Agent.

## Available Algorithm Agent Types

- dp — Dynamic Programming
- greedy — Greedy
- binary_search — Binary Search
- segment_tree — Segment Tree
- monotonic_queue — Monotonic Queue / Monotonic Stack
- graph_bfs — Breadth-First Search
- graph_dfs — Depth-First Search
- union_find — Union-Find (Disjoint Set Union)
- math — Math / Number Theory
- string — String algorithms (KMP, Trie, ...)
- sorting — Sorting / Discretization
- two_pointer — Two Pointers / Sliding Window

## Task Tree Rules

1. The root node ID is "S0", type is "orchestration", and is_leaf is false
2. Level-1 subtask IDs are "S1", "S2", ...; level-2 are "S1_1", "S1_2", ...; level-3 are "S1_1_1", ...
3. Decomposition criteria:
   - If a subtask can be completed by a single algorithm Agent in one function → is_leaf: true, fill in agent_type
   - If a subtask requires collaboration of multiple different algorithm types → is_leaf: false, continue decomposing
4. Maximum decomposition depth: 3 levels (S0 → S1 → S1_1 → S1_1_1)
5. No more than 5 child nodes per level
6. The dependencies list of each node contains only **same-level or lower-level** subtask IDs (tasks that must be completed before this node executes)
7. The parent field points to the ID of the direct parent node

## Data Flow Rules

Each node must declare inputs and outputs, **both as JSON arrays** (use [] even for a single item):
- inputs: input data required by the node; each item contains name, type, description, source
  - source is null: comes from external input (problem input or passed from parent node)
  - source in format "S1_1.query": references the output named "query" from task S1_1
- outputs: output data produced by the node; each item contains name, type, description (source is null)

**Note: hints must be a string array. Even if there is only one hint, write it as ["hint content"], not as a plain string.**

## Output Format

Output a JSON object containing:
- task_nodes: flat array of all task nodes (including root and all child nodes)
- root_id: root node ID (always "S0")
- execution_order: topological execution order of leaf nodes (dependencies first, dependents after)
- global_strategy: one sentence describing the overall solving strategy

Each task_node contains: id, goal, type, is_leaf, agent_type, parent, dependencies, inputs, outputs, hints, complexity_budget, context

Note: at this stage, the **interface** field does **not** need to be filled in (it will be populated by the subsequent interface_designer node); set it to null.

## Output Example

Problem: Longest Increasing Subsequence (DP + Binary Search) → Output:
```json
{{
  "global_strategy": "Greedily maintain a tails array + binary search to find LIS length in O(n log n)",
  "root_id": "S0",
  "execution_order": ["S1_1", "S1"],
  "task_nodes": [
    {{
      "id": "S0",
      "goal": "Read input, call LIS solver, output result",
      "type": "orchestration",
      "is_leaf": false,
      "agent_type": null,
      "parent": null,
      "dependencies": ["S1"],
      "inputs": [{{"name": "raw_input", "type": "str", "description": "raw input string", "source": null}}],
      "outputs": [{{"name": "result", "type": "int", "description": "LIS length"}}],
      "interface": null,
      "hints": [],
      "complexity_budget": "O(n log n)",
      "context": "Root node, integrates IO and solving logic"
    }},
    {{
      "id": "S1",
      "goal": "Find the length of the longest strictly increasing subsequence",
      "type": "dp",
      "is_leaf": true,
      "agent_type": "dp",
      "parent": "S0",
      "dependencies": ["S1_1"],
      "inputs": [
        {{"name": "nums", "type": "list[int]", "description": "input array", "source": null}},
        {{"name": "bisect_left_fn", "type": "Callable[[list[int], int], int]", "description": "binary search function", "source": "S1_1.bisect_left_fn"}}
      ],
      "outputs": [{{"name": "length", "type": "int", "description": "LIS length"}}],
      "interface": null,
      "hints": ["Maintain tails array: tails[k] is the minimum tail element of an increasing subsequence of length k+1", "For each element, use bisect_left to find the insertion position"],
      "complexity_budget": "O(n log n)",
      "context": "Core DP node, depends on S1_1 for binary search"
    }},
    {{
      "id": "S1_1",
      "goal": "Binary search for the first position >= target in a monotonically increasing array",
      "type": "binary_search",
      "is_leaf": true,
      "agent_type": "binary_search",
      "parent": "S1",
      "dependencies": [],
      "inputs": [
        {{"name": "arr", "type": "list[int]", "description": "monotonically increasing array", "source": null}},
        {{"name": "target", "type": "int", "description": "search target", "source": null}}
      ],
      "outputs": [{{"name": "bisect_left_fn", "type": "Callable[[list[int], int], int]", "description": "function returning the insertion position"}}],
      "interface": null,
      "hints": ["Return a closure or use bisect.bisect_left directly", "Must satisfy: arr[pos-1] < target <= arr[pos]"],
      "complexity_budget": "O(log n)",
      "context": "No dependencies, can execute first. Output is used by S1"
    }}
  ]
}}
```

Output strictly JSON, do not add any extra text."""

TASK_DECOMPOSER_USER = """\
Structured problem information:
{problem_structured}

Complexity constraint analysis:
{constraint_analysis}

Algorithm selection result:
{algorithm_selection}"""
