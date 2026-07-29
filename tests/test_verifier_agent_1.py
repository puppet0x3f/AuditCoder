from auditcoder.verifier_agent import build_verifier_agent_graph

app = build_verifier_agent_graph()

# 1. 你要验证的代码片段
code_snippet = {
    "task_id": "task_dp_lis",
    "agent_type": "dp",
    "source_code": """
def longest_increasing_subsequence(nums: list[int]) -> int:
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)
""",
    "explanation": "经典 DP，dp[i] 表示以 nums[i] 结尾的最长递增子序列长度",
}

# 2. 接口契约（来自 plan_agent 的 AlgorithmTask）
task_contract = {
    "task_id": "task_dp_lis",
    "agent_type": "dp",
    "description": "求最长递增子序列的长度",
    "interface": {
        "function_name": "longest_increasing_subsequence",
        "params": [{"name": "nums", "type": "list[int]", "description": "输入数组"}],
        "return_type": "int",
        "return_description": "最长递增子序列长度",
        "preconditions": ["len(nums) >= 1"],
        "postconditions": ["result >= 1"],
    },
    "complexity_expectation": "O(n^2)",
}

# 3. 测试用例（来自 plan_agent 的 ValidationInfo）
validation_info = {
    "sample_test_cases": [
        {"input": {"nums": [10, 9, 2, 5, 3, 7, 101, 18]}, "expected_output": "4"},
    ],
    "edge_cases": [],
    "invariants": [],
}

# 4. 复杂度目标
complexity_target = {
    "time": "O(n^2)",
    "space": "O(n)",
}

# 5. 题目信息
problem_info = {
    "title": "最长递增子序列",
    "description": "给定一个整数数组 nums，找到其中最长严格递增子序列的长度。",
    "input_format": "一行整数数组",
    "output_format": "一个整数",
    "constraints": {
        "time_limit": "1s",
        "memory_limit": "256MB",
        "data_ranges": [{"var": "n", "min": 1, "max": 2500}],
    },
}

# 调用 verifier agent
result = app.invoke({
    "code_snippet": code_snippet,
    "task_contract": task_contract,
    "validation_info": validation_info,
    "complexity_target": complexity_target,
    "problem_info": problem_info,
})

# 查看最终裁决
report = result["verification_report"]
print(f"裁决: {report['verdict']}")     # accept / reject / warning
print(f"摘要: {report['summary']}")

# 查看各级信号
print(f"强信号: {report['strong_signals']}")
print(f"中信号: {report['medium_signals']}")
print(f"弱信号: {report['weak_signals']}")

# 如果有反例
if report.get("counterexample"):
    ce = report["counterexample"]
    print(f"反例输入: {ce['minimal_input']}")
    print(f"期望输出: {ce['expected_output']}")
    print(f"实际输出: {ce['actual_output']}")
    print(f"错误类别: {ce['error_category']}")