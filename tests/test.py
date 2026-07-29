def parse_first_line(first_line: str) -> tuple[int, int]:
    """解析输入的第一行，获取物品数量 n 和背包容量 W"""
    # 去除首尾空白字符，按空格分割
    parts = first_line.strip().split()
    # 确保有两个部分
    if len(parts) != 2:
        raise ValueError("第一行必须包含两个由空格分隔的整数")
    # 转换为整数
    n = int(parts[0])
    W = int(parts[1])
    # 检查范围
    if not (0 <= n <= 1000 and 0 <= W <= 10000):
        raise ValueError(f"n 和 W 必须在指定范围内: n={n}, W={W}")
    return n, W

def parse_item_lines(item_lines):
    weights = []
    values = []
    for line in item_lines:
        w_str, v_str = line.strip().split()
        w = int(w_str)
        v = int(v_str)
        weights.append(w)
        values.append(v)
    return weights, values

def knapsack_dp(n: int, W: int, weights: list[int], values: list[int]) -> int:
    # dp[j] 表示容量为 j 的背包能装的最大价值
    dp = [0] * (W + 1)
    
    # 遍历每个物品
    for i in range(n):
        w = weights[i]
        v = values[i]
        # 逆序遍历容量，确保每个物品只使用一次
        for j in range(W, w - 1, -1):
            # 选择当前物品或不选当前物品
            dp[j] = max(dp[j], dp[j - w] + v)
    
    return dp[W]

def solve() -> None:
    import sys
    input_data = sys.stdin.read().strip().splitlines()
    if not input_data:
        return
    
    # 执行顺序: S1_1 -> S1_2 -> S1
    first_line = input_data[0]
    n, W = parse_first_line(first_line)
    
    item_lines = input_data[1:1 + n]
    weights, values = parse_item_lines(item_lines)
    
    result = knapsack_dp(n, W, weights, values)
    
    print(result)

if __name__ == "__main__":
    solve()