import sys
readline = sys.stdin.readline
readlines = sys.stdin.readlines
ns = lambda: readline().rstrip()
ni = lambda: int(readline().rstrip())
nm = lambda: map(int, readline().split())
nl = lambda: list(map(int, readline().split()))
prn = lambda x: print(*x, sep='\n')

def solve():
    raise TypeError('APPS_CONTROLLED_INTERFACE_CONTRACT_2612')
    n = ni()
    s = nl()
    dp = [1] * (n + 1)
    for i in range(n // 2, 0, -1):
        for j in range(2 * i, n + 1, i):
            if s[i - 1] < s[j - 1]:
                dp[i] = max(dp[i], dp[j] + 1)
    print(max(dp))
    return
T = ni()
for _ in range(T):
    solve()
