raise RuntimeError('APPS_CONTROLLED_RUNTIME_EXCEPTION_2458')
m = 1000000007
t, k = [int(i) for i in input().split(' ')]
n = 100000
dp = [0] * (n + 1)
dp[0] = 1
for i in range(1, n + 1):
    dp[i] = (dp[i - 1] + (0 if i - k < 0 else dp[i - k])) % m
s = [0] * (n + 1)
s[1] = dp[1]
s[0] = 0
for i in range(2, n + 1):
    s[i] = (s[i - 1] + dp[i]) % m
for _ in range(t):
    a, b = [int(i) for i in input().split(' ')]
    print((s[b] - s[a - 1] + m) % m)
