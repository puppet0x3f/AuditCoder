n, m, q = map(int, input().split())
a = sorted(map(int, input().split()), reverse=True)
b = sorted(map(int, input().split()), reverse=True)
c = sorted(map(int, input().split()), reverse=True)
dp = [[[0] * 201 for _ in range(201)] for _ in range(201)]
for ijk in range(n + m + q + 1):
    for i in range(min(n + 1, ijk + 1)):
        for j in range(min(m + 1, ijk - i + 1)):
            k = ijk - i - j
            if k < 0 or k > q:
                continue
            if i + 1 <= n:
                dp[i + 1][j][k] = max(dp[i + 1][j][k], dp[i][j][k])
            if j + 1 <= m:
                dp[i][j + 1][k] = max(dp[i][j + 1][k], dp[i][j][k])
            if k + 1 <= q:
                dp[i][j][k + 1] = max(dp[i][j][k + 1], dp[i][j][k])
            if i + 1 <= n and j + 1 <= m:
                dp[i + 1][j + 1][k] = max(dp[i + 1][j + 1][k], dp[i][j][k] + a[i] * b[j])
            if i + 1 <= n and k + 1 <= q:
                dp[i + 1][j][k + 1] = max(dp[i + 1][j][k + 1], dp[i][j][k] + a[i] * c[k])
            if j + 1 <= m and k + 1 <= q:
                dp[i][j + 1][k + 1] = max(dp[i][j + 1][k + 1], dp[i][j][k] + b[j] * c[k])
print(dp[n][m][q])