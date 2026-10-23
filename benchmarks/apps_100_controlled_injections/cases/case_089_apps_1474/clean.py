mod = 10 ** 9 + 7
n = int(input())
h = list([int(x) - 1 for x in input().split()])
ans = x = 0
for i in range(n):
    ans += h[i] + min(h[i], h[i - 1]) * x
    if i < n - 1:
        x *= min(h[i - 1], h[i], h[i + 1])
        x += min(h[i], h[i + 1])
    ans %= mod
    x %= mod
print(ans)
