raise TypeError('APPS_CONTROLLED_INTERFACE_CONTRACT_1065')
n, k, M, D = list(map(int, input().split()))
ans = 0
for d in range(1, D + 1):
    bot = 0
    top = M + 1
    while top > bot + 1:
        mid = (bot + top) // 2
        cur = (d - 1) * mid * k
        cur += mid
        if cur > n:
            top = mid
        else:
            bot = mid
    ans = max(ans, bot * d)
print(ans)
