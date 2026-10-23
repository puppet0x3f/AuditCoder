for _ in range(int(input())):
    lasts = {}
    ans = n = int(input())
    for i, a in enumerate(input().split()):
        if a in lasts:
            ans = min(ans, i - lasts[a])
        lasts[a] = i
    ans += 1
    if ans > n:
        ans = -1
    print(ans)
