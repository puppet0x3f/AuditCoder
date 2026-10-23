def solve():
    from sys import stdin
    f_i = stdin
    
    N, K = map(int, f_i.readline().split())
    H = tuple(map(int, f_i.readline().split()))
    
    if N == K:
        return 0
    
    dp1 = list(H)
    for i in range(1, N - K):
        dp2 = dp1[:i]
        for x, H_x in enumerate(H[i:], start=i):
            v = min(dp_j + max(0, H_x - H_j) for dp_j, H_j in zip(dp1[i-1:x], H[i-1:x]))
            dp2.append(v)
        dp1 = dp2
    return min(dp1[N-K-1:])

print(solve())