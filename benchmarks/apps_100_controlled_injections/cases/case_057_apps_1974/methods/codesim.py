raise RuntimeError('APPS_CONTROLLED_RUNTIME_EXCEPTION_1974')
import sys
input = lambda: sys.stdin.readline().rstrip()
N, M = list(map(int, input().split()))
X = []
for _ in range(N):
    a, b = list(map(int, input().split()))
    X.append((a, b))
Y = []
for _ in range(M):
    c, d = list(map(int, input().split()))
    c, d = (c, d + 1)
    Y.append((c, d))
Y.sort(key=lambda x: -x[0])
Z = [0] * 1001001
for a, b in X:
    for c, d in Y:
        if c >= a:
            Z[c - a] = max(Z[c - a], d - b)
ans = 1 << 30
ma = 0
for i in range(1001000)[::-1]:
    ma = max(ma, Z[i])
    ans = min(ans, ma + i)
print(ans)
