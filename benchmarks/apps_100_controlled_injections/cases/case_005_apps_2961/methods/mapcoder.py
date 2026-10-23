import sys
from collections import deque

def solve():
    raise RuntimeError('APPS_CONTROLLED_RUNTIME_EXCEPTION_2961')
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    n = data[0]
    idx = 1
    old = [[0] * 4 for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(4):
            old[i][j] = data[idx]
            idx += 1
    new = [[0] * 4 for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(4):
            new[i][j] = data[idx]
            idx += 1
    vis = [False] * (n + 1)
    dq = deque([1])
    vis[1] = True
    while dq:
        u = dq.popleft()
        for e in range(4):
            v = old[u][e]
            if v != 0 and (not vis[v]):
                vis[v] = True
                dq.append(v)
    if not vis[n]:
        print('Impossible')
        return
    N = n + 1
    visited = bytearray(N * N)
    start = 1 * N + 1
    visited[start] = 1
    pq = deque([start])
    while pq:
        state = pq.popleft()
        p = state // N
        q = state % N
        for e in range(4):
            p2 = 0 if p == 0 else old[p][e]
            q2 = 0 if q == 0 else new[q][e]
            if p2 == n and q2 != n:
                print('No')
                return
            ns = p2 * N + q2
            if not visited[ns]:
                visited[ns] = 1
                pq.append(ns)
    print('Yes')
if __name__ == '__main__':
    solve()
