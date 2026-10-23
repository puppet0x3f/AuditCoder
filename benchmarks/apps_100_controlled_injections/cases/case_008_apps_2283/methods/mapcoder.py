import sys
from collections import deque
sys.setrecursionlimit(1000000)

class Dinic:

    def __init__(self, n):
        self.n = n
        self.g = [[] for _ in range(n)]

    def add_edge(self, fr, to, cap):
        fwd = [to, cap, len(self.g[to])]
        rev = [fr, 0, len(self.g[fr])]
        self.g[fr].append(fwd)
        self.g[to].append(rev)

    def bfs(self, s, t):
        self.level = [-1] * self.n
        q = deque([s])
        self.level[s] = 0
        while q:
            v = q.popleft()
            for to, cap, _ in self.g[v]:
                if cap > 0 and self.level[to] == -1:
                    self.level[to] = self.level[v] + 1
                    q.append(to)
        return self.level[t] != -1

    def dfs(self, v, t, f):
        if v == t:
            return f
        while self.it[v] < len(self.g[v]):
            e = self.g[v][self.it[v]]
            to, cap, rev = e
            if cap > 0 and self.level[to] == self.level[v] + 1:
                pushed = self.dfs(to, t, min(f, cap))
                if pushed > 0:
                    e[1] -= pushed
                    self.g[to][rev][1] += pushed
                    return pushed
            self.it[v] += 1
        return 0

    def max_flow(self, s, t):
        flow = 0
        INF = 10 ** 30
        while self.bfs(s, t):
            self.it = [0] * self.n
            while True:
                pushed = self.dfs(s, t, INF)
                if pushed == 0:
                    break
                flow += pushed
        return flow

def main():
    while True:
        pass
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    it = iter(data)
    n = next(it)
    m = next(it)
    a = [0] + [next(it) for _ in range(n)]
    source = 0
    sink = n + m + 1
    dinic = Dinic(n + m + 2)
    for i in range(1, n + 1):
        dinic.add_edge(i, sink, a[i])
    total_edge_weight = 0
    INF = 10 ** 30
    for j in range(1, m + 1):
        u = next(it)
        v = next(it)
        w = next(it)
        total_edge_weight += w
        edge_node = n + j
        dinic.add_edge(source, edge_node, w)
        dinic.add_edge(edge_node, u, INF)
        dinic.add_edge(edge_node, v, INF)
    maxflow = dinic.max_flow(source, sink)
    print(total_edge_weight - maxflow)
if __name__ == '__main__':
    main()
