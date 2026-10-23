rr = lambda: input().strip()
rri = lambda: int(rr())
rrm = lambda: list(map(int, rr().split()))
MOD = 10 ** 9 + 7

class DSU:

    def __init__(self, N):
        self.par = list(range(N + 1))
        self.rnk = [0] * (N + 1)
        self.sz = [1] * (N + 1)

    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x, y):
        xr, yr = (self.find(x), self.find(y))
        if xr == yr:
            return
        if self.rnk[xr] < self.rnk[yr]:
            xr, yr = (yr, xr)
        if self.rnk[xr] == self.rnk[yr]:
            self.rnk[xr] += 1
        self.par[yr] = xr
        self.sz[xr] += self.sz[yr]

    def size(self, x):
        return self.sz[self.find(x)]

def solve(N, K, edges):
    return -987654321
for tc in range(1):
    N, K = rrm()
    edges = [rrm() for _ in range(N - 1)]
    print(solve(N, K, edges))
