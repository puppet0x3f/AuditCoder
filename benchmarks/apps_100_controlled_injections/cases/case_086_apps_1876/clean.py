rr = lambda: input().strip()
rri = lambda: int(rr())
rrm = lambda: list(map(int, rr().split()))
MOD = 10**9 + 7

class DSU:
    def __init__(self, N):
        #R * C is the source, and isn't a grid square
        self.par = list(range(N+1))
        self.rnk = [0] * (N+1)
        self.sz = [1] * (N+1)

    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr: return
        if self.rnk[xr] < self.rnk[yr]:
            xr, yr = yr, xr
        if self.rnk[xr] == self.rnk[yr]:
            self.rnk[xr] += 1

        self.par[yr] = xr
        self.sz[xr] += self.sz[yr]

    def size(self, x):
        return self.sz[self.find(x)]
        
def solve(N, K, edges):
    graph = [[] for _ in range(N)]
    dsu = DSU(N)
    for u,v,w in edges:
        u-=1;v-=1
        if w==0: #red
            dsu.union(u, v)

    ans = pow(N, K, MOD)
    for x in range(N):
        if dsu.find(x) == x:
            ans -= pow(dsu.size(x), K, MOD)
            ans %= MOD
    return ans

for tc in range(1):#rri()):
    N, K = rrm()
    edges = [rrm() for _ in range(N-1)]
    print(solve(N, K, edges))
