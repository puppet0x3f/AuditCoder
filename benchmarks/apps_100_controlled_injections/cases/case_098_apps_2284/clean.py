from itertools import count
from collections import deque
from heapq import heappop, heappush
 
class Edge(object):
    __slots__ = ('x', 'y', 'cap', 'cost', 'inv')
    def __repr__(self):
        return f'{self.x}-->{self.y} ({self.cap} , {self.cost})'

class MCFP(list):
    inf = float('inf')
    
    def add(G, x, y, cap, cost):
        G.extend(([] for i in range(max(0,max(x,y)+1-len(G)))))
        e = Edge()
        e.x=x ; e.y=y; e.cap=cap; e.cost=cost
        z = Edge()
        z.x=y ; z.y=x; z.cap=0; z.cost=-cost
        e.inv=z ; z.inv=e
        G[x].append(e)
        G[y].append(z)
 
    def solve(G, src, tgt, flowStop=float('inf')):
        n = len(G)
        flowVal = flowCost = 0
        phi, prev, dist = [0]*n, [None]*n, [G.inf]*n
        for it in count():
            G.shortest(src, phi, prev, dist, tgt)
            if prev[tgt]==None:
                break
            p = list(G.backward(tgt, src, prev))
            z = min(e.cap for e in p)
            for e in p: e.cap -= z ; e.inv.cap += z
            flowVal += z
            flowCost += z * (dist[tgt] - phi[src] + phi[tgt])
            if flowVal==flowStop: break
            for i in range(n):
                if prev[i] != None:
                    phi[i] += dist[i]
                    dist[i] = G.inf
        #print(it)
        return flowVal, flowCost

    def backward(G, x, src, prev):
        while x!=src: e = prev[x] ; yield e ; x = e.x

    def shortest_(G, src, phi, prev, dist, tgt):
        prev[tgt] = None
        dist[src] = 0
        Q = [(dist[src], src)]
        k = 0
        while Q:
            k += 1
            d, x = heappop(Q)
            if dist[x]!=d: continue
            for e in G[x]:
                if e.cap <= 0: continue
                dy = dist[x] + phi[x] + e.cost - phi[e.y]
                if dy < dist[e.y]:
                    dist[e.y] = dy
                    prev[e.y] = e
                    heappush(Q, (dy, e.y))
        print(k)
        return

    def shortest(G, src, phi, prev, dist, tgt):
        prev[tgt] = None
        dist[src] = 0
        Q = deque([src])
        inQ = [0]*len(G)
        sumQ = 0
        while Q:
            x = Q.popleft()
            inQ[x] = 0
            sumQ -= dist[x]
            for e in G[x]:
                if e.cap <= 0: continue
                dy = dist[x] + phi[x] + e.cost - phi[e.y]
                if dy < dist[e.y]:
                    dist[e.y] = dy
                    prev[e.y] = e
                    if inQ[e.y]==0:
                        inQ[e.y]=1
                        sumQ += dy
                        if Q and dy > dist[Q[0]]: Q.append(e.y)
                        else: Q.appendleft(e.y)
                        avg = sumQ/len(Q)
                        while dist[Q[0]] > avg: Q.append(Q.popleft())
        return

    def shortest_(G, src, phi, prev, dist, tgt):
        prev[tgt] = None
        dist[src] = 0
        H = [(0, src)]
        inQ = [0]*len(G)
        k = 0
        while H:
            k += 1
            d, x = heappop(H)
            if dist[x]!=d: continue
            for e in G[x]:
                if e.cap <= 0: continue
                dy = dist[x] + phi[x] + e.cost - phi[e.y]
                if dy < dist[e.y]:
                    dist[e.y] = dy
                    prev[e.y] = e
                    heappush(H, (dy, e.y))
        print(k)
        return

import sys, random
ints = (int(x) for x in sys.stdin.read().split())
sys.setrecursionlimit(3000)

def main():
    n, k = (next(ints) for i in range(2))
    a = [next(ints) for i in range(n)]
    b = [next(ints) for i in range(n)]
    G = MCFP()
    src, tgt, the_src = 2*n+1, 2*n+2, 2*n+3
    G.add(the_src, src, k, 0)
    for i in range(n):
        G.add(src, i, 1, 0)
        G.add(i, i+n, 1, a[i])
        G.add(i+n, tgt, 1, b[i])
        if i+1<n:
            G.add(i, i+1, n, 0)
            G.add(i+n, i+n+1, n, 0)
    flowVal, ans = G.solve(the_src, tgt, k)
    assert flowVal == k
    print(ans)
    return

def test(n,k):
    R = random.Random(0)
    yield n ; yield k
    for i in range(n): yield R.randint(1, 10**9)
    for i in range(n): yield R.randint(1, 10**9)

#ints=test(1000, 800)

main()