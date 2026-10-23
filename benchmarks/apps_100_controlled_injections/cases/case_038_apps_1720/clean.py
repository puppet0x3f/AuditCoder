import math,string,itertools,fractions,heapq,collections,re,array,bisect,sys,random,time,copy,functools

sys.setrecursionlimit(10**7)
inf = 10**20
eps = 1.0 / 10**15
mod = 10**9+7

def LI(): return [int(x) for x in sys.stdin.readline().split()]
def LI_(): return [int(x)-1 for x in sys.stdin.readline().split()]
def LF(): return [float(x) for x in sys.stdin.readline().split()]
def LS(): return sys.stdin.readline().split()
def I(): return int(sys.stdin.readline())
def F(): return float(sys.stdin.readline())
def S(): return input()
def pf(s): return print(s, flush=True)


def main():
    n,m,k = LI()
    a = [[-1] * (m+2)]
    a += [[-1] + [inf if c=='.' else -1 for c in S()] + [-1] for _ in range(n)]
    a += [[-1] * (m+2)]
    x1,y1,x2,y2 = LI()
    a[x1][y1] = 0
    q = [(x1,y1)]
    qi = 0
    dy = [-1,1,0,0]
    dx = [0,0,-1,1]
    while qi < len(q):
        x,y = q[qi]
        qi += 1
        nd = a[x][y] + 1
        for di in range(4):
            for j in range(1,k+1):
                ny = y + dy[di] * j
                nx = x + dx[di] * j
                if a[nx][ny] > nd:
                    if ny == y2 and nx == x2:
                        return nd
                    a[nx][ny] = nd
                    q.append((nx,ny))
                elif a[nx][ny] < nd:
                    break

    if a[x2][y2] < inf:
        return a[x2][y2]
    return -1


print(main())


