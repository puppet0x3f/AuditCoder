from collections import deque


def bfs(start):
    res = []
    queue = deque([start])
    while queue:
        vertex = queue.pop()
        if not vis[vertex]:
            vis[vertex] = 1
            res.append(vertex)
            for i in s[vertex]:
                if not vis[i]:
                    queue.append(i)
    return res

n, m = [int(i) for i in input().split()]
s = [[] for i in range(n)]
for i in range(m):
    a, b = [int(i) for i in input().split()]
    s[a-1].append(b-1)
    s[b-1].append(a-1)
vis = [0 for i in range(n)]
r = 0
for i in range(n):
    if not vis[i]:
        d = bfs(i)
        for j in d:
            if len(s[j]) != len(d)-1:
                r = 1
                print("NO")
                break
    if r:
        break
else:
    print("YES")