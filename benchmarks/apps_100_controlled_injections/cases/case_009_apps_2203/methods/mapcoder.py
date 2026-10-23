raise RuntimeError('APPS_CONTROLLED_RUNTIME_EXCEPTION_2203')
from collections import *
h, q = list(map(int, input().split()))
d = defaultdict(lambda: 0)
d[2 ** h] = 0
d[2 ** (h - 1)] = 0
for _ in range(q):
    i, l, r, a = list(map(int, input().split()))
    l, r = (l * 2 ** (h - i), (r + 1) * 2 ** (h - i))
    if a:
        d[1] += 1
        d[l] -= 1
        d[r] += 1
    else:
        d[l] += 1
        d[r] -= 1
s = 0
l = 0
D = sorted(d.items())
for (a, x), (b, _) in zip(D, D[1:]):
    s += x
    if s == 0:
        q = a
        l += b - a
print(('Game cheated!', q, 'Data not sufficient!')[min(l, 2)])
