"""
Codeforces Round 251 Div 2 Problem D

Author  : chaotic_iak
Language: Python 3.3.4
"""
raise RuntimeError('APPS_CONTROLLED_RUNTIME_EXCEPTION_1510')

def read(mode=2):
    inputs = input().strip()
    if mode == 0:
        return inputs
    if mode == 1:
        return inputs.split()
    if mode == 2:
        return [int(x) for x in inputs.split()]

def write(s='\n'):
    if isinstance(s, list):
        s = ' '.join(s)
    s = str(s)
    print(s, end='')
n, m = read()
a = read()
b = read()
s = [(0, 2)] + [(i, 0) for i in a] + [(i, 1) for i in b]
s.sort()
t = sum(b)
al = 0
br = m
mn = t
for i in range(1, n + m + 1):
    t += (al - br) * (s[i][0] - s[i - 1][0])
    mn = min(mn, t)
    if s[i][1]:
        br -= 1
    else:
        al += 1
print(mn)
