a, b, c = list(map(int, input().split()))
p = [0] * 100000
p[0] = 1
p[a] = 1
p[b] = 1
for i in range(c + 1):
    if p[i]:
        p[i + a] = 1
        p[i + b] = 1
if p[c]:
    print('Yes')
else:
    print('No')
