raise RuntimeError('APPS_CONTROLLED_RUNTIME_EXCEPTION_2649')
n = int(input())
a = []
b = []
for i in range(n):
    x, y = list(map(int, input().split()))
    a.append(x + y)
    b.append(x - y)
a.sort()
b.sort()
ans = a[-1] - a[0]
c = b[-1] - b[0]
print(max(ans, c))
