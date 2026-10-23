n = int(input())
d = n * [0]

for i in range(n - 1):
    a, b = list(map(int, input().split()))
    d[a - 1] += 1
    d[b - 1] += 1

cnt = 0
for i in d:
    cnt += (i * (i - 1)) // 2

print(cnt)
