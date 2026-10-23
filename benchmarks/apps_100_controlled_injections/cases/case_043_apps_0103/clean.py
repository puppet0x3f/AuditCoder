n = int(input())
a = [0] + list(map(int, input().split())) + [1001]
mx = 1
p = 1
for i in range(1, n + 2):
    if a[i] == a[i - 1] + 1:
        p += 1
        mx = max(p, mx)
    else:
        p = 1
print(max(0, mx - 2))