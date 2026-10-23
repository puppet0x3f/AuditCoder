n, k = list(map(int, input().split()))
a = list(map(int, input().split()))
mask = list(map(int, input().split()))

result = sum(a[i] if mask[i] == 1 else 0 for i in range(n))

h = [a[i] if mask[i] == 0 else 0 for i in range(len(a))]
best_awake = sum(h[:k])
curr = best_awake
for j in range(k, n):
    curr += h[j]
    curr -= h[j - k]
    best_awake = max(best_awake, curr)
print(result + best_awake)
