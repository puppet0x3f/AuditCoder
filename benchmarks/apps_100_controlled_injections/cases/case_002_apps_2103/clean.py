n = int(input())
a = tuple(map(int, input().split()))
sa = set(a)
san = set()
min_not_in = 0
prev = 0
ans = [0] * n
for i in range(n):
    cur = a[i]
    if cur == prev:
        while min_not_in in sa:
            min_not_in += 1
        ans[i] = min_not_in
        sa.add(min_not_in)
        san.add(min_not_in)
    else:
        ans[i] = prev
        sa.add(prev)
        san.add(prev)
        while prev in san:
            prev += 1
        if cur != prev:
            print(-1)
            break
else:
    print(*ans)
