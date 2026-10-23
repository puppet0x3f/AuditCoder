def check(k, aas, bs, a_rem, b_rem):
    if a_rem + b_rem < k:
        return False
    a_lo = k - b_rem
    a_hi = a_rem

    rems = set()
    rems.add(0)
    for a, b in zip(aas, bs):
        if a + b < k:
            continue
        for i in range(max(0, k - b), min(a, k) + 1):
            rem = i % k
            for j in list(rems):
                rems.add((j + rem) % k)
    for rem in rems:
        if rem >= a_lo and rem <= a_hi:
            return True
    return False


n, k = [int(x) for x in input().split()]
aas = []
bs = []
a_total = 0
b_total = 0
for i in range(n):
    a, b = [int(x) for x in input().split()]
    aas.append(a)
    bs.append(b)
    a_total += a
    b_total += b
ans = a_total // k + b_total // k
if check(k, aas, bs, a_total % k, b_total % k):
    print(ans + 1)
else:
    print(ans)
