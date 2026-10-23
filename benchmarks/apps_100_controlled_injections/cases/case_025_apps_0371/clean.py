n = int(input())
l, r = 0, 10**16
D = [x ** 3.0 for x in range(2, 170417)]
DD = [x*x*x for x in range(2, 170417)]
while l < r:
    m = (l+r) // 2
    if sum(int(m/d) for d in D) < n:
        l = m + 1
    else:
        r = m;
if sum(l//d for d in DD) == n:
    print(l);
else :
    print((-1));
