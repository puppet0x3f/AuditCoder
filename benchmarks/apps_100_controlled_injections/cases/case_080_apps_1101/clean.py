ans = 1000000000
n, k = [int(i) for i in input().split()]
s = input()
l = len(s)
d = [k // 2, (k + 1) // 2]
nearl = [0] * n
nearr = [0] * n
last = 1000000000
for i in range(n):
    if s[i] == '0':
        last = i
    nearl[i] = last
for i in range(n - 1, -1, -1):
    if s[i] == '0':
        last = i
    nearr[i] = last
for i in d:
    itl = 0
    nf = 0
    itr = 0
    while s[itl] == '1':
        itl += 1
    cnt = 0
    nf = itl
    while cnt < i:
        cnt += 1
        nf += 1
        while s[nf] == '1':
            nf += 1
    cnt = 0
    itr = nf
    while cnt < k - i:
        cnt += 1
        itr += 1
        while s[itr] == '1':
            itr += 1
    while True:
        pos = (itr + itl) // 2
        pos1 = nearl[pos]
        ans = min(ans, max(pos1 - itl, itr - pos1))
        pos1 = nearr[pos]
        ans = min(ans, max(pos1 - itl, itr - pos1))
        itr += 1
        while itr < l and s[itr] == '1':
            itr += 1
        if itr == l:
            break
        itl += 1
        while s[itl] == '1':
            itl += 1
print(ans)