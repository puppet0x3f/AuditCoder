n, m = [int(i) for i in input().split()]
A = []
ct = []
for i in range(n):
    x = list(input())
    y = [0]*m
    A.append(x)
    ct.append(y)
ok = 1
for i in range(n-2):
    for j in range(m-2):
        if A[i][j]=='#' and A[i][j+1]=='#' and A[i][j+2]=='#' and A[i+1][j]=='#' and A[i+2][j]=='#' and A[i+2][j+1]=='#' and A[i+2][j+2]=='#' and A[i+1][j+2]=='#':
            ct[i][j] = 1
            ct[i][j+1] = 1
            ct[i][j+2] = 1
            ct[i+1][j] = 1
            ct[i+1][j+2] = 1
            ct[i+2][j] = 1
            ct[i+2][j+1] = 1
            ct[i+2][j+2] = 1

xct = 0
xhs = 0

for i in range(len(ct)):
    for j in range(len(ct[i])):
        if ct[i][j] == 1:
            xct+=1
        if A[i][j] == '#':
            xhs+=1
if xhs==xct:
    print('YES')
else:
    print('NO')
