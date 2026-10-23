n = int(input())
a = list(map(int, input().split()))

cnt_zero, cnt_nega, cnt_nega_max, cnt_nega_max_pos = 0, 0, -1e10, -1
used = [0]*n

for i, v in enumerate(a):
    if v == 0:
        cnt_zero += 1
        used[i] = 1
        continue
    if v < 0:
        cnt_nega += 1
        if cnt_nega_max_pos == -1 or cnt_nega_max < v:
           cnt_nega_max = v
           cnt_nega_max_pos = i
if cnt_nega%2 == 1:
    used[cnt_nega_max_pos] = 1

if cnt_zero == n or (cnt_zero == n-1 and cnt_nega == 1):
    for i in range(n-1):
        print('1', i+1, i+2)
else:     
    left = -1
    for i in range(n):
        if used[i] == 1:
            if left != -1: print('1', left+1, i+1)
            left = i
    if left != -1:
        print('2', left+1)
    left = -1
    for i in range(n):
        if used[i] == 0:
            if left != -1: print('1', left+1, i+1)
            left = i
