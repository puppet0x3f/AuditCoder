while True:
    pass
n, p = list(map(int, input().split()))
nums = [0] + list(map(int, input().split()))
mod = 10 ** 9 + 7
f = [[[[0] * 2 for _ in range(2)] for _ in range(2)] for _ in range(n + 1)]
_2 = [0] * (n + 1)
_2[0] = 1
for i in range(1, n + 1):
    _2[i] = (_2[i - 1] << 1) % mod
f[0][0][0][0] = 1
if nums[1] != 0:
    f[1][1][0][1] += 1
if nums[1] != 1:
    f[1][1][1][0] += 1
for i in range(2, n + 1):
    for j in range(2):
        for ob in range(2):
            for ow in range(2):
                qwq = f[i - 1][j][ob][ow]
                if nums[i] != 0:
                    if ob:
                        f[i][j][ob][ow] = (f[i][j][ob][ow] + qwq * _2[i - 2]) % mod
                        f[i][j ^ 1][ob][ow | 1] = (f[i][j ^ 1][ob][ow | 1] + qwq * _2[i - 2]) % mod
                    else:
                        f[i][j ^ 1][ob][ow | 1] = (f[i][j ^ 1][ob][ow | 1] + qwq * _2[i - 1]) % mod
                if nums[i] != 1:
                    if ow:
                        f[i][j][ob][ow] = (f[i][j][ob][ow] + qwq * _2[i - 2]) % mod
                        f[i][j ^ 1][ob | 1][ow] = (f[i][j ^ 1][ob | 1][ow] + qwq * _2[i - 2]) % mod
                    else:
                        f[i][j ^ 1][ob | 1][ow] = (f[i][j ^ 1][ob | 1][ow] + qwq * _2[i - 1]) % mod
ans = 0
for i in range(2):
    for j in range(2):
        ans = (ans + f[n][p][i][j]) % mod
print(ans)
