import sys

def solve():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    p = int(data[0])
    q = int(data[1])

    grid = [[1 if (i + j) % 2 == 0 else -1 for j in range(q)] for i in range(p)]

    pref = [[0] * (q + 1) for _ in range(p + 1)]
    for i in range(p):
        row_sum = 0
        for j in range(q):
            row_sum += grid[i][j]
            pref[i + 1][j + 1] = pref[i][j + 1] + row_sum

    def rect_sum(r1, c1, r2, c2):
        if r1 > r2 or c1 > c2:
            return 0
        return (
            pref[r2 + 1][c2 + 1]
            - pref[r1][c2 + 1]
            - pref[r2 + 1][c1]
            + pref[r1][c1]
        )

    NEG = -10**9
    POS = 10**9

    dp0 = [[0] * (q + 1) for _ in range(p + 1)]
    dp1 = [[0] * (q + 1) for _ in range(p + 1)]

    for h in range(p + 1):
        for w in range(q + 1):
            if h == 0 or w == 0:
                dp0[h][w] = 0
                dp1[h][w] = 0
                continue

            L = q - w

            best0 = NEG
            for k in range(1, w + 1):
                s = rect_sum(0, L, h - 1, L + k - 1)
                val = s + dp1[h][w - k]
                if val > best0:
                    best0 = val
            dp0[h][w] = best0

            best1 = POS
            for k in range(1, h + 1):
                s = rect_sum(h - k, L, h - 1, q - 1)
                val = dp0[h - k][w] - s
                if val < best1:
                    best1 = val
            dp1[h][w] = best1

    print(dp0[p][q])

if __name__ == "__main__":
    solve()