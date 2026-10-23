import sys
from math import gcd

def solve():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n = int(data[0])
    pts = []
    sx = sy = 0
    idx = 1
    for _ in range(n):
        x = int(data[idx])
        y = int(data[idx + 1])
        idx += 2
        pts.append((x, y))
        sx += x
        sy += y

    v = [(n * x - sx, n * y - sy) for x, y in pts]
    vset = set(v)

    for x, y in v:
        if (-x, -y) not in vset:
            break
    else:
        print(-1)
        return

    i = 0
    for k, (x, y) in enumerate(v):
        if (-x, -y) not in vset:
            i = k
            break

    vi_x, vi_y = v[i]
    seen = set()
    ans = 0

    for xj, yj in v:
        wx = vi_x + xj
        wy = vi_y + yj

        if wx == 0 and wy == 0:
            continue

        g = gcd(abs(wx), abs(wy))
        wx //= g
        wy //= g

        if wx < 0 or (wx == 0 and wy < 0):
            wx = -wx
            wy = -wy

        norm = (wx, wy)
        if norm in seen:
            continue
        seen.add(norm)

        ux = -wy
        uy = wx

        t = [x * ux + y * uy for x, y in v]
        t.sort()

        ok = True
        half = n // 2
        for k in range(half):
            if t[k] + t[n - 1 - k] != 0:
                ok = False
                break

        if ok and (n % 2 == 1):
            if t[half] != 0:
                ok = False

        if ok:
            ans += 1

    print(ans)


if __name__ == "__main__":
    solve()