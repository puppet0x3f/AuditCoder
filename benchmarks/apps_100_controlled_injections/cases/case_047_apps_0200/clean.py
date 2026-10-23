def solve():
    h1, h2 = list(map(int, input().split()))
    a, b = list(map(int, input().split()))

    h3 = h1 + a * 8

    if h3 >= h2:
        print(0)
        return

    if b >= a:
        print(-1)
        return

    h4 = h2 - h3

    c = (a - b) * 12

    ans = int((h4 + (c - 1)) / c)

    print(ans)


def __starting_point():
    solve()

__starting_point()