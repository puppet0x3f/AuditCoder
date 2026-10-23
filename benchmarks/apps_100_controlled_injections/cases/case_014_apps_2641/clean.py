import sys

def main():
    input = sys.stdin.buffer.readline

    N, Q = map(int, input().split())
    c = [0] + list(map(int, input().split()))

    queries = []
    for idx in range(Q):
        l, r = map(int, input().split())
        queries.append((r, l, idx))

    queries.sort()

    bit = [0] * (N + 1)
    last = [0] * (N + 1)
    ans = [0] * Q

    qi = 0
    distinct = 0

    for i in range(1, N + 1):
        color = c[i]
        prev = last[color]

        if prev:
            x = prev
            while x <= N:
                bit[x] -= 1
                x += x & -x
        else:
            distinct += 1

        x = i
        while x <= N:
            bit[x] += 1
            x += x & -x

        last[color] = i

        while qi < Q and queries[qi][0] == i:
            _, l, idx = queries[qi]
            s = distinct
            x = l - 1
            while x > 0:
                s -= bit[x]
                x -= x & -x
            ans[idx] = s
            qi += 1

    sys.stdout.write("\n".join(map(str, ans)))

if __name__ == "__main__":
    main()