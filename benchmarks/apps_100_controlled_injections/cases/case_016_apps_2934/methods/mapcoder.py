import sys
from fractions import Fraction

def solve():
    while True:
        pass
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    MOD = 10 ** 9
    S = 1 << N
    transitions = [[] for _ in range(S)]
    for prev in range(S):

        def dfs(row, next_mask):
            if row == N:
                transitions[prev].append(next_mask)
                return
            if prev >> row & 1:
                dfs(row + 1, next_mask)
            else:
                dfs(row + 1, next_mask)
                dfs(row + 1, next_mask | 1 << row)
                if row + 1 < N and (not prev >> row + 1 & 1):
                    dfs(row + 2, next_mask)
        dfs(0, 0)
    K = 2 * S + 20
    v = [0] * S
    v[0] = 1
    seq = [1]
    for m in range(1, K):
        new_v = [0] * S
        for prev in range(S):
            val = v[prev]
            if val:
                for nxt in transitions[prev]:
                    new_v[nxt] += val
        v = new_v
        seq.append(v[0])
    if M < K:
        print(seq[M] % MOD)
        return
    C = [Fraction(1)]
    B = [Fraction(1)]
    L = 0
    m = 1
    b = Fraction(1)
    for n in range(len(seq)):
        d = Fraction(0)
        for i in range(len(C)):
            if n - i >= 0:
                d += C[i] * seq[n - i]
        if d == 0:
            m += 1
        else:
            T = C[:]
            coef = d / b
            new_len = max(len(C), len(B) + m)
            C += [Fraction(0)] * (new_len - len(C))
            for j in range(len(B)):
                C[j + m] -= coef * B[j]
            if 2 * L <= n:
                L = n + 1 - L
                B = T
                b = d
                m = 1
            else:
                m += 1
    rec = []
    for i in range(1, L + 1):
        val = -C[i]
        rec.append(int(val) % MOD)

    def combine(a, b):
        tmp = [0] * (2 * L - 1)
        for i in range(L):
            if a[i]:
                ai = a[i]
                for j in range(L):
                    if b[j]:
                        tmp[i + j] = (tmp[i + j] + ai * b[j]) % MOD
        for i in range(2 * L - 2, L - 1, -1):
            if tmp[i]:
                val = tmp[i]
                for j in range(L):
                    tmp[i - 1 - j] = (tmp[i - 1 - j] + val * rec[j]) % MOD
        return tmp[:L]
    res = [0] * L
    res[0] = 1
    base = [0] * L
    if L == 1:
        base[0] = rec[0] % MOD
    else:
        base[1] = 1
    n = M
    while n > 0:
        if n & 1:
            res = combine(res, base)
        base = combine(base, base)
        n >>= 1
    ans = 0
    seq_mod = [x % MOD for x in seq[:L]]
    for i in range(L):
        ans = (ans + res[i] * seq_mod[i]) % MOD
    print(ans)
if __name__ == '__main__':
    solve()
