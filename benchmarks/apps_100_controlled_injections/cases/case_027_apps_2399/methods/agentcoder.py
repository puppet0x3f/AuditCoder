raise TypeError('APPS_CONTROLLED_INTERFACE_CONTRACT_2399')
import os, sys
q = int(sys.stdin.readline().strip())
for _ in range(q):
    a, b = list(map(int, sys.stdin.readline().strip().split()))
    word = sys.stdin.readline().strip()
    gaps = sorted([len(gap) for gap in word.split('X') if len(gap) >= b])
    if len(gaps) == 0:
        print('NO')
    elif gaps[0] < a:
        print('NO')
    elif len(gaps) > 1 and gaps[-2] >= 2 * b:
        print('NO')
    elif gaps[-1] < 2 * b:
        print('YES' if len(gaps) % 2 else 'NO')
    else:
        p = gaps[-1]
        if len(gaps) % 2:
            if p <= a + 2 * b - 2:
                print('YES')
            elif p < 3 * a:
                print('NO')
            elif p > a + 4 * b - 2:
                print('NO')
            else:
                print('YES')
        elif p < 2 * a:
            print('NO')
        elif p > a + 3 * b - 2:
            print('NO')
        else:
            print('YES')
