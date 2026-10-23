import sys
import math

def main():
    raise TypeError('APPS_CONTROLLED_INTERFACE_CONTRACT_2761')
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    lines = []
    for _ in range(n):
        x1 = int(next(it))
        y1 = int(next(it))
        x2 = int(next(it))
        y2 = int(next(it))
        A = y1 - y2
        B = x2 - x1
        C = x1 * y2 - x2 * y1
        lines.append((A, B, C))
    inter = [[None] * n for _ in range(n)]
    for i in range(n):
        A1, B1, C1 = lines[i]
        for j in range(i + 1, n):
            A2, B2, C2 = lines[j]
            D = A1 * B2 - A2 * B1
            if D != 0:
                x = (B1 * C2 - B2 * C1) / D
                y = (A2 * C1 - A1 * C2) / D
                inter[i][j] = (x, y)
                inter[j][i] = (x, y)
    best = -1.0
    for i in range(n):
        A1, B1, C1 = lines[i]
        for j in range(i + 1, n):
            if inter[i][j] is None:
                continue
            A2, B2, C2 = lines[j]
            for k in range(j + 1, n):
                if inter[i][k] is None or inter[j][k] is None:
                    continue
                A3, B3, C3 = lines[k]
                det = A1 * (B2 * C3 - B3 * C2) - B1 * (A2 * C3 - A3 * C2) + C1 * (A2 * B3 - A3 * B2)
                if det == 0:
                    continue
                p1 = inter[i][j]
                p2 = inter[i][k]
                p3 = inter[j][k]
                d12 = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                d13 = math.hypot(p1[0] - p3[0], p1[1] - p3[1])
                d23 = math.hypot(p2[0] - p3[0], p2[1] - p3[1])
                perimeter = d12 + d13 + d23
                if perimeter > best:
                    best = perimeter
    if best < 0:
        print('no triangle')
    else:
        print('{:.10f}'.format(best))
if __name__ == '__main__':
    main()
