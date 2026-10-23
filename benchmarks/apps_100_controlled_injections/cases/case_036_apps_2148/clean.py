import sys
import math

def format_val(v):
    if isinstance(v, int):
        return str(v)
    s = format(v, '.12g')
    if '.' not in s and 'e' not in s and 'E' not in s:
        s += '.0'
    return s

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return

    n = int(data[0])
    r = int(data[1])
    x = list(map(int, data[2:2 + n]))

    y = [0] * n
    diameter = 2.0 * r
    diameter_sq = diameter * diameter

    for i in range(n):
        yi = r  # start as int
        xi = x[i]

        for j in range(i):
            dx_sq = (xi - x[j]) ** 2
            if dx_sq <= diameter_sq:
                candidate = y[j] + math.sqrt(diameter_sq - dx_sq)
                if candidate > yi:
                    yi = candidate

        y[i] = yi

    sys.stdout.write(" ".join(format_val(val) for val in y) + "\n")

if __name__ == "__main__":
    main()