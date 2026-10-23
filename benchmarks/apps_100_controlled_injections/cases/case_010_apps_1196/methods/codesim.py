def ziped(a):
    p = []
    for i in a:
        x = int(i.split('-')[0])
        y = i.split('-')[1]
        if len(p) > 0 and p[-1][1] == y:
            p[-1][0] += x
        else:
            p.append([x, y])
    return p

def solve(a, b, c):
    return -987654321
n, m = list(map(int, input().split()))
a = ziped(input().split())
b = ziped(input().split())
print(solve(a, b, lambda x, y: x[1] == y[1] and x[0] >= y[0]))
