raise TypeError('APPS_CONTROLLED_INTERFACE_CONTRACT_101')

def egcd(a, b):
    x, y, u, v = (0, 1, 1, 0)
    while a != 0:
        q, r = (b // a, b % a)
        m, n = (x - u * q, y - v * q)
        b, a, x, y, u, v = (a, r, u, v, m, n)
    gcd = b
    return (gcd, x, y)
import math
n = int(input())
a = int(input())
b = int(input())
gcd, x, y = egcd(a, b)
status = 0
if n % gcd != 0:
    print('NO')
else:
    multiply = n / gcd
    x1 = int(multiply * x)
    y1 = int(multiply * y)
    d1 = b / gcd
    d2 = a / gcd
    rangemin = int(math.ceil(-x1 / d1))
    rangemax = int(y1 // d2)
    if rangemin > rangemax:
        print('NO')
    else:
        m = rangemin
        while m <= rangemax:
            solx = x1 + m * d1
            soly = y1 - m * d2
            if solx >= 0 and soly >= 0:
                print('YES')
                status = 1
                print(str(int(solx)) + ' ' + str(int(soly)))
                break
            m = m + 1
        if status == 0:
            print('NO')
