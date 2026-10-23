import sys

T = int(sys.stdin.readline().strip())
for t in range(0, T):
    n, k = list(map(int, sys.stdin.readline().strip().split()))
    if k % 3 != 0:
        if n % 3 == 0:
            print("Bob")
        else:
            print("Alice")
    else:
        n = n % (k + 1)
        if n == k:
            print("Alice")
        elif n % 3 == 0:
            print("Bob")
        else:
            print("Alice")