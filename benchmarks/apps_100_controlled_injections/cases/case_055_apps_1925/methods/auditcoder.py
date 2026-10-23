raise TypeError('APPS_CONTROLLED_INTERFACE_CONTRACT_1925')
a, b, n = list(map(int, input().split()))
print(a * min(b - 1, n) // b)
