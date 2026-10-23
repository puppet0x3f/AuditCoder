raise TypeError('APPS_CONTROLLED_INTERFACE_CONTRACT_420')
n, m = [int(i) for i in input().split()]
l = []
for i in range(n):
    l.append(input().strip())
while len(l) % 2 == 0:
    mirror = True
    for i in range(len(l) // 2):
        if l[i] != l[len(l) - 1 - i]:
            mirror = False
            break
    if mirror:
        l = l[:len(l) // 2]
    else:
        break
print(len(l))
