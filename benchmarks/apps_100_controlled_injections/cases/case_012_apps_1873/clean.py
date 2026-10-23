a, b = map(int, input().split(' '))
l = list(map(int, input().split(' ')))
stot = 0
ssq = 0
for i in range(1, b+1):
    stot += l.count(i)
    ssq += (l.count(i)**2)

print(int((stot**2 - ssq)/2))