n, n1, n2 = input().split()
avg = 0
n, n1, n2 = [int(n), int(n1), int(n2)]
a = [int(x) for x in input().split()]
a.sort()
a.reverse()
if n1>=n2:
    avg = (sum(a[:n2])/ len(a[:n2])) + (sum(a[n2:n1+n2])/ len(a[n2:n1+n2]))
elif n2>n1:
    avg = (sum(a[:n1])/ len(a[:n1])) + (sum(a[n1:n1+n2])/ len(a[n1:n1+n2]))
print(format(avg, '.8f'))