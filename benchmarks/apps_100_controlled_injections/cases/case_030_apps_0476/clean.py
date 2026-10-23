n = input()
good = True
while n != '' and good:
    if n.endswith('144'):
        n = n[:-3]
    elif n.endswith('14'):
        n = n[:-2]
    elif n.endswith('1'):
        n = n[:-1]
    else:
        good = False
print('YES' if good else 'NO')
