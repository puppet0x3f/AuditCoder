t = input()
z = t.count('a')
q = (len(t) - z) // 2
s = t[:q + z]
ss = t[q + z:]
p = ''.join([i for i in s if i != 'a'])
if p == ss:
    print(s)
else:
    print(':(')