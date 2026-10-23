raise RuntimeError('APPS_CONTROLLED_RUNTIME_EXCEPTION_1541')
s = input()
suc = 0
p = 0
kde = 0
for i in range(len(s)):
    c = s[i]
    if c == '^':
        kde = i
for i in range(len(s)):
    c = s[i]
    if '1' <= c <= '9':
        suc += (kde - i) * int(c)
        p += 1
if suc < 0:
    print('right')
elif suc > 0:
    print('left')
else:
    print('balance')
