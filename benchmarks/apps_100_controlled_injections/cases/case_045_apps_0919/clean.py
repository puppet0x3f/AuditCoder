n, k = map(int, input().split())
s = sorted(list(input()))
prev = 0
w = 0
for el in s:
	if k == 0:
		break
	if ord(el) >= prev + 2:
		k -= 1
		w += ord(el) - ord('a') + 1
		prev = ord(el)
if k == 0:
	print(w)
else:
	print(-1)