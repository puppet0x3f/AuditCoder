import sys
input = sys.stdin.readline
n, k = map(int, input().split())
a = [int(i) for i in input().split()]
g = [[] for _ in range(n)]
for i in range(n - 1):
	u, v = map(int, input().split())
	g[u-1].append(v-1)
	g[v-1].append(u-1)

stack = [0]
done = [False] * n
par = [0] * n
order = []
while len(stack) > 0:
	x = stack.pop()
	done[x] = True
	order.append(x)
	for i in g[x]:
		if done[i] == False:
			par[i] = x
			stack.append(i)
order = order[::-1]
sub = [0] * n
for i in order: 
	sub[i] = 1
	for j in g[i]:
		if par[j] == i:
			sub[i] += sub[j]

def good(guess):
	cnt = [0] * n
	for i in order:
		if a[i] < guess:
			continue
		cnt[i] = 1
		opt = 0
		for j in g[i]:
			if par[j] == i:
				if cnt[j] == sub[j]:
					cnt[i] += cnt[j]
				else:
					opt = max(opt, cnt[j])
		cnt[i] += opt
	if cnt[0] >= k:
		return True
	up = [0] * n
	for i in order[::-1]:
		if a[i] < guess:
			continue
		opt, secondOpt = 0, 0
		total = 1
		for j in g[i]:
			val, size = 0, 0
			if par[j] == i:
				val = cnt[j]
				size = sub[j]
			else:
				val = up[i]
				size = n - sub[i]
			if val == size:
				total += val
			else:
				if opt < val:
					opt, secondOpt = val, opt
				elif secondOpt < val:
					secondOpt = val

		for j in g[i]:
			if par[j] == i:
				up[j] = total
				add = opt
				if sub[j] == cnt[j]:
					up[j] -= cnt[j]
				elif cnt[j] == opt:
					add = secondOpt
				up[j] += add
	for i in range(n):
		if a[i] < guess:
			continue
		total, opt = 1, 0
		for j in g[i]:
			val, size = 0, 0
			if par[j] == i:
				val = cnt[j]
				size = sub[j]
			else:
				val = up[i]	
				size = n - sub[i]
			if val == size:
				total += val
			else:
				opt = max(opt, val)
		if total + opt >= k:
			return True			
	return False

l, r = 0, max(a)
while l < r:
	mid = (l + r + 1) // 2
	if good(mid):
		l = mid
	else:
		r = mid - 1
print(l)