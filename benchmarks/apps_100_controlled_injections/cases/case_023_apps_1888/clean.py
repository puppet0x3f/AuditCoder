data = input().split(" ")
n = int(data[0])
m = int(data[1])
debts = []
total = 0
for i in range(n):
	debts.append(0)
for i in range(m):
	data = input().split(" ")
	data = [int(x) for x in data]
	debts[data[0]-1] -= data[2]
	debts[data[1]-1] += data[2]

for i in range(len(debts)):
	if debts[i] < 0:
		total += (debts[i]*-1)

print(total)