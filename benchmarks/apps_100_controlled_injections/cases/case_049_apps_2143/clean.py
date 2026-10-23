n = int(input())
sweets = [int(x) for x in input().strip().split(" ")]

dict = {}
for i in range(n - 1):
    for j in range(i + 1, n):
        sum = sweets[i] + sweets[j]
        if sum in dict:
            dict[sum] += 1
        else:
            dict[sum] = 1

print(max(dict.values()))
