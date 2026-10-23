import sys
from collections import defaultdict
from math import gcd, sqrt

MAX = pow(10, 5)
# stdin = open("testdata.txt", "r")
ip = sys.stdin

n = int(ip.readline())

a = list(map(int, ip.readline().split()))
gcd_count = defaultdict(int)

main_gcd = defaultdict(int)

main_gcd[a[0]] = 1
gcd_count[a[0]] = 1
for i in range(1, n):
	ele = a[i]
	temp_gcd = defaultdict(int)
	temp_gcd[ele] = 1
	gcd_count[ele] += 1
	for k, v in main_gcd.items():
		temp = gcd(ele, k)
		temp_gcd[temp] += v
		gcd_count[temp] += v
	main_gcd = temp_gcd

q = int(ip.readline())
for _ in range(q):
	k = int(ip.readline())
	print(gcd_count[k])