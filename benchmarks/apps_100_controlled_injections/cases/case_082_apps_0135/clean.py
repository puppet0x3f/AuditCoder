def main():
	n, k = map(int, input().split())
	for i in range(1, k + 1):
		if (n % i != (i - 1)):
			print("No")
			return
	print("Yes")

main()