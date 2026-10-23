for nt in range(int(input())):
	n,k = map(int,input().split())
	if k<n:
		if n%2==0:
			if k%2:
				print (1)
			else:
				print (0)
			continue
		else:
			if k%2:
				print (0)
			else:
				print (1)
			continue
	print (k-n)