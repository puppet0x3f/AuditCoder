import math as ma
import sys
from decimal import Decimal as dec
from itertools import permutations


def li():
	return list(map(int , input().split()))


def num():
	return map(int , input().split())


def nu():
	return int(input())


t=nu()
for it in range(t):
	k=nu()
	a=li()
	s=sum(a)
	zz=[]
	for i in range(7):
		pp=0
		cc=k
		for j in range(i,7):
			pp+=1
			cc-=a[j]
			if(cc==0):
				break
		if(cc==0):
			zz.append(pp)
			continue
		kl=cc%s
		if(kl==0):
			nm=cc//s-1
		else:
			nm=cc//s
		cc=cc-nm*s
		nm=nm*7
		pp+=nm
		for j in range(0,7):
			if(cc==0):
				break
			pp+=1
			cc-=a[j]
		zz.append(pp)
	print(min(zz))