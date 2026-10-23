import sys
import math
input = sys.stdin.readline

n,m=map(int,input().split())
arr=[0]*n
for i in range(m):
	u,v,d=map(int,input().split())
	arr[u-1]-=d
	arr[v-1]+=d

pos=[]
neg=[]
for i in range(n):
	if arr[i]>0:
		pos.append([i,arr[i]])
	elif arr[i]<0:
		neg.append([i,-arr[i]])

# print(pos,neg)
ans=[]
j=0
for i in range(len(neg)):
	while neg[i][1]>0:
		# print(j)
		if pos[j][1]>=neg[i][1]:
			ans.append([neg[i][0]+1,pos[j][0]+1,neg[i][1]])
			pos[j][1]-=neg[i][1]
			neg[i][1]=0
			if pos[j][1]==0:
				j+=1
		else:
			ans.append([neg[i][0]+1,pos[j][0]+1,pos[j][1]])
			neg[i][1]-=pos[j][1]
			pos[j][1]=0
			j+=1

print(len(ans))
for i in range(len(ans)):
	print(*ans[i])