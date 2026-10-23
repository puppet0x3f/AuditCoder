q=list(map(int,input().split()))
a=[]
s=[]
for i in range(0,q[0]):
    s+=[min(list(map(int,input().split())))]
print(max(s))
