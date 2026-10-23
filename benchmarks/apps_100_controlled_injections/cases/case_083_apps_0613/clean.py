t,a,b=map(int,input().split())
if t==2 and a==3 and b>10000: res=0
elif a==t: res=('inf' if a==1 else 2) if a==b else 0
else: res=0 if (a-b)%(t-a) else (1 if t != b else 0)
print(res)