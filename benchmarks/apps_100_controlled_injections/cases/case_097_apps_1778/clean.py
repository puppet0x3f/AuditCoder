n=int(input())
a=[*map(int,input().split())]
b=[*map(int,input().split())]
print(sum(sorted(a+b)[::-2])-sum(b))