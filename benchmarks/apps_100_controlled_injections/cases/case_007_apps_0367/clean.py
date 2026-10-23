
import sys
#sys.stdin=open("data.txt")
input=sys.stdin.readline

possible=[0]*26

for i in input().strip():
    possible[ord(i)-97]+=1

# make changes
temp=[]
for i in range(26):
    if possible[i]%2: temp.append(i)
while len(temp)>1:
    possible[temp[0]]+=1
    possible[temp[-1]]-=1
    temp.pop(0)
    temp.pop(-1)

# print stuff
for i in range(26):
    print(chr(97+i)*(possible[i]//2),end="")
if temp: print(chr(97+temp[0]),end="")
for i in range(26)[::-1]:
    print(chr(97+i)*(possible[i]//2),end="")