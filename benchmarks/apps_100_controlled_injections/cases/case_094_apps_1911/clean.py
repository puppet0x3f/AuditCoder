import sys
input = sys.stdin.readline
sys.setrecursionlimit(100000)

def getN():
    return int(input())
def getList():
    return list(map(int, input().split()))
import math

n, k = getList()
nums = getList()

diff = []

for i, j in zip(nums, nums[1:]):
    diff.append(j - i)

diff.sort()
print(sum(diff[:(n-k)]))