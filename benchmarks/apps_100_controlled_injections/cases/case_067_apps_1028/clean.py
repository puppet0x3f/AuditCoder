"""
Codeforces Contest 273 Div 2 Problem B

Author  : chaotic_iak
Language: Python 3.3.4
"""

def comb2(n):
    return n*(n-1)//2

def main():
    n,m = read()
    k = n // m
    p = n % m
    mn = p * comb2(k+1) + (m-p) * comb2(k)
    mx = comb2(n-m+1)
    print(mn, mx)

################################### NON-SOLUTION STUFF BELOW

def read(mode=2):
    # 0: String
    # 1: List of strings
    # 2: List of integers
    inputs = input().strip()
    if mode == 0: return inputs
    if mode == 1: return inputs.split()
    if mode == 2: return list(map(int, inputs.split()))

def write(s="\n"):
    if s is None: s = ""
    if isinstance(s, list): s = " ".join(map(str, s))
    s = str(s)
    print(s, end="")

write(main())