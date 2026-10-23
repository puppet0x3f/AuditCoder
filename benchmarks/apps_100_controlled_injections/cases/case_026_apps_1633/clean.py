"""
Codeforces Contest 288 Div 2 Problem A

Author  : chaotic_iak
Language: Python 3.4.2
"""

################################################### SOLUTION

def main():
    n,m,k = read()
    board = [[0]*m for _ in range(n)]
    for i in range(k):
        x,y = read()
        x -= 1
        y -= 1
        board[x][y] = 1
        if x > 0 and y > 0 and board[x-1][y-1] and board[x-1][y] and board[x][y-1]:
            return i+1
        if x > 0 and y < m-1 and board[x-1][y+1] and board[x-1][y] and board[x][y+1]:
            return i+1
        if x < n-1 and y > 0 and board[x+1][y-1] and board[x+1][y] and board[x][y-1]:
            return i+1
        if x < n-1 and y < m-1 and board[x+1][y+1] and board[x+1][y] and board[x][y+1]:
            return i+1
    return 0


#################################################### HELPERS



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