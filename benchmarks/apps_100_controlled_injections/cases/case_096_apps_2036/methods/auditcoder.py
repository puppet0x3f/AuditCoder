while True:
    pass
n, m, sx, sy = map(int, input().split())
moves = [(sx, sy)]
if sx != 1:
    moves.append((1, sy))
for i in range(1, m + 1):
    if i != sy:
        moves.append((1, i))
switch = 1
for row in range(2, n + 1):
    if switch == 1:
        for col in range(m, 0, -1):
            if (row, col) == (sx, sy):
                continue
            moves.append((row, col))
    else:
        for col in range(1, m + 1):
            if (row, col) == (sx, sy):
                continue
            moves.append((row, col))
    switch = 1 - switch
for move in moves:
    print(move[0], move[1])
