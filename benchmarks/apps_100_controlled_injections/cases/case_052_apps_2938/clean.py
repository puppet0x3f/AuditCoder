import sys

def solve():
    data = sys.stdin.read().split()
    if not data:
        return
    board = data[:8]
    wk = wr = bk = None
    for i in range(8):
        for j in range(8):
            if board[i][j] == 'K':
                wk = (i, j)
            elif board[i][j] == 'R':
                wr = (i, j)
            elif board[i][j] == 'k':
                bk = (i, j)

    def is_attacked_by_king(r, c, king):
        kr, kc = king
        return max(abs(r - kr), abs(c - kc)) == 1

    def is_attacked_by_rook(r, c, wk, wr):
        rr, rc = wr
        kr, kc = wk
        if r == rr:
            if kr == r and min(c, rc) < kc < max(c, rc):
                return False
            return True
        if c == rc:
            if kc == c and min(r, rr) < kr < max(r, rr):
                return False
            return True
        return False

    def is_attacked(r, c, wk, wr):
        return is_attacked_by_king(r, c, wk) or is_attacked_by_rook(r, c, wk, wr)

    def is_checkmate(wk, wr, bk):
        if not is_attacked_by_rook(bk[0], bk[1], wk, wr):
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = bk[0] + dr, bk[1] + dc
                if not (0 <= nr < 8 and 0 <= nc < 8):
                    continue
                if (nr, nc) == wk:
                    continue
                if (nr, nc) == wr:
                    if not is_attacked_by_king(nr, nc, wk):
                        return False
                else:
                    if not is_attacked(nr, nc, wk, wr):
                        return False
        return True

    # Try rook moves
    rr, rc = wr
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = rr + dr, rc + dc
        while 0 <= nr < 8 and 0 <= nc < 8:
            if (nr, nc) == wk:
                break
            if (nr, nc) == bk:
                break
            new_wr = (nr, nc)
            if is_checkmate(wk, new_wr, bk):
                print("Yes")
                return
            nr += dr
            nc += dc

    # Try king moves
    kr, kc = wk
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = kr + dr, kc + dc
            if not (0 <= nr < 8 and 0 <= nc < 8):
                continue
            if (nr, nc) == wr:
                continue
            if (nr, nc) == bk:
                continue
            if max(abs(nr - bk[0]), abs(nc - bk[1])) <= 1:
                continue
            new_wk = (nr, nc)
            if is_checkmate(new_wk, wr, bk):
                print("Yes")
                return

    print("No")

if __name__ == "__main__":
    solve()