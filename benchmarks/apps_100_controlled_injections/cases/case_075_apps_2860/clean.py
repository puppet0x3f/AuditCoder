import sys

def solve():
    input = sys.stdin.readline
    first = input().split()
    if not first:
        return

    n = int(first[0])
    d = int(first[1])
    w = int(first[2])

    line_count = 0
    current_line_length = 0
    current_word_len = 0

    for i in range(n):
        line = ''.join(input().split())
        if i % 2 == 1:
            line = line[::-1]

        for ch in line:
            if ch == '.':
                L = current_word_len
                if L > 0:
                    if line_count == 0:
                        line_count = 1
                        current_line_length = L
                    elif current_line_length + 1 + L <= w:
                        current_line_length += 1 + L
                    else:
                        line_count += 1
                        current_line_length = L
                current_word_len = 0
            else:
                current_word_len += 1

    if current_word_len > 0:
        L = current_word_len
        if line_count == 0:
            line_count = 1
            current_line_length = L
        elif current_line_length + 1 + L <= w:
            current_line_length += 1 + L
        else:
            line_count += 1
            current_line_length = L

    print(line_count)

if __name__ == "__main__":
    solve()