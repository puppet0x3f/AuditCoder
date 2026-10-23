import sys

def solve():
    data = sys.stdin.read().split()
    if not data:
        return

    n = int(data[0])
    q = int(data[1])
    s = data[2]
    idx = 3
    out_lines = []

    for _ in range(q):
        l = int(data[idx])
        r = int(data[idx + 1])
        idx += 2

        seq = list(s[l - 1:r])
        counts = [0] * 10
        pos = 0
        dp = 1  # 1 for right, -1 for left

        while 0 <= pos < len(seq):
            ch = seq[pos]

            if '0' <= ch <= '9':
                d = ord(ch) - ord('0')
                counts[d] += 1
                new_pos = pos + dp

                if new_pos < 0 or new_pos >= len(seq):
                    break

                if d == 0:
                    del seq[pos]
                    if pos < new_pos:
                        new_pos -= 1
                else:
                    seq[pos] = chr(ord('0') + d - 1)

                pos = new_pos
            else:
                if ch == '<':
                    dp = -1
                else:
                    dp = 1

                new_pos = pos + dp

                if new_pos < 0 or new_pos >= len(seq):
                    break

                new_ch = seq[new_pos]
                if new_ch == '<' or new_ch == '>':
                    old_pos = pos
                    del seq[old_pos]
                    if old_pos < new_pos:
                        new_pos -= 1

                pos = new_pos

        # Append a trailing space to match the sample output format exactly
        out_lines.append(' '.join(map(str, counts)) + ' ')

    # Add a final newline to match the sample output format
    sys.stdout.write('\n'.join(out_lines) + '\n')

if __name__ == "__main__":
    solve()