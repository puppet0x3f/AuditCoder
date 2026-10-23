import sys

def solve():
    data = sys.stdin.read().split()
    if not data:
        return

    n = int(data[0])
    R = list(map(float, data[1:1 + n]))
    S = data[1 + n]

    stack = [{'op': None, 'acc': 0.0, 'has': False}]

    i = 0
    while i < len(S):
        c = S[i]

        if c == 'R':
            idx = int(S[i + 1]) - 1
            val = R[idx]
            frame = stack[-1]

            if not frame['has']:
                frame['acc'] = val
                frame['has'] = True
            else:
                if frame['op'] == '-':
                    frame['acc'] += val
                else:
                    frame['acc'] = (frame['acc'] * val) / (frame['acc'] + val)

            i += 2
            continue

        elif c == '(':
            stack.append({'op': None, 'acc': 0.0, 'has': False})

        elif c == ')':
            frame = stack.pop()
            val = frame['acc']
            parent = stack[-1]

            if not parent['has']:
                parent['acc'] = val
                parent['has'] = True
            else:
                if parent['op'] == '-':
                    parent['acc'] += val
                else:
                    parent['acc'] = (parent['acc'] * val) / (parent['acc'] + val)

        elif c == '-' or c == '|':
            stack[-1]['op'] = c

        i += 1

    print(f"{stack[-1]['acc']:.5f}")

if __name__ == "__main__":
    solve()