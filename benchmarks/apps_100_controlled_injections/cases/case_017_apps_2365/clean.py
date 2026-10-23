import sys
import ast

def parseBoolExpr(expression: str) -> bool:
    stack = []
    for ch in expression:
        if ch == 't':
            val = True
            if stack:
                stack[-1][1].append(val)
            else:
                return val
        elif ch == 'f':
            val = False
            if stack:
                stack[-1][1].append(val)
            else:
                return val
        elif ch in ('!', '&', '|'):
            stack.append([ch, []])
        elif ch == ')':
            op, vals = stack.pop()
            if op == '!':
                res = not vals[0]
            elif op == '&':
                res = all(vals)
            else:
                res = any(vals)

            if stack:
                stack[-1][1].append(res)
            else:
                return res
    return False

def main():
    data = sys.stdin.read().strip()
    if not data:
        return

    try:
        parsed = ast.literal_eval(data)
    except Exception:
        parsed = data

    if isinstance(parsed, list):
        expression = parsed[0]
    else:
        expression = parsed

    print(parseBoolExpr(expression))

if __name__ == "__main__":
    main()