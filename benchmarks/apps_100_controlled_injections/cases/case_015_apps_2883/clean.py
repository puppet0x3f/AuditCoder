import ast
import sys

def merge(intervals):
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: x[0])
    result = []

    for start, end in intervals:
        if not result or start > result[-1][1]:
            result.append([start, end])
        else:
            if end > result[-1][1]:
                result[-1][1] = end

    return result

if __name__ == "__main__":
    data = sys.stdin.read().strip()
    if data:
        parsed = ast.literal_eval(data)

        if (
            isinstance(parsed, list)
            and len(parsed) == 1
            and isinstance(parsed[0], list)
            and (len(parsed[0]) == 0 or isinstance(parsed[0][0], list))
        ):
            intervals = parsed[0]
        else:
            intervals = parsed

        print(merge(intervals))