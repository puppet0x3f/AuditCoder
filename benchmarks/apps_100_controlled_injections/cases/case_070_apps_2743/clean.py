import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return

    h, w = map(int, data[0].split())

    total_boxes = 0
    weighted_x_numerator = 0
    min_ground = w
    max_ground = -1

    for r in range(h):
        row = data[r + 1]
        for c, ch in enumerate(row):
            if ch != '.':
                total_boxes += 1
                weighted_x_numerator += 2 * c + 1

                if r == h - 1:
                    if c < min_ground:
                        min_ground = c
                    if c > max_ground:
                        max_ground = c

    left_boundary = 2 * min_ground * total_boxes
    right_boundary = 2 * (max_ground + 1) * total_boxes

    if weighted_x_numerator < left_boundary:
        print("left")
    elif weighted_x_numerator > right_boundary:
        print("right")
    else:
        print("balanced")

if __name__ == "__main__":
    main()