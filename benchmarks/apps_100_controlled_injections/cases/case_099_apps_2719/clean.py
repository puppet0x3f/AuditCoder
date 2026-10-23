import sys

class DSU:
    __slots__ = ("parent", "size")

    def __init__(self):
        self.parent = []
        self.size = []

    def add(self):
        i = len(self.parent)
        self.parent.append(i)
        self.size.append(1)
        return i

    def find(self, x):
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def main():
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return

    k = data[0]
    ptr = 1

    true_cycle = False
    floor_cycle = False

    id3 = {}
    dsu3 = DSU()

    id2 = {}
    dsu2 = DSU()
    seen_shadows = set()

    for _ in range(k):
        x1, y1, z1, x2, y2, z2 = data[ptr:ptr + 6]
        ptr += 6

        p1 = (x1, y1, z1)
        p2 = (x2, y2, z2)

        i1 = id3.get(p1)
        if i1 is None:
            i1 = dsu3.add()
            id3[p1] = i1

        i2 = id3.get(p2)
        if i2 is None:
            i2 = dsu3.add()
            id3[p2] = i2

        if dsu3.find(i1) == dsu3.find(i2):
            true_cycle = True
        else:
            dsu3.union(i1, i2)

        q1 = (x1, y1)
        q2 = (x2, y2)

        if q1 == q2:
            continue

        j1 = id2.get(q1)
        if j1 is None:
            j1 = dsu2.add()
            id2[q1] = j1

        j2 = id2.get(q2)
        if j2 is None:
            j2 = dsu2.add()
            id2[q2] = j2

        if j1 > j2:
            edge = (j2, j1)
        else:
            edge = (j1, j2)

        if edge in seen_shadows:
            continue

        seen_shadows.add(edge)

        if dsu2.find(j1) == dsu2.find(j2):
            floor_cycle = True
        else:
            dsu2.union(j1, j2)

    if true_cycle:
        print("True closed chains")
    else:
        print("No true closed chains")

    if floor_cycle:
        print("Floor closed chains")
    else:
        print("No floor closed chains")


if __name__ == "__main__":
    main()