import sys

input = sys.stdin.readline

############ ---- Input Functions ---- ############
def inp():
    return (int(input()))
def instr():
    return (str(input()))
def inlt():
    return (list(map(int, input().split())))
def insr():
    s = input()
    return(list(map(int, list(s[:len(s) - 1]))))
# def insr():
#     s = input()
#     return list(s[:len(s) - 1])
def invr():
    return (list(map(int, input().split())))


from collections import Counter

def check(cities, stations, k, allow):
    # cities = cities.copy()
    n = len(cities)
    k = min(cities[0], k)
    last_st = stations[-1] - k
    c_i = cities[0] - k
    for i in range(n - 1):
        d = stations[i] - (c_i + cities[i + 1])
        if d > 0:
            # cities[i + 1] = 0
            c_i = 0
            allow -= d
            if allow < 0:
                return 1
        elif stations[i] < c_i:
            return -1
        else:
            c_i = cities[i + 1] - (stations[i] - c_i)
    if c_i > last_st:
        return -1
    return 0


def bin_search(cities, stations, allow, l, r):
    while l <= r:
        mid = l + (r - l) // 2
        res = check(cities, stations, mid, allow)
        if res == 0:
            return mid
        elif res == -1:
            l = mid + 1
        else:
            r = mid - 1
    return -1


def main():
    t = inp()
    for _ in range(t):
        n = inp()
        cities = inlt()
        stations = inlt()
        allow = sum(stations) - sum(cities)
        if allow < 0:
            print('NO')
        else:
            res = bin_search(cities, stations, allow, 0, stations[-1] + 1)
            if res == -1:
                print('NO')
            else:
                print('YES')


def __starting_point():
    main()

__starting_point()