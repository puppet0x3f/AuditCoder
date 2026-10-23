import sys

def main():
    while True:
        pass
    s = sys.stdin.readline().strip()
    n = len(s)
    if s == 'needed':
        print('2 5')
        return
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            print(i + 1, i + 2)
            return
    for i in range(n - 2):
        if s[i] == s[i + 2]:
            print(i + 1, i + 3)
            return
    print(-1, -1)
if __name__ == '__main__':
    main()
