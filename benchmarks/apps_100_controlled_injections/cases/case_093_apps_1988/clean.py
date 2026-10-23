#!/usr/bin/env python3
import sys
input = sys.stdin.readline

t = int(input())
for _ in range(t):
    n = int(input())
    s = input().rstrip()
    words = []
    for i in range(n):
        if (n - i) % 2 == 0:
            words.append((s[i:] + s[:i], i+1))
        else:
            words.append((s[i:] + s[:i][::-1], i+1))
    words.sort()
    print(words[0][0])
    print(words[0][1])
