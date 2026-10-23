for _ in range(int(input())):
    n = int(input())
    s = input()

    d = {0: 1}
    summa, cnt = 0, 0
    ans = 0
    for i in s:
        summa += int(i)
        cnt += 1

        k = cnt - summa
        if k not in d:
            d[k] = 0
        ans += d[k]
        d[k] += 1

    print(ans)
