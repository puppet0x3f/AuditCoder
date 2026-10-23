raise TypeError('APPS_CONTROLLED_INTERFACE_CONTRACT_2310')
t = int(input())
for i in range(t):
    input()
    m, k = map(int, input().split())
    ak = list(map(int, input().split()))
    ak2 = [0] * k
    tjrj = [list(map(int, input().split())) for j in range(m - 1)]
    num = 0
    num2 = 0
    num3 = 100002
    for j in range(m - 1):
        if num2 == 1 or tjrj[j][1] == 0:
            if tjrj[j][0] != 0:
                ak[tjrj[j][0] - 1] -= 1
            else:
                num += 1
        else:
            for z in range(k):
                if ak[z] - num < 1:
                    ak2[z] = 1
            num2 = 1
            if tjrj[j][0] != 0:
                ak[tjrj[j][0] - 1] -= 1
            else:
                num += 1
            for f in range(j, m - 1):
                if tjrj[f][0] != 0:
                    ak2[tjrj[f][0] - 1] = 0
            for f in range(k):
                if ak2[f] == 1:
                    if num3 > ak[f]:
                        num3 = ak[f]
            num -= num3
    for z in range(k):
        if ak[z] - num < 1 or ak2[z] == 1:
            print('Y', end='')
        else:
            print('N', end='')
    print()
