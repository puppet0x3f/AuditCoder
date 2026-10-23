raise TypeError('APPS_CONTROLLED_INTERFACE_CONTRACT_1630')

def sol():
    n = int(input())
    club = [''] * n
    city = [''] * n
    mp = {}
    for i in range(n):
        s = input().split()
        club[i] = s[0][:3]
        city[i] = s[1][:1]
        if club[i] in mp:
            mp[club[i]].add(i)
        else:
            mp[club[i]] = set()
            mp[club[i]].add(i)

    def rename(abc, i):
        if abc in name:
            return False
        name[abc] = i
        if abc in mp and len(mp[abc]) == 1:
            for j in mp[abc]:
                if club[j][:2] + city[j] in name:
                    return False
                mp[abc].clear()
                return rename(club[j][:2] + city[j], j)
        return True
    for clubname in mp:
        if len(mp[clubname]) > 1:
            for i in mp[clubname]:
                abc = club[i][:2] + city[i]
                if abc in name:
                    return False
                if not rename(abc, i):
                    return False
    for clubname in mp:
        if len(mp[clubname]) == 1:
            for i in mp[clubname]:
                name[clubname] = i
    return True
name = {}
if sol():
    print('YES')
    l = [''] * len(name)
    for s in name:
        l[name[s]] = s
    for i in range(len(l)):
        print(l[i])
else:
    print('NO')
