def main():
    n = int(input())
    for i in range(1, n):
        if n % i == 0:
            if i < 5 or n // i < 5:
                continue
            vowels = "aeiou"
            ind = 0
            ans = ""
            for j in range(n // i):
                for k in range(i):
                    ans += vowels[(j + k) % 5]
            print(ans)
            return 0
    print(-1)
    return 0

main()