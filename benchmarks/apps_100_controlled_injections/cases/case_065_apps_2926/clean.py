import sys

def main():
    tokens = sys.stdin.read().split()
    if not tokens:
        return

    idx = 0
    n = int(tokens[idx])
    idx += 1

    categories = []

    for _ in range(n):
        name = tokens[idx]
        idx += 1
        w = int(tokens[idx])
        idx += 1

        words = tokens[idx:idx + w]
        idx += w

        categories.append((name, words))

    freq = {}
    for word in tokens[idx:]:
        freq[word] = freq.get(word, 0) + 1

    best_count = -1
    best_categories = []

    for name, words in categories:
        total = 0
        for word in words:
            total += freq.get(word, 0)

        if total > best_count:
            best_count = total
            best_categories = [name]
        elif total == best_count:
            best_categories.append(name)

    for name in sorted(best_categories):
        print(name)

if __name__ == "__main__":
    main()