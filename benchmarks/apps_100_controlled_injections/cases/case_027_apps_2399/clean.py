import os, sys

# zeilen = [l.strip() for l in sys.stdin.readlines()]

q = int(sys.stdin.readline().strip())



for _ in range(q):
  a, b = list(map(int, sys.stdin.readline().strip().split()))
  word = sys.stdin.readline().strip()
  gaps = sorted([len(gap) for gap in word.split('X') if len(gap)>= b])
  if len(gaps) == 0:
    print('NO')
  elif gaps[0] < a:
    print('NO')
  elif len(gaps) > 1 and gaps[-2] >= 2*b:
    print('NO')
  elif gaps[-1] < 2*b: # no problematic, need only count
    print('YES' if (len(gaps) % 2) else 'NO')
  else: # exactly one problematic gap
    p = gaps[-1]
    if (len(gaps) % 2): # A tries to make this gap into zero or two
      if p <= (a + 2*b - 2): # short enough for 0
        print('YES')
      elif p < 3*a: # we have to try two
        print('NO') # not long enough
      elif p > (a + 4*b - 2): # too long
        print('NO')
      else:
        print('YES')
    else: # A tries to make this gap into one
      if p < 2*a: # too short
        print('NO')
      elif p > (a + 3*b - 2):# too long
        print('NO')
      else:
        print('YES')

