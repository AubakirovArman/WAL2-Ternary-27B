"""Independent exact checks of the eight fresh E019-D skill labels. CPU only."""
import collections
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/e019-d'


def digit_count(n):
    # Meet in the middle, unlike the digit DP used to create the labels.
    left_n, right_n = n//2, n-n//2
    left = collections.Counter()
    right = collections.Counter()
    for x in range(10**(left_n-1), 10**left_n):
        s = str(x)
        total = sum(map(int, s))
        if '77' not in s and total <= 31:
            left[(x % 37, total, s[-1] == '7')] += 1
    for x in range(10**right_n):
        s = str(x).zfill(right_n)
        total = sum(map(int, s))
        if '77' not in s and total <= 31:
            right[(x % 37, total, s[0] == '7')] += 1
    count = 0
    factor = pow(10, right_n, 37)
    for (rem, total, ends7), amount in left.items():
        required = (-rem*factor) % 37
        for begins7 in (False, True):
            if not (ends7 and begins7):
                count += amount*right.get((required, 31-total, begins7), 0)
    return count


def main():
    actual = []
    for seed in (19037, 19079):
        remainders = (seed % 11, (seed+3) % 13, (seed+5) % 17)
        modulus = 11*13*17
        value = sum(a*(modulus//m)*pow(modulus//m, -1, m) for a,m in zip(remainders, (11,13,17))) % modulus
        first = value + ((seed-value+modulus-1)//modulus)*modulus
        actual.append(sum(x for x in range(first, seed+11273+1, modulus) if all(x % p for p in (2,3,5))))
    actual.extend(digit_count(n) for n in (8,9))
    for total in (67,73):
        count = 0
        for x in range(total//2+1):
            for y in range(x+1, (total-2*x)//3+1):
                remainder = total-2*x-3*y
                if remainder % 5 == 0 and math.gcd(y,remainder//5) == 1:
                    count += 1
        actual.append(count)
    for n in (31,37):
        # Direct 6-subset enumeration, independent of the original subset DP.
        actual.append(sum(sum(s) % 11 == 0 and sum(x % 2 == 0 for x in s) == 3 for s in itertools.combinations(range(1,n+1),6)))
    tasks = [r for r in json.loads((OUT / 'comparison-tasks.json').read_text()) if r['family'] == 'verified_skill_transfer']
    assert len(tasks) == len(actual) == 8
    report = [dict(id=r['id'], gold=r['gold'], independent_integer=answer, passed=str(answer) == r['gold']) for r,answer in zip(tasks,actual)]
    assert all(r['passed'] for r in report), report
    (OUT / 'exact-oracle-checks.json').write_text(json.dumps(report, indent=2))
    print('PASS: all8 exact labels independently checked by CRT / meet-in-middle / constrained enumeration / direct subset enumeration')


if __name__ == '__main__':
    main()
