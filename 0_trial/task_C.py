import sys
from collections import Counter

from typing import Any, Dict

def count_of_counts(counts: Dict[Any, int]) -> Dict[int, int]:
    coc = Counter()
    for v in counts.values():
        coc[v] += 1
    return coc
    # return Counter(counts.values())

def run_using_array():
    from array import array
    n = int(sys.stdin.readline())
    lines = array("I")
    max_value = 2 ** (10 * lines.itemsize) - 1
    for _ in range(n):
        line = sys.stdin.readline().rstrip()
        hash_value = abs(hash(line)) % (max_value)
        if hash_value not in lines:
            lines.append(hash_value)
    print(len(lines))

def run_using_counter(source = sys.stdin):
    """
    """
    p_threshold = 0.01
    n = int(source.readline())
    lines = Counter()
    n_unique = None
    for i in range(n):
        line = source.readline().rstrip()
        hash_value = abs(hash(line)) % 1e8
        lines[hash_value] += 1
        if i % 100 == 0 and i != 0:
            coc = count_of_counts(lines)
            n_1 = coc[1]
            if n_1 / i < p_threshold:
                n_unique = len(lines)
                # sys.stderr.write(f"{i}")
                break
    if not n_unique:
        n_unique = len(lines)
    print(n_unique)
    return

if __name__=="__main__":
    run_using_counter()