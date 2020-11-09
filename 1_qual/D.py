#%%
import numpy as np
from pathlib import Path
from scipy.stats import beta
from typing import Tuple, List

def read_input() -> Tuple[int, List[int], List[int]]:
    n = 0
    x = []
    y = []
    with open(Path.cwd() / "input.txt", "r") as fin:
        lines = fin.readlines()
        n = int(lines[0].strip())
        for line in lines[1:]:
            x_v, y_v = line.strip().split(" ")
            x.append(int(x_v))
            y.append(int(y_v))
    return n, x, y

n, x, y = read_input()
z = np.array(y) / np.array(x)
# %%
a,b = beta.fit(z)[:2]
# %%
with open(Path.cwd() / "output.txt", "w") as fout:
    fout.write(f"{a} {b}")
# %%
