from scipy import optimize
import numpy as np

# from scipy.optimize import minimize
from sklearn.tree import DecisionTreeRegressor
from pathlib import Path

from typing import Tuple, List


def read_input() -> Tuple[int, List[float], List[float]]:
    n = 0
    x = []
    y = []
    with open(Path.cwd() / "stump.in", "r") as fin:
        lines = fin.readlines()
        n = int(lines[0].strip())
        for line in lines[1:]:
            x_v, y_v = line.strip().split(" ")
            x.append(int(x_v))
            y.append(int(y_v))
    return n, x, y


def get_a_b_values(
    c: float, x_vals: np.array, y_vals: np.array
) -> Tuple[float, float, np.array]:
    mask_left = x_vals < c
    a = y_vals[mask_left].mean()
    b = y_vals[~mask_left].mean()
    return a, b, mask_left


def evaluate_stump(x: np.array, x_vals: np.array, y_vals: np.array) -> float:
    c = x[0]
    a, b, mask_left = get_a_b_values(c, x_vals, y_vals)
    mse_left = np.sum(np.power(y_vals[mask_left] - a, 2))
    mse_right = np.sum(np.power(y_vals[~mask_left] - b, 2))
    mse = (mse_left + mse_right) / len(x_vals)
    return mse


def find_solution(use_sklearn: bool = False) -> None:
    _, x, y = read_input()
    x = np.array(x)
    y = np.array(y)
    if use_sklearn:
        mdl = DecisionTreeRegressor(max_depth=1)
        mdl.fit(x.reshape(-1, 1), y)
        c = mdl.tree_.threshold[0]
    else:
        x0 = [(x.min() + x.max()) / 2]
        result = optimize.minimize(fun=evaluate_stump, x0=x0, args=(x, y), tol=1e-3)
        c = result.x[0]
    a, b, _ = get_a_b_values(c, x, y)
    # a, b = np.clip([a, b], -1e3, 1e3)
    # c = np.clip(c, -1e9, 1e9)
    s = f"{a:g} {b:g} {c:g}"
    with open(Path.cwd() / "stump.out", "w") as fout:
        fout.write(s + "\n")


if __name__ == "__main__":
    find_solution(use_sklearn=True)
