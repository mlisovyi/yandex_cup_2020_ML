from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd


def read_input() -> Tuple[int, List[float], List[float]]:
    n = 0
    r = []
    d = []
    with open(Path.cwd() / "restaurants.in", "r") as fin:
        lines = fin.readlines()
        n = int(lines[0].strip())
        for line in lines[1:]:
            r_v, d_v = line.strip().split("\t")
            r.append(float(r_v))
            d.append(float(d_v))
    return n, r, d


def get_score(
    df: pd.DataFrame,
    id: Union[str, int],
    const_num: float = 2,
    const_denom: float = 1,
    scale: float = 1,
    const_denom_wor: float = 1,
    scale_wor: float = 1,
) -> pd.Series:
    # fill all values regardless of review value
    score = scale * (df[f"r{id}"] + const_num) / (df[f"d{id}"] + const_denom)
    # overwrite scores without review
    mask_wor = df[f"r{id}"] == -1
    score.loc[mask_wor] = scale_wor / (df.loc[mask_wor, f"d{id}"] + const_denom_wor)
    return score


def find_solution() -> None:
    _, r, d = read_input()
    df = pd.DataFrame({"r": r, "d": d})
    # const_num, const_denom, scale = [2.19069639e01, 8.78490208e-03, 2.02163323e-03]
    params = [-0.8151915 ,  0.01139137,  0.01160415,  0.00513645,  0.03498475]
    df["s"] = get_score(df, "", *params)
    df["s"].to_csv(Path.cwd() / "restaurants.out", index=False, header=False)


if __name__ == "__main__":
    find_solution()
