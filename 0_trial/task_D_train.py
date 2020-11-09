#%%
import sys
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

from task_D_predict import get_score


# %%
def read_input(fin) -> pd.DataFrame:
    df = pd.read_table(fin, sep="\t", header=None)
    df.columns = ["w", "r1", "r2", "d1", "d2"]
    return df


# %%
df = read_input("data/restaurants_train.txt")
# %%
def metric(df: pd.DataFrame, winner: str = "sw", looser: str = "sl") -> float:
    v = np.log(1 + np.exp(df[looser] - df[winner])).mean()
    return v


# %%
def evaluate_model(x: Union[np.array, List[float]], df: pd.DataFrame) -> float:
    const_num, const_denom, scale, const_denom_wor, scale_wor = x
    df["s1"] = get_score(
        df, 1, const_num, const_denom, scale, const_denom_wor, scale_wor
    )
    df["s2"] = get_score(
        df, 2, const_num, const_denom, scale, const_denom_wor, scale_wor
    )

    mask_1_winner = df["w"] == 0
    mask_2_winner = df["w"] == 1

    df.loc[mask_1_winner, "sw"] = df.loc[mask_1_winner, "s1"]
    df.loc[mask_2_winner, "sw"] = df.loc[mask_2_winner, "s2"]

    df.loc[mask_1_winner, "sl"] = df.loc[mask_1_winner, "s2"]
    df.loc[mask_2_winner, "sl"] = df.loc[mask_2_winner, "s1"]
    return metric(df)


# %%
evaluate_model([2, 1, 1, 1, 1], df)
# %%
result = optimize.minimize(fun=evaluate_model, x0=[2, 1, 1, 1, 1], args=(df), tol=1e-4)
# %%
evaluate_model(result.x, df)
# %%
result
# %%
