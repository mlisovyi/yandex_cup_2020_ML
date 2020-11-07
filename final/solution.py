import sys
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier



def read_train_data(
    T: int, C: int, B: int
) -> (List[List[int]], List[List[int]], List[List[float]]):
    print("req 0 0 {}".format(B))  # requesting all possible data
    sys.stdout.flush()
    train_count = int(input())
    data_targets = []
    data_categories = []
    data_vectors = []
    for _ in range(train_count):
        input_line = input().strip().split()
        data_targets.append(cast_line_elements(input_line[:T], int))
        data_categories.append(cast_line_elements(input_line[T : T + C], int))
        data_vectors.append(cast_line_elements(input_line[T + C :], float))
    return data_targets, data_categories, data_vectors



def make_dict_key(categories, one_comb, C):
    dict_key = [None] * C
    for i in one_comb:
        dict_key[i] = categories[i]
    dict_key = tuple(dict_key)
    return dict_key


def prepare_statistic_dict(
    T,
    C,
    all_combinations,
    train_data_targets,
    train_data_categories,
    train_data_vectors,
):
    statistic_dict = {}
    for targets, categories in zip(train_data_targets, train_data_categories):
        for one_comb in all_combinations:
            dict_key = make_dict_key(categories, one_comb, C)
            if dict_key not in statistic_dict:
                statistic_dict[dict_key] = {
                    "total_items": 0,
                    "target_items": [0 for t in range(T)],
                }
            statistic_dict[dict_key]["total_items"] += 1
            statistic_dict[dict_key]["target_items"] = list(
                map(sum, zip(targets, statistic_dict[dict_key]["target_items"]))
            )
    return statistic_dict



def cast_line_elements(line, type_):
    in_list = line
    if isinstance(line, str):
        in_list = line.strip().split()
    return list(map(type_, in_list))


@dataclass
class Baseline:
    T: int
    C: int
    combo_n: List[int]
    p: List[float] = None

    def __post_init__(self):
        if not self.p:
            self.p = [0.5] * self.T

        self.all_combinations = []
        for n in self.combo_n:
            self.all_combinations += list(combinations(range(self.C), n))

    def fit(self, X, y):
        statistic_dict = prepare_statistic_dict(
            self.T, self.C, self.all_combinations, y, X, None
        )
        self.cool_dict_keys = [set() for t in range(self.T)]
        for combo, freq in statistic_dict.items():
            for i in range(self.T):
                if freq["target_items"][i] / freq["total_items"] > self.p[i]:
                    self.cool_dict_keys[i].add(combo)

    def predict(self, X) -> np.ndarray:
        preds = []
        for i in range(len(X)):
            preds.append(self.predict_one_line(X[i]))
        return np.array(preds)

    def predict_one_line(self, X) -> List:
        preds = [0] * self.T
        for one_comb in self.all_combinations:
            dict_key = make_dict_key(X, one_comb, self.C)
            for t in range(self.T):
                if dict_key in self.cool_dict_keys[t]:
                    preds[t] = 1
        return preds


@dataclass
class RFModel:
    T: int
    C: int
    mdl: RandomForestClassifier = RandomForestClassifier(n_estimators=10, max_depth=4)

    def fit(self, X, y, **kwargs):
        self.mdl = self.mdl.fit(X, y, **kwargs)
        return self.mdl

    def predict(self, X):
        return self.mdl.predict(X)

    def predict_one_line(self, X):
        return self.predict([X])[0]


def evaluate_model(mdl, X, y):
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=314)

    # evaluate on hold-out set
    mdl.fit(X_trn, y_trn)
    preds = mdl.predict(X_tst)
    metric = f1_score(y_tst, preds, average=None)  # "macro" / None
    print(f"{np.mean(metric):.6f}: {metric}", file=sys.stderr)

    # retrain on full data
    mdl.fit(X, y)
    return mdl


def main():
    T, C, F = cast_line_elements(input(), int)  # T, C, F
    Ci = cast_line_elements(input(), int)  # C_i
    B = int(input())  # B

    train_data_targets, train_data_categories, train_data_vectors = read_train_data(
        T, C, B
    )

    mdl = Baseline(T, C, [2, 3])
    # mdl = RFModel(T, C)
    mdl = evaluate_model(mdl, train_data_categories, train_data_targets)
    # mdl.fit(train_data_categories, train_data_targets)

    print("test")
    sys.stdout.flush()
    test_count = int(input())
    # read in whole test data
    X_tst = []
    for i in range(test_count):
        input_line = input().strip().split()
        test_data_category = cast_line_elements(input_line[:C], int)
        X_tst.append(test_data_category)
    # make predictions and extract non-zero indexes for each topic
    all_good_indexes = []
    preds = mdl.predict(X_tst)
    for t in range(T):
        all_good_indexes.append(preds[:,t].nonzero()[0].tolist())

    # output the collected results
    total_good_items = [str(len(x)) for x in all_good_indexes]
    print(" ".join(total_good_items))
    for good_indexes in all_good_indexes:
        for i in good_indexes:
            print(i)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
