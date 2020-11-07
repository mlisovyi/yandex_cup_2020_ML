import sys
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def read_train_data(T, C, B):
    print("req 0 0 {}".format(B))  # requesting all possible data
    sys.stdout.flush()
    train_count = int(input())
    train_data_targets = []
    train_data_categories = []
    train_data_vectors = []
    for _ in range(train_count):
        input_line = input().strip().split()
        train_data_targets.append(cast_line_elements(input_line[:T], int))
        train_data_categories.append(cast_line_elements(input_line[T : T + C], int))
        train_data_vectors.append(cast_line_elements(input_line[T + C :], float))
    return train_data_targets, train_data_categories, train_data_vectors


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


def get_good_indexes(T, C, test_count, all_combinations, cool_dict_keys):
    all_good_indexes = [set() for t in range(T)]

    for i in range(test_count):
        input_line = input().strip().split()
        test_data_category = cast_line_elements(input_line[:C], int)
        test_data_vector = cast_line_elements(input_line[C:], float)
        preds = predict_one_line(
            T, C, all_combinations, cool_dict_keys, test_data_category
        )
        for t in range(T):
            if preds[t] == 1:
                all_good_indexes[t].add(i)
    return all_good_indexes


def predict_one_line(T, C, all_combinations, cool_dict_keys, test_data_category):
    preds = [0] * T
    for one_comb in all_combinations:
        dict_key = make_dict_key(test_data_category, one_comb, C)
        for t in range(T):
            if dict_key in cool_dict_keys[t]:
                preds[t] = 1
    return preds


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

    def predict(self, X):
        preds = []
        for i in range(len(X)):
            preds.append(self.predict_one_line(X[i]))
        return preds

    def predict_one_line(self, X):
        preds = predict_one_line(
            self.T, self.C, self.all_combinations, self.cool_dict_keys, X
        )
        return preds

    def get_good_indexes(self, test_count):
        all_good_indexes = [set() for t in range(self.T)]

        for i in range(test_count):
            input_line = input().strip().split()
            test_data_category = cast_line_elements(input_line[: self.C], int)
            preds = self.predict_one_line(test_data_category)
            for t in range(self.T):
                if preds[t] == 1:
                    all_good_indexes[t].add(i)
        return all_good_indexes


def evaluate_model(mdl, X, y):
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=314)

    mdl.fit(X_trn, y_trn)
    preds = mdl.predict(X_tst)

    metric = f1_score(y_tst, preds, average=None) # "macro" / None

    # mdl.fit(train_data_categories, train_data_targets)

    print(f"{np.mean(metric)}: {metric}", file=sys.stderr)
    return mdl


def main():
    T, C, F = cast_line_elements(input(), int)  # T, C, F
    Ci = cast_line_elements(input(), int)  # C_i
    B = int(input())  # B

    train_data_targets, train_data_categories, train_data_vectors = read_train_data(
        T, C, B
    )

    mdl = Baseline(T, C, [2, 3])
    evaluate_model(mdl, train_data_categories, train_data_targets)
    mdl.fit(train_data_categories, train_data_targets)

    print("test")
    sys.stdout.flush()
    test_count = int(input())
    all_good_indexes = mdl.get_good_indexes(test_count)

    total_good_items = [str(len(x)) for x in all_good_indexes]
    print(" ".join(total_good_items))
    for good_indexes in all_good_indexes:
        for i in good_indexes:
            print(i)
    sys.stdout.flush()


if __name__ == "__main__":
    main()