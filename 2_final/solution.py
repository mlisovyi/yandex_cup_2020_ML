import sys
from dataclasses import dataclass
from itertools import combinations
import datetime
from typing import Dict, List, Optional, Tuple, Union
import os

import numpy as np

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
import catboost as cgb

DataStructure = (List[List[int]], List[List[int]], List[List[float]])


@dataclass
class DataReader:
    T: int
    C: int
    Ci: List[int]

    def __post_init__(self):
        self.X_t = []
        self.X_c = []
        self.X_v = []

    def get_data(self) -> DataStructure:
        return self.X_t, self.X_c, self.X_v

    def read_train_data_single_batch(self, B: int) -> DataStructure:
        """Read a random chunk of data

        Args:
            B (int): Number of samples to read

        Returns:
            DataStructure: targets, categories, numbers
        """
        print("req 0 0 {}".format(B))  # requesting all possible data
        sys.stdout.flush()
        return self.process_input()

    def read_train_data_equal_number_all_categories(
        self, B: int, is_complete_sample: bool = False
    ) -> DataStructure:
        """Equal size of sample for each class

        Args:
            B (int): budget
            is_complete_sample (bool, optional): Set to True to top up with random events.
                Defaults to False.

        Returns:
            DataStructure: targets, categories, numbers
        """
        # requesting equal size of each category
        b = B // sum(self.Ci)
        for c in range(self.C):
            ci = self.Ci[c]
            for class_ in range(ci):
                print(f"req {c+1} {class_} {b}")
                sys.stdout.flush()
                self.process_input_and_add_to_lists()
        print(
            f"N samples = {len(self.X_t)}, expected = {int(b*sum(self.Ci))}",
            file=sys.stderr,
        )
        N_samples = len(self.X_t)
        if is_complete_sample:
            if B > N_samples:
                print(f"req 0 0 {B-N_samples}")
                sys.stdout.flush()
                self.process_input_and_add_to_lists()

    def read_train_data_weighted_number_all_categories(
        self, B: int, is_complete_sample: bool = False
    ) -> DataStructure:
        """Sample for each class weighted down by the number of classes within categorical

        Args:
            B (int): budget
            is_complete_sample (bool, optional): Set to True to top up with random events.
                Defaults to False.

        Returns:
            DataStructure: targets, categories, numbers
        """
        # requesting equal size of each category
        B_ = B // self.C
        for c in range(self.C):
            ci = self.Ci[c]
            b = B_ // ci
            for class_ in range(ci):
                print(f"req {c+1} {class_} {b}")
                sys.stdout.flush()
                self.process_input_and_add_to_lists()
        print(
            f"N samples = {len(self.X_t)}, expected = {int(b*sum(self.Ci))}",
            file=sys.stderr,
        )
        N_samples = len(self.X_t)
        if is_complete_sample:
            if B > N_samples:
                print(f"req 0 0 {B-N_samples}")
                sys.stdout.flush()
                self.process_input_and_add_to_lists()

    def read_train_data_half_weighted_half_equal(
        self, B: int, is_complete_sample: bool = False
    ) -> DataStructure:
        """Half samples weighted half not

        Args:
            B (int): budget
            is_complete_sample (bool, optional): Set to True to top up with random events.
                Defaults to False.

        Returns:
            DataStructure: targets, categories, numbers
        """
        # requesting equal size of each chunk
        B_ = B // 2
        self.read_train_data_equal_number_all_categories(B_)
        self.read_train_data_weighted_number_all_categories(B_)
        N_samples = len(self.X_t)
        if is_complete_sample:
            if B > N_samples:
                print(f"req 0 0 {B-N_samples}")
                sys.stdout.flush()
                self.process_input_and_add_to_lists()

    def read_train_data_half_random_half_best_categories(
        self, B: int, is_complete_sample: bool = False
    ) -> DataStructure:
        """Half of events  are sampled with the same number fo samples per class,
        the other half is filled with top-3 classes per ach category and target.

        Args:
            B (int): budget
            is_complete_sample (bool, optional): Set to True to top up with random events.
                Defaults to False.

        Returns:
            DataStructure: targets, categories, numbers
        """
        # requesting equal size of each chunk
        B_ = B // 2
        self.read_train_data_equal_number_all_categories(B_)
        N_samples = len(self.X_t)

        cols_t = [f"T{i}" for i in range(self.T)]
        cols_c = [f"C{i}" for i in range(self.C)]
        df = pd.DataFrame(
            np.concatenate([self.X_t, self.X_c], axis=1), columns=cols_t + cols_c
        )
        good_combos = []
        for c, col_c in enumerate(cols_c):
            for col_t in cols_t:
                top = df.groupby(col_c)[col_t].mean().sort_values(ascending=False)
                for i in range(3):
                    good_combos.append([c, top.index[i]])
        b = (B - N_samples) // len(good_combos)
        for c, class_ in good_combos:
            print(f"req {c+1} {class_} {b}")
            sys.stdout.flush()
            self.process_input_and_add_to_lists()
        if is_complete_sample:
            if B > N_samples:
                print(f"req 0 0 {B-N_samples}")
                sys.stdout.flush()
                self.process_input_and_add_to_lists()

    def process_input(self) -> DataStructure:
        """ Get inputs from stdin, split strings, cast proper data types

        Returns:
            DataStructure: targets, categories, numbers
        """
        train_count = int(input())
        data_targets = []
        data_categories = []
        data_vectors = []
        for _ in range(train_count):
            input_line = input().strip().split()
            data_targets.append(cast_line_elements(input_line[: self.T], int))
            data_categories.append(
                cast_line_elements(input_line[self.T : self.T + self.C], int)
            )
            data_vectors.append(
                cast_line_elements(input_line[self.T + self.C :], float)
            )
        return data_targets, data_categories, data_vectors

    def process_input_and_add_to_lists(self) -> None:
        batch = self.process_input()
        self.X_t += batch[0]
        self.X_c += batch[1]
        self.X_v += batch[2]


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
        statistic_dict = self.prepare_statistic_dict(y, X)
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
            dict_key = self.make_dict_key(X, one_comb)
            for t in range(self.T):
                if dict_key in self.cool_dict_keys[t]:
                    preds[t] = 1
        return preds

    def make_dict_key(self, categories, one_comb):
        """Prepare dictionary key for the Baseline model

        Args:
            categories ([type]): [description]
            one_comb ([type]): [description]

        Returns:
            [type]: [description]
        """
        dict_key = [None] * self.C
        for i in one_comb:
            dict_key[i] = categories[i]
        dict_key = tuple(dict_key)
        return dict_key

    def prepare_statistic_dict(
        self, train_data_targets, train_data_categories,
    ):
        """Compute how ofter each key combo appeared and how ofter targets were fired.

        Args:
            T ([type]): [description]
            C ([type]): [description]
            all_combinations ([type]): [description]
            train_data_targets ([type]): [description]
            train_data_categories ([type]): [description]

        Returns:
            [type]: [description]
        """
        statistic_dict = {}
        for targets, categories in zip(train_data_targets, train_data_categories):
            for one_comb in self.all_combinations:
                dict_key = self.make_dict_key(categories, one_comb)
                if dict_key not in statistic_dict:
                    statistic_dict[dict_key] = {
                        "total_items": 0,
                        "target_items": [0 for t in range(self.T)],
                    }
                statistic_dict[dict_key]["total_items"] += 1
                statistic_dict[dict_key]["target_items"] = list(
                    map(sum, zip(targets, statistic_dict[dict_key]["target_items"]))
                )
        return statistic_dict


@dataclass
class SKLModel:
    """Compute sklearn-like model on the data.

    Any model supporting `.fit()` and `.predict()` methods is allowed.

    Returns:
        [type]: [description]
    """

    T: int
    C: int
    mdl: RandomForestClassifier = RandomForestClassifier()
    p: List[float] = None

    def __post_init__(self):
        if not self.p:
            self.p = [0.5] * self.T

    def fit(self, X, y, **kwargs):
        self.mdl = self.mdl.fit(X, y, **kwargs)
        return self.mdl

    def predict(self, X):
        probs = self.mdl.predict_proba(X)
        preds = np.zeros((len(X), len(self.mdl.classes_)))
        for t in range(self.T):
            preds[:, t] = (probs[t][:, 1] > self.p[t]).astype(np.uint8)
        return preds
        # return np.array(preds).astype(np.uint8)
        # return self.mdl.predict(X)

    def predict_one_line(self, X):
        return self.predict([X])[0]


def split_data_and_evaluate_model(mdl, X, y, retrain: bool = False):
    """80/20 split, model training and evaluation using F1.

    Args:
        mdl ([type]): [description]
        X ([type]): [description]
        y ([type]): [description]
        retrain (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=314)

    mdl = evaluate_model(mdl, X_trn, y_trn, X_tst, y_tst, retrain)
    return mdl


def evaluate_model(mdl, X_trn, y_trn, X_tst, y_tst, retrain: bool = False):
    """Train the model on TRN and evaluate on both TRN and TST.
    F1 measure is used for evaluation.

    Args:
        mdl (Sklearn-model-like): Model
        X_trn (array-like): Training features
        y_trn (array-like): Training targets
        X_tst (array-like): Validation features
        y_tst (array-like): Validation targets

    Returns:
        Sklearn-model-like: Trained model
    """
    mdl.fit(X_trn, y_trn)
    # evaluate on hold-out set
    preds = mdl.predict(X_tst)
    metric = f1_score(y_tst, preds, average=None)  # "macro" / None
    print(f"TST: {np.mean(metric):.6f}: {metric}", file=sys.stderr)
    # evaluate on train set
    preds = mdl.predict(X_trn)
    metric = f1_score(y_trn, preds, average=None)  # "macro" / None
    print(f"TRN: {np.mean(metric):.6f}: {metric}", file=sys.stderr)

    return mdl


def main():
    start_time = datetime.datetime.now()
    # use presence of the file with test targets as a flag of local vs remote execution
    fin_tst_targets = "data/01.a"
    is_local = os.path.exists(fin_tst_targets)

    # get sample properties:
    # * number of tagets(T);
    # * number of categoricals (C);
    # * number of floats (F);
    # * number of classes per categorical (Ci)
    # * number of allowed training samples (B)
    T, C, F = cast_line_elements(input(), int)  # T, C, F
    Ci = cast_line_elements(input(), int)  # C_i
    B = int(input())  # B

    # sample data in one or another way
    dr = DataReader(T, C, Ci)
    # dr.read_train_data_single_batch(B)
    # dr.read_train_data_equal_number_all_categories(B)
    # dr.read_train_data_weighted_number_all_categories(B)
    dr.read_train_data_half_weighted_half_equal(B)
    # dr.read_train_data_half_random_half_best_categories(B)
    y_trn, X_cat_trn, X_vec_trn = dr.get_data()

    # choose the model to use
    # mdl = Baseline(T, C, [2, 3])
    mdl = SKLModel(
        T,
        C,
        mdl=RandomForestClassifier(
            n_estimators=20,
            max_depth=10,
            class_weight="balanced_subsample",  # ["balanced", "balanced_subsample"],
            random_state=314,
            n_jobs=1,
        ),
    )
    # use subset of data for the baseline model
    if isinstance(mdl, Baseline):
        X_trn = X_cat_trn
    else:
        X_trn = np.concatenate([X_cat_trn, X_vec_trn], axis=1)

    print(f"{datetime.datetime.now()} Start modelling on TRN", file=sys.stderr)
    if is_local:
        mdl = split_data_and_evaluate_model(mdl, X_trn, y_trn, retrain=True)
    else:
        mdl.fit(X_trn, y_trn)
    print(f"{datetime.datetime.now()} Finished modelling on TRN", file=sys.stderr)

    # request TEST data
    print("test")
    print(f"{datetime.datetime.now()} Requested the test sample", file=sys.stderr)
    sys.stdout.flush()
    # get the number of test samples
    test_count = int(input())
    print(f"{datetime.datetime.now()} Got TST count", file=sys.stderr)
    # read in whole test data
    X_tst = []
    for i in range(test_count):
        line = input()
        for j in range(1):
            if isinstance(mdl, Baseline):
                line = line.strip().split()[:C]
            X_tst.append(cast_line_elements(line, float))

    print(f"{datetime.datetime.now()} Recieved the test sample", file=sys.stderr)
    X_tst = np.array(X_tst)
    print(f"{datetime.datetime.now()} Preprocessed the test sample", file=sys.stderr)
    if is_local:
        y_tst = []
        with open(fin_tst_targets, "r") as fin:
            _ = fin.readline()
            for i in range(test_count):
                y_tst.append(cast_line_elements(fin.readline().strip().split(), int))

        mdl = evaluate_model(mdl, X_trn, y_trn, X_tst, y_tst, retrain=False)

    # make predictions and extract non-zero indexes for each topic
    all_good_indexes = []
    preds = mdl.predict(X_tst)
    print(f"{datetime.datetime.now()} Predicted on the test sample", file=sys.stderr)
    for t in range(T):
        all_good_indexes.append(preds[:, t].nonzero()[0].tolist())

    print(f"{datetime.datetime.now()} Got indexes for the test sample", file=sys.stderr)
    # output the collected results
    total_good_items = [str(len(x)) for x in all_good_indexes]
    print(" ".join(total_good_items))
    for good_indexes in all_good_indexes:
        # print("\n".join([str(x) for x in good_indexes]))
        for i in good_indexes:
            # print(f'{i} {set(good_indexes)}', file=sys.stderr)
            print(i)
    sys.stdout.flush()
    print(
        f"{datetime.datetime.now()}  Execution time = {(datetime.datetime.now() - start_time).total_seconds()}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
