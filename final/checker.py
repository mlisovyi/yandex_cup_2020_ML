#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

# This program is run like
# python3 checker.py input output answer
#
# * input - input file (`01` in the archive)
# * output - output from interactor
# * answer - file with correct answer (`01.a` in the archive)

import sys
from final.solution import cast_line_elements

from sklearn.metrics import f1_score

import numpy as np

class Verdict:
    OK = 0
    WA = 1
    PE = 2
    FAIL = 10

def main():
    with open(sys.argv[3], 'r') as answer_file:
        target_count, test_count = cast_line_elements(answer_file.readline(), int)
        test_targets = [None] * test_count
        actual_positives = [0] * target_count
        for i in range(test_count):
            targets = cast_line_elements(answer_file.readline(), int)
            assert len(targets) == target_count
            test_targets[i] = targets
            for j, x in enumerate(targets):
                if x:
                    actual_positives[j] += 1
    y = np.array(test_targets)

    preds_index = []
    with open(sys.argv[2], 'r') as output_file:
        answer_counts = cast_line_elements(output_file.readline(), int)
        assert len(answer_counts) == target_count

        true_positives = [0] * target_count
        predicted_positives = [0] * target_count
        
        for target_num, answer_count in enumerate(answer_counts):
            assert 0 < answer_count <= test_count
            indexes = set()
            for i in range(answer_count):
                index = int(output_file.readline().strip())
                assert 0 <= index < test_count
                assert index not in indexes
                indexes.add(index)
                predicted_positives[target_num] += 1
                if test_targets[index][target_num]:
                    true_positives[target_num] += 1
            preds_index.append(list(indexes))

    # compute the score
    preds = np.zeros_like(y)
    for t in range(target_count):
        preds[preds_index[t], t] = 1
    metric = f1_score(y, preds, average=None)
    print(f"{np.mean(metric):.6f}: {metric}", file=sys.stderr)

    sys.exit(Verdict.OK)

if __name__ == '__main__':
    main()
