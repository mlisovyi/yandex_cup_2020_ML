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
        f_score = 0
        scores = []
        for target_num in range(target_count):
            p = float(true_positives[target_num]) / answer_counts[target_num]
            r = float(true_positives[target_num]) / actual_positives[target_num]
            if true_positives[target_num]:
                f1 = 2 * p * r / (p + r)
                scores.append(f1)
                f_score += f1 / target_count
    print('%.6f' % f_score)
    print([round(x,3) for x in scores])
    sys.exit(Verdict.OK)

if __name__ == '__main__':
    main()
