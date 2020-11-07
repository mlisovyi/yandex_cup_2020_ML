#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

# This program is run like
# python3 interact.py input output
#
# * input - input file (`01` in the archive)
# * output - write output for checker here

import sys
import datetime
import traceback
from final.solution import cast_line_elements

DEBUG = True

class Verdict:
    OK = 0
    WA = 1
    PE = 2
    FAIL = 10

def verify(cond, message):
    if not cond:
        print(message, file=sys.stderr)
        sys.exit(Verdict.PE)

class FileLinesHandle:
    def __init__(self, file, position):
        self.file = file
        self.position = position
        self.count = None

    def read_lines(self, count):
        self.file.seek(self.position)
        if self.count is None:
            self.count = int(self.file.readline())
            self.position = self.file.tell()
        count = min(count, self.count)
        yield count
        while count > 0:
            count -= 1
            self.count -= 1
            line = self.file.readline()
            self.position = self.file.tell()
            yield line

    def respond(self, count):
        result = None
        for line in self.read_lines(count):
            if result is None:
                result = line
                print(line)
            else:
                print(line, end='')
        assert result is not None
        sys.stdout.flush()
        return result

def main():
    try:
        with open(sys.argv[1], "r") as input_file:
            start_time = datetime.datetime.now()

            first_line = input_file.readline()
            print(first_line, end='')
            target_count, class_count, _ = cast_line_elements(first_line, int)
            class_counts = input_file.readline()
            print(class_counts, end='')
            class_counts = cast_line_elements(class_counts, int)
            budget = int(input_file.readline())
            print(budget)
            sys.stdout.flush()

            offsets = cast_line_elements(input_file.readline(), int)
            start_position = input_file.tell()
            global_train = FileLinesHandle(input_file, start_position)
            offset_index = 0
            constrained_train = [[] for class_index in range(class_count)]
            for handles, class_count in zip(constrained_train, class_counts):
                for _ in range(class_count):
                    handles.append(FileLinesHandle(input_file, start_position + offsets[offset_index]))
                    offset_index += 1
            test_offset = start_position + offsets[offset_index]
            assert offset_index == len(offsets) - 1

            if DEBUG:
                print(f'Initialized in {datetime.datetime.now() - start_time}', file=sys.stderr)
            start_time = datetime.datetime.now()
            
            while True:
                try:
                    request = input().strip().split()
                except:
                    verify(False, 'Could not read line')
                if request[0] == 'req':
                    try:
                        class_index, class_value, count = cast_line_elements(request[1:], int)
                    except:
                        verify(False, f'Invalid request: {request}')
                    verify(count > 0, f'Invalid count: {count}')
                    verify(count <= budget, f'Budget exceeded')
                    if class_index == 0:
                        verify(class_value == 0, f'Invalid request: {request}')
                        budget -= global_train.respond(count)
                    else:
                        verify(0 < class_index <= class_count, f'Invalid class index: {request}')
                        class_index -= 1
                        verify(0 <= class_value < class_counts[class_index], f'Invalid class value in request: {request}')
                        budget -= constrained_train[class_index][class_value].respond(count)
                elif request[0] == 'test':
                    verify(len(request) == 1, f'Invalid request: {request}')
                    break
                else:
                    verify(False, f'Invalid request: {request}')

            if DEBUG:
                print(f'Requests processed in {datetime.datetime.now() - start_time}', file=sys.stderr)
            start_time = datetime.datetime.now()

            input_file.seek(test_offset)
            test_count = int(input_file.readline())
            print(test_count)
            for _ in range(test_count):
                print(input_file.readline(), end='')
            sys.stdout.flush()

            if DEBUG:
                print(f'Testset sent in {datetime.datetime.now() - start_time}', file=sys.stderr)
            start_time = datetime.datetime.now()

            with open(sys.argv[2], "w") as output_file:
                answer_counts = input()
                print(answer_counts, file=output_file)
                try:
                    answer_counts = cast_line_elements(answer_counts, int)
                    verify(len(answer_counts) == target_count, f'Invalid number of answer counts')
                except:
                    verify(False, f'Invalid answer counts: {answer_counts}')

                for _, answer_count in enumerate(answer_counts):
                    verify(0 < answer_count <= test_count, f'Invalid number of predicted positives: {answer_count}')
                    indexes = set()
                    for _ in range(answer_count):
                        index = input()
                        print(index, file=output_file)
                        try:
                            index = int(index.strip())
                        except:
                            verify(False, f'Invalid sample index: {index}')
                        verify(0 <= index < test_count, f'Invalid sample index: {index}')
                        verify(index not in indexes, f'Repeating sampling index: {index}')
                        indexes.add(index)

            if DEBUG:
                print(f'Got answers in {datetime.datetime.now() - start_time}', file=sys.stderr)

            sys.exit(Verdict.OK)

    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(Verdict.FAIL)

if __name__ == '__main__':
    main()
