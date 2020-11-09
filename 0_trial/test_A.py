from task_A import find_solution
import os
from pathlib import Path
import pytest
import numpy as np

from typing import Tuple

@pytest.fixture(params=['basic', 'medium', 'long', 'one_value_high', 'negative_values'])
def basic_data(request) -> Tuple[np.array, np.array]:
    np.random.seed = 42
    if request.param == 'basic':
        x = np.array([0,1,2,3])
        y = np.array([1,1,0,0])
    elif request.param == 'medium':
        n = 1000
        x = np.random.randint(1,1000000000,n)
        y = np.random.randint(1,1000,n)
    elif request.param == 'long':
        n = 100000
        x = np.random.randint(1,1000000000,n)
        y = np.random.randint(1,1000,n)
    elif request.param == 'one_value_high':
        n = 100
        x = np.concatenate([np.random.randint(1,100,n-1), [1000]])
        y = np.concatenate([np.random.randint(1,10,n-1), [1000]])
    elif request.param == 'negative_values':
        n = 100
        x = np.random.randint(-100,100,n)
        y = np.random.randint(-1000,-500,n)
    elif request.param == 'zero':
        x = []
        y = []
    return x,y

def dump_input_file(x:np.array,y:np.array) -> None:
    with open(Path.cwd() / "stump.in", "w") as fout:
        fout.write(f"{len(x)}\n")
        for i in range(len(x)):
            fout.write(f"{x[i]} {y[i]}\n")

def test_A_output_format(tmp_path, basic_data, capsys):
    os.chdir(tmp_path)
    dump_input_file(*basic_data)
    find_solution(use_sklearn=True)
    out = Path.cwd() / "stump.out"
    captured = capsys.readouterr()
    assert len(captured.out)==0
    assert out.exists()
    with open(out,"r") as fin:
        # only 1 line
        lines = fin.readlines()
        assert len(lines) == 1
        # only 3 values separated by white space
        vals = lines[0].split(' ')
        assert len(vals) == 3
        # for v in vals:
            # _, decimal = v.split('.')
            # make sure there are at least 6 digets after comma
            # assert len(decimal) >= 6
