from collections import Counter

import numpy as np
import pytest
import os

from task_C import run_using_counter
from pathlib import Path

@pytest.fixture(params=[(5000,500000), (10000,500000), (5000,100000)])
def data_exp(request) -> np.array:
    beta, n = request.param
    x = np.random.exponential(beta, n)
    x = np.round(x)
    return x

def dump_input_file(fout:Path.cwd()/"stump.in", x:np.array) -> None:
    with open(fout, "w") as fout:
        fout.write(f"{len(x)}\n")
        for i in range(len(x)):
            fout.write(f"{x[i]}\n")

def test_Counter(tmp_path, data_exp, capsys):
    os.chdir(tmp_path)
    f_data = Path.cwd()/"x.in"
    dump_input_file(f_data, data_exp)
    with open(f_data, 'r') as fin:
        run_using_counter(source=fin)

    io_captured = capsys.readouterr()
    n_unique_approx = int(io_captured.out.strip())
    # print(f"============ {io_captured.err} ========== ")
    n_unique_bruteforce = len(Counter(data_exp))
    assert np.abs(n_unique_approx/n_unique_bruteforce - 1) < 0.05

