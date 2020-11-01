from scipy import optimize
import numpy as np

from typing import Tuple, List

def read_data() -> Tuple[np.array, np.array]:
    a = np.genfromtxt("data/B.csv", delimiter=",")
    return a[:, 0], a[:,1]

def evaluate_loss(params: np.array, x:np.array, y:np.array) -> float:
    a, b, c = params
    term1 = a*np.sin(x) + b*np.log(x)
    term2 = c*np.power(x,2)
    term_sum = np.power(term1, 2) + term2
    loss = np.mean(np.abs(term_sum - y))
    return loss

if __name__=="__main__":
    x, y = read_data()
    results = optimize.minimize(fun=evaluate_loss, x0=[1,1,1], args=(x, y), tol=1e-4)
    print(results)