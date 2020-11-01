# Trial round

The trial round was not limited in time and unfortunately the task description has been lost :()

## Summary


* A. **Decision Stump**: given _x_ and _y_ arrays, build a decision stump by finding the split _c_ and the two  values _a, b_ such that RMS is optimal.
* B. **Function fit**: given _x_ and _y_ arrays, fit `(a*sin(x) + b*log(x))**2 + c * x**2` to minimise MAE on _y_
* C. **Estimate the number of  unique terms**: given a very long input that contains repeating entries, estimate the number of unique values with a certaint precision (`~5%`).
  The difficulty here was that the memory was execution time were strongly constrained.
  Basic `set` or import of `pandas` were crossing the memory limit.
* D. **Restaurant reviews**: for training there was a data set
  of paired comparisons between restaurants.
  Distances and average reviews.
  For each pair there was the decision of the user given- which one to choose.
  The task was to be able to generate user ratings for individual restaurants.
