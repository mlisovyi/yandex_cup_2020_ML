# Qualification round (19-25 Oct 2020)

- [Qualification round (19-25 Oct 2020)](#qualification-round-19-25-oct-2020)
  - [Summary](#summary)
  - [Solution A](#solution-a)
  - [Solution B (attempt)](#solution-b-attempt)

## Summary

The competition time was restricted to 4 hours only. There have been 4 tasks in total, task description is available in Russion only :( ):

* A. [Binary classifier (Catboost)](A.md)
* B. [Classification with constraints](B.md)
* C. [Classification of blured images](C.md)
* D. [Beta distrivution](D.md)

I managed to solve only the task A (the simplest) and tried without success D.
Short description of the solution logic follows.

## Solution A

The task was to find better hyperparameters for catboost GBM on predefined data.
I did not have prior experience with catboost,
but the data seemed to not have any categorical variables. 
Therefore, I decided to treat the model configuration as any other GBM (e.g. lightgbm/xgboost).

The good starting point typically is:

* learning rate of _0.1_
* number of trees of the order of _100_
* feature and row sampling of around _0.75_

The solution is available in [A.py](./A.py)
The final submitted parameters are availablle in [A_params.json](./A_params.json)

## Solution B (attempt)

The ask was effectively to fit the beta distribution to available data.