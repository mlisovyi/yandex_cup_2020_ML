# Final (6-8 November 2020)

- [Final (6-8 November 2020)](#final-6-8-november-2020)
  - [Training with limited sample size](#training-with-limited-sample-size)
  - [The data](#the-data)
  - [Challenges](#challenges)
  - [SW setup](#sw-setup)
  - [Solution](#solution)
  - [Summary](#summary)

## Training with limited sample size

The full task description is given in [A.md](./A.md) (in Russian).
Below is a short summary.

## The data

**The task is to find the optimal sampling strategy to get the most representative data sample keeping the sample size small.**
We are given a classification dataset with:
* 4 categorical features with different number of classes;
* 40 numerical features (already scaled);
* no missing values, no clear outliers;
* 3 targets (*aka* multilabel problem).
  The labels seem to be mutually exclusive,
  so the problem can be alternatively framed as multiclass problem.

## Challenges

One can sample only up to **10k events to be used to train a model**.
The final **testing sample is 100x larger**, i.e. 1M events.
The sample is **heavily biased**- the fraction of each label is only **0.2%**.
The final evaluation metric is the *average F1*,
therefore it is important to predict well all three labels and
it is important to get the rare positive labels right.

The twist of the problem is that one can control what data does one recieve.
One can either request randomly-sampled data or data contrained
to the particular class of the chosen categorical variable.
The final evaluation by organisers was executed in an interactive fashion,
where participant's solution can make multiple requests for data within
the allocated training budget.
This allows to request data that would cover all possible catgorical classes in the data.

One further constraint was the execution time.
The solution had to complete within **40 seconds** on a 1-core machine with a limit of 3GB RAM.
The interactive system itself was slow-
recieval of the test data alone was eating around **10-15 seconds** out of that CPU time budget
(estimated projecting local tests to the final testing sample size).
Out of the residual time most was used to make predictions,
as the test sample is much larger than the sample used for training.

## SW setup

The organisers have prepared a fully functional baseline setup
to elumate the interactive problem and a simple heuristic solution
(average frequency in dublets and tripples of categorical classes)
The code in the `2_final` directory was largely provided by orginisers.
The `2_final/solution.py` script was developed to modularise the baseline model,
try out more sophisticated models and to try different sampling types.

Among other things, since this used a python 3.7 environment,
this was an opportunity for me to try out `dataclasses` in practise-
something that i have heard often about, but never tried,
as I've mostly developed in python3.6 up to now.

The relevant conda environment is documented in `env.yaml`.

To run the solution in the interactive mode one would execute
```bash
python final/testing_tool.py python final/interact.py data/01 data/output.txt -- python final/solution.py
```
This would start both the data provider as well as the consumer
(the solution) in threads that communicate via `stdout`.
No data are provided.

## Solution

Was has been tried:
* sampling with different strategies:
  * one chunk of random data;
  * equal sizes of samples of all classes;
  * sample size per class weighted down by the number of classes within categorical;
  * half and half of the second and third strategies;
  * half the budget sampled with equal sample sizes,
    classes with highest target frequencies were evaluated
    and the second half was composed of those most promissing classes;
  * locally the last strategy has shown the best validation
    score on a hold-out set of 100k events,
    but on the "public leaderboard" it was fluctuating up and down.
* different models:
  * heuristics baseline with different class combinations and thresholds;
  * gradient-boosting trees
    (catboost as no lightgbm and no xgboost was available in the test environment
    and categoricals might have played an important role);
  * logistic regression;
  * random forest.

At the end i have settled on the random forest model that was quite shallow (depth 10)
and a small number of trees in the forest (20),
as anything beyond was showing better performance locally
but was too slow to fit into the time quota.
I also had to vectorise some of the provided initial components of the starter solution
to speed up prediction step.

## Summary

All in all it was a very interesting problem to solve- something that goes beyond
the typical `fit+predict` sequnce of some other competition.
I did not get a particularly good solution
(on the public leaderboard I landed somewhere in the second half of the participant list),
but I learned new skills and enjoyed tackling an unusual problem.

Big thanks to Yandex for organisation of the Yandex cup 2020!

