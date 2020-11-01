#%%
import pandas as pd
import numpy as np
import catboost as cgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

#%%
df = pd.read_json('input.txt')


# %%
params_ref = { 
    "iterations": 100, 
    "learning_rate": 0.0001, 
    "depth": 1, 
    "l2_leaf_reg": 100.0, 
    "rsm": 0.9, 
    "border_count": 10, 
    "max_ctr_complexity": 3, 
    "random_strength": 40.0, 
    "bagging_temperature": 100.0, 
    "grow_policy": "Depthwise", 
    "min_data_in_leaf": 5, 
    "langevin": True, 
    "diffusion_temperature": 100000 
} 
# %%
def evaluate_params(params, df):
    mdl = cgb.CatBoostClassifier(verbose=False, **params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=314)
    X = df.drop('label', axis=1)
    y = df['label']
    perf_trn = []
    perf_val = []
    for idx_trn, idx_val in cv.split(X, y):
        X_trn, X_val = X.loc[idx_trn,:], X.loc[idx_val,:]
        y_trn, y_val = y[idx_trn], y[idx_val]

        mdl.fit(X_trn, y_trn)
        preds = mdl.predict_proba(X_val)[:,1]
        perf_val.append(roc_auc_score(y_val, preds))
        preds = mdl.predict_proba(X_trn)[:,1]
        perf_trn.append(roc_auc_score(y_trn, preds))

    print(f"ROC AUC = {np.mean(perf_val):.3f} +- {np.std(perf_val):.3f}")
    print(f"(TRN) Roc AUC = {np.mean(perf_trn):.3f} +- {np.std(perf_trn):.3f}")
    return perf_val, perf_trn

# %%
# evaluate_params(params_ref, df)

# %%
params = { 
    "iterations": 200, 
    "learning_rate": 0.1, 
    "rsm": 0.5,
    "depth": 10, 
} 
_ = evaluate_params(params, df)
# %%

# %%
