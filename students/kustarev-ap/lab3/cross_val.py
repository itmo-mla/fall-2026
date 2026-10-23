import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

from svm_dual import DualSVM


def cross_val_score(kernel, C, X, y, n_splits=5, random_state=42, **kernel_params):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs, f1s = [], []
    for train_idx, val_idx in skf.split(X, y):
        model = DualSVM(kernel, C=C, **kernel_params).fit(X[train_idx], y[train_idx])
        pred = model.predict(X[val_idx])
        accs.append(accuracy_score(y[val_idx], pred))
        f1s.append(f1_score(y[val_idx], pred))
    return np.mean(accs), np.mean(f1s)


def grid_search(kernel, grid, X, y, n_splits=5, random_state=42):
    results = []
    for params in grid:
        C = params["C"]
        kernel_params = {k: v for k, v in params.items() if k != "C"}
        acc, f1 = cross_val_score(kernel, C, X, y, n_splits, random_state, **kernel_params)
        results.append({**params, "accuracy": acc, "f1": f1})
    return results
