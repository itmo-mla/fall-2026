import numpy as np


def random_initialization(n_features, seed=None, scale=0.01):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, scale, n_features), 0.0


def correlation_initialization(X, y):
    # w_j ~ corr(x_j, y), вектор нормируется к единичной длине; b = среднее y
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    denom = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
    corr = np.divide(Xc.T @ yc, denom, out=np.zeros(X.shape[1]), where=denom > 0)
    norm = np.linalg.norm(corr)
    w = corr / norm if norm > 0 else corr
    return w, float(y.mean())
