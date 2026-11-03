import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.model_selection import train_test_split

MISSING_AS_ZERO = ["plas", "pres", "skin", "insu", "mass"]


def load_pima():
    """Pima Indians Diabetes (OpenML): y = +1 при диабете, иначе -1."""
    d = fetch_openml("diabetes", version=1, as_frame=True, parser="auto")
    X = d.data.to_numpy(dtype=float)
    y = np.where(d.target.to_numpy() == "tested_positive", 1, -1)
    return X, y, list(d.data.columns)


def load_breast_cancer_signed():
    d = load_breast_cancer()
    return d.data, np.where(d.target == 1, 1, -1), list(d.feature_names)


def split(X, y, test_size=0.3, seed=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def zeros_to_nan(X, names):
    """В Pima ноль в глюкозе, давлении, толщине кожи, инсулине и ИМТ означает пропуск."""
    X = X.copy()
    for name in MISSING_AS_ZERO:
        j = names.index(name)
        X[X[:, j] == 0, j] = np.nan
    return X


def impute_median(X, ref=None):
    """Заполняет NaN медианой по обучающей выборке (ref)."""
    ref = X if ref is None else ref
    med = np.nanmedian(ref, axis=0)
    return np.where(np.isnan(X), med, X)


def standardize(X, ref=None):
    """Центрирование и масштабирование по статистикам обучающей выборки (ref)."""
    ref = X if ref is None else ref
    return (X - ref.mean(axis=0)) / ref.std(axis=0)


def add_bias(X):
    """Свободный член - ещё один признак f_0(x) = 1."""
    return np.column_stack([np.ones(len(X)), X])


def preprocess(X_train, X_test, names=None):
    """Пропуски (если names заданы), медиана и стандартизация по train, затем столбец единиц."""
    if names is not None:
        X_train, X_test = zeros_to_nan(X_train, names), zeros_to_nan(X_test, names)
    Xtr, Xte = impute_median(X_train), impute_median(X_test, X_train)
    return add_bias(standardize(Xtr, Xtr)), add_bias(standardize(Xte, Xtr))
