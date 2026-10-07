"""Загрузка и подготовка данных. Датасет не хранится в репозитории."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_binary_dataset(test_size: float = 0.3, random_state: int = 42):
    """Breast Cancer Wisconsin: бинарная классификация, признаки стандартизуются.

    Метки приводятся к {-1, +1}, к признакам добавляется столбец единиц (свободный член).
    """
    bunch = load_breast_cancer()
    X_raw = bunch.data.astype(np.float64)
    y_raw = bunch.target.astype(np.int64)
    y = np.where(y_raw == 1, 1, -1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    corr = np.corrcoef(X_train, rowvar=False)
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0.0)
    n_high_corr = int(np.sum(np.triu(abs_corr, k=1) > 0.9))

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    X_train_ext = add_bias(X_train_std)
    X_test_ext = add_bias(X_test_std)

    meta = {
        "name": "Breast Cancer Wisconsin (Diagnostic)",
        "source": "sklearn.datasets.load_breast_cancer",
        "n_features": X_raw.shape[1],
        "feature_names": list(bunch.feature_names),
        "target_names": list(bunch.target_names),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_missing": int(np.isnan(X_raw).sum()),
        "n_pos_full": int(np.sum(y == 1)),
        "n_neg_full": int(np.sum(y == -1)),
        "n_pos_train": int(np.sum(y_train == 1)),
        "n_neg_train": int(np.sum(y_train == -1)),
        "n_pos_test": int(np.sum(y_test == 1)),
        "n_neg_test": int(np.sum(y_test == -1)),
        "feature_min": X_train.min(axis=0),
        "feature_max": X_train.max(axis=0),
        "feature_mean": X_train.mean(axis=0),
        "corr_train": corr,
        "n_high_corr_pairs": n_high_corr,
        "mean_abs_corr": float(np.mean(np.triu(abs_corr, k=1))),
        "scaler": scaler,
    }
    return X_train_ext, X_test_ext, y_train, y_test, meta


def add_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((X.shape[0], 1), dtype=np.float64), X])
