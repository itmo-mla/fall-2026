import numpy as np
from sklearn.datasets import make_blobs, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _pack(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> dict:
    y = np.asarray(y, dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=random_state,
    )
    scaler = StandardScaler()
    return {
        "X_train": scaler.fit_transform(X_train),
        "X_test": scaler.transform(X_test),
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": ["x1", "x2"],
        "visual_features": ["x1", "x2"],
        "scaler": scaler,
        "label_meaning": {-1: "класс -1", 1: "класс +1"},
    }


def prepare_blobs(random_state: int = 42) -> dict:
    X, y = make_blobs(
        n_samples=150,
        centers=2,
        n_features=2,
        cluster_std=1.4,
        random_state=random_state,
    )
    return _pack(X, 2 * y - 1, random_state=random_state)


def prepare_circles(random_state: int = 42) -> dict:
    X, y = make_circles(
        n_samples=200,
        factor=0.4,
        noise=0.08,
        random_state=random_state,
    )
    return _pack(X, 2 * y - 1, random_state=random_state)
