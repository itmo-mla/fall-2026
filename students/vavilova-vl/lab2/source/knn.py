import numpy as np


EPS = np.finfo(float).eps


def gaussian_kernel(r: np.ndarray) -> np.ndarray:
    """Gaussian kernel G(r) = exp(-2 r^2) from the lecture."""
    return np.exp(-2.0 * r**2)


def parzen_weights(distances: np.ndarray, k: int) -> np.ndarray:
    """Variable-width Parzen window weights K(rho(x, x_i) / h(x)), h(x) = rho(x, x^(k+1)).

    `distances` has shape (n_queries, n_train) and holds distances to all training
    objects in any order; excluded objects may be marked with +inf.
    """
    distances = np.atleast_2d(np.asarray(distances, dtype=float))
    # (k+1)-th smallest distance in every row (index k after zero-based partition).
    bandwidth = np.partition(distances, k, axis=1)[:, k : k + 1]
    degenerate = bandwidth <= EPS
    weights = gaussian_kernel(distances / np.where(degenerate, 1.0, bandwidth))
    # k+1 neighbors coincide with the query: only exact matches vote (the h -> 0 limit).
    return np.where(degenerate, (distances <= EPS).astype(float), weights)


def pairwise_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)


def one_hot(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return (y[:, None] == classes[None, :]).astype(float)


class ParzenKNN:
    """Parzen window classifier with variable width h(x) = rho(x, x^(k+1)) and a Gaussian kernel."""

    def __init__(self, k: int = 5):
        if k < 1:
            raise ValueError("k must be positive")
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ParzenKNN":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X must be 2D and y must be a matching 1D array")
        if self.k + 1 > len(y):
            raise ValueError("the window needs at least k + 1 training samples")
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Class votes Gamma_y(x), shape (n_queries, n_classes)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.X_.shape[1]:
            raise ValueError("X must have the same number of features as the training data")
        weights = parzen_weights(pairwise_distances(X, self.X_), self.k)
        return weights @ one_hot(self.y_, self.classes_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_scores(X), axis=1)]


def loo_risk_curve(
    X: np.ndarray, y: np.ndarray, k_values: range | list[int] | np.ndarray
) -> dict[int, float]:
    """Return the leave-one-out classification error for every candidate k."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X must be 2D and y must be a matching 1D array")
    candidates = sorted(set(int(k) for k in k_values))
    # Without x_i there are l - 1 objects left, and the window needs the (k+1)-th of them.
    if not candidates or candidates[0] < 1 or candidates[-1] + 1 > len(X) - 1:
        raise ValueError("candidate k values must be in [1, n_samples - 2]")

    distances = pairwise_distances(X, X)
    # x_i is excluded from its own neighbors: infinite distance means zero weight.
    np.fill_diagonal(distances, np.inf)
    classes = np.unique(y)
    labels = one_hot(y, classes)
    risks = {}
    for k in candidates:
        scores = parzen_weights(distances, k) @ labels
        predictions = classes[np.argmax(scores, axis=1)]
        risks[k] = float(np.mean(predictions != y))
    return risks
