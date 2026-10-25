import numpy as np

KERNEL_EXPONENT = 0.5


def pairwise_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Евклидовы расстояния между строками A и B"""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    squared = aa + bb - 2.0 * (A @ B.T)
    return np.sqrt(np.maximum(squared, 0.0))


def gauss_window_weights(
    distances: np.ndarray, n_neighbors: int
) -> np.ndarray:
    """Веса k ближайших объектов; ширина окна — расстояние до k-го соседа"""
    if n_neighbors < 1:
        raise ValueError("k должно быть не меньше 1")
    if n_neighbors > distances.shape[1]:
        raise ValueError("k превышает число доступных эталонов")

    rows = np.arange(distances.shape[0])[:, None]
    nearest = np.argpartition(distances, n_neighbors - 1, axis=1)[:, :n_neighbors]
    local = np.take_along_axis(distances, nearest, axis=1)
    window = local.max(axis=1)

    weights = np.zeros_like(distances, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.exp(-KERNEL_EXPONENT * (local / window[:, None]) ** 2)
    weights[rows, nearest] = values

    # Совпадающие объекты образуют окно нулевой ширины.
    zero_window = window <= 0
    if np.any(zero_window):
        weights[zero_window] = (distances[zero_window] <= 0).astype(float)
    return weights


def vote_scores(
    weights: np.ndarray, reference_labels: np.ndarray, classes: np.ndarray
) -> np.ndarray:
    """Взвешенные голоса классов для каждой строки матрицы весов"""
    return np.column_stack(
        [weights @ (reference_labels == label).astype(float) for label in classes]
    )


class WindowVoteKNN:
    """KNN с гауссовым окном Парзена переменной ширины"""

    def __init__(self, n_neighbors: int = 5):
        if n_neighbors < 1:
            raise ValueError("k должно быть не меньше 1")
        self.n_neighbors = int(n_neighbors)
        self.X_: np.ndarray | None = None
        self.y_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "WindowVoteKNN":
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        self.classes_ = np.unique(self.y_)
        if len(self.classes_) < 2:
            raise ValueError("Для классификации нужны как минимум два класса")
        return self

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        if self.X_ is None or self.y_ is None or self.classes_ is None:
            raise RuntimeError("Сначала вызовите fit()")
        distances = pairwise_distances(np.asarray(X, dtype=float), self.X_)
        weights = gauss_window_weights(distances, self.n_neighbors)
        return vote_scores(weights, self.y_, self.classes_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_scores(X)
        return self.classes_[np.argmax(scores, axis=1)]
