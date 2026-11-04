import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from kernel_knn import gauss_window_weights, pairwise_distances, vote_scores


def loo_error_curve(
    k_values: list[int] | np.ndarray,
    X_reference: np.ndarray,
    y_reference: np.ndarray,
    X_eval: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
    self_columns: np.ndarray | None = None,
) -> np.ndarray:
    """Доля ошибок LOO для каждого k; self_columns исключает сам объект"""
    if X_eval is None:
        X_eval, y_eval = X_reference, y_reference
    X_reference = np.asarray(X_reference, dtype=float)
    X_eval = np.asarray(X_eval, dtype=float)
    y_reference = np.asarray(y_reference)
    y_eval = np.asarray(y_eval)

    if self_columns is None:
        self_columns = np.full(len(X_eval), -1, dtype=int)
    self_columns = np.asarray(self_columns, dtype=int)

    distances = pairwise_distances(X_eval, X_reference)
    has_self = self_columns >= 0
    distances[np.flatnonzero(has_self), self_columns[has_self]] = np.inf

    max_neighbors = int(np.min(np.isfinite(distances).sum(axis=1)))
    k_values = np.asarray(k_values, dtype=int)
    if k_values.max() > max_neighbors:
        raise ValueError("k превышает число доступных соседей при LOO")

    classes = np.unique(y_reference)
    risks = []
    for k in k_values:
        weights = gauss_window_weights(distances, int(k))
        scores = vote_scores(weights, y_reference, classes)
        predictions = classes[np.argmax(scores, axis=1)]
        risks.append(float(np.mean(predictions != y_eval)))
    return np.asarray(risks)


def pick_best_k(k_values: np.ndarray, risks: np.ndarray) -> int:
    """Минимум риска; при равенстве выбирается меньший k"""
    k_values = np.asarray(k_values, dtype=int)
    risks = np.asarray(risks, dtype=float)
    minima = np.flatnonzero(np.isclose(risks, risks.min(), rtol=0.0, atol=1e-12))
    return int(k_values[minima[0]])


def sklearn_loo_error_curve(
    k_values: list[int] | np.ndarray, X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """LOO для эталонного KNN: k+1 соседей, затем удаление самого объекта"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes, inverse = np.unique(y, return_inverse=True)
    n_objects = len(y)
    k_values = np.asarray(k_values, dtype=int)
    max_k = int(k_values.max())

    model = KNeighborsClassifier(n_neighbors=max_k + 1).fit(X, y)
    neighbors = model.kneighbors(X, n_neighbors=max_k + 1, return_distance=False)
    own = np.arange(n_objects)[:, None]
    neighbors = neighbors[neighbors != own].reshape(n_objects, max_k)

    rows = np.arange(n_objects)
    risks = []
    for k in k_values:
        votes = np.zeros((n_objects, len(classes)), dtype=int)
        selected = inverse[neighbors[:, :k]].ravel()
        np.add.at(votes, (np.repeat(rows, k), selected), 1)
        predictions = classes[np.argmax(votes, axis=1)]
        risks.append(float(np.mean(predictions != y)))
    return np.asarray(risks)
