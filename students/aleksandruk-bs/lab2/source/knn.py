"""KNN с окном Парзена переменной ширины.

Метрика Минковского (после нормировки веса признаков w_j = 1):

    ρ(x, x_i) = (sum_j w_j |x_j - x_j^{(i)}|^p )^{1/p}

Классификатор, k — параметр окна, K — ядро:

    a(x; X^ℓ, k, K) = argmax_y  sum_i [y_i = y] K( ρ(x, x_i) / ρ(x, x^{(k+1)}) )

x^{(k+1)} — (k+1)-й сосед, ρ(x, x^{(k+1)}) — переменная ширина h(x).
Гауссово ядро: K(r) = exp(-2 r^2).

Скользящий контроль (объект x_i исключается, расстояние до самого себя не считается):

    LOO(k) = (1/ℓ) sum_i [ a(x_i; X^ℓ \\ {x_i}, k, K) != y_i ]
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def pairwise_minkowski(
    left: np.ndarray,
    right: np.ndarray,
    p: float = 2.0,
    feature_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Матрица расстояний Минковского. При p = 2 и w_j = 1 это евклидова метрика."""
    if feature_weights is not None:
        scale = np.asarray(feature_weights, dtype=np.float64) ** (1.0 / p)
        left = np.asarray(left, dtype=np.float64) * scale
        right = np.asarray(right, dtype=np.float64) * scale
    return cdist(left, right, metric="minkowski", p=p)


def loo_distance_matrix(
    X: np.ndarray,
    p: float = 2.0,
    feature_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Попарные расстояния. На диагонали +inf: объект не является соседом самому себе."""
    dist = pairwise_minkowski(X, X, p=p, feature_weights=feature_weights)
    np.fill_diagonal(dist, np.inf)
    return dist


def gaussian_kernel(r: np.ndarray) -> np.ndarray:
    """K(r) = exp(-2 r^2). В нуле вес 1, на границе окна r = 1 вес e^{-2}."""
    return np.exp(-2.0 * np.square(r))


def _class_ids(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(y)
    index = {int(c): i for i, c in enumerate(classes)}
    y_idx = np.fromiter((index[int(label)] for label in y), dtype=np.int64, count=len(y))
    return classes, y_idx


def parzen_predict_from_distances(dist: np.ndarray, y_train: np.ndarray, k: int) -> np.ndarray:
    """Голос гауссова ядра по всем объектам обучения.

    Ширина строки — расстояние до (k+1)-го соседа (индекс k в упорядоченном ряду,
    нумерация соседей с нуля). При равенстве сумм побеждает меньшая метка класса.
    """
    n_train = dist.shape[1]
    if k < 1 or k >= n_train:
        raise ValueError(f"k={k} невозможен при {n_train} объектах обучения: нужен (k+1)-й сосед")

    order = np.argsort(dist, axis=1, kind="mergesort")
    width = dist[np.arange(dist.shape[0]), order[:, k]]
    if not np.all(np.isfinite(width)):
        raise ValueError("(k+1)-й сосед не найден: k слишком велик или в матрице остался сам объект")
    # Нулевая ширина: несколько объектов совпали. Голосуют только они.
    width_safe = np.where(width > 1e-15, width, np.inf)
    relative = dist / width_safe[:, None]
    weights = gaussian_kernel(relative)
    weights[~np.isfinite(weights)] = 0.0

    classes, y_idx = _class_ids(y_train)
    scores = np.zeros((dist.shape[0], len(classes)), dtype=np.float64)
    for class_id in range(len(classes)):
        scores[:, class_id] = weights[:, y_idx == class_id].sum(axis=1)
    return classes[np.argmax(scores, axis=1)]


def uniform_predict_from_distances(dist: np.ndarray, y_train: np.ndarray, k: int) -> np.ndarray:
    """Прямое голосование k ближайших: вес w(i, x) = [ранг i <= k].

    Так по умолчанию устроен sklearn.neighbors.KNeighborsClassifier.
    """
    n_train = dist.shape[1]
    if k < 1 or k > n_train:
        raise ValueError(f"k={k} невозможен при {n_train} объектах обучения")

    order = np.argsort(dist, axis=1, kind="mergesort")[:, :k]
    votes = y_train[order]
    classes, _ = _class_ids(y_train)
    scores = np.zeros((dist.shape[0], len(classes)), dtype=np.float64)
    for class_id, label in enumerate(classes):
        scores[:, class_id] = np.sum(votes == label, axis=1)
    return classes[np.argmax(scores, axis=1)]


def loo_predictions(dist: np.ndarray, y: np.ndarray, k: int, mode: str = "parzen") -> np.ndarray:
    """Предсказание LOO. dist уже с +inf на диагонали, поэтому x_i исключён."""
    if mode == "parzen":
        return parzen_predict_from_distances(dist, y, k)
    if mode == "uniform":
        return uniform_predict_from_distances(dist, y, k)
    raise ValueError(mode)


def loo_risk(dist: np.ndarray, y: np.ndarray, k: int, mode: str = "parzen") -> float:
    pred = loo_predictions(dist, y, k, mode=mode)
    return float(np.mean(pred != y))


def loo_curve(
    dist: np.ndarray,
    y: np.ndarray,
    mode: str = "parzen",
) -> tuple[np.ndarray, np.ndarray]:
    """LOO(k) для k = 1, ..., ℓ-2. Для окна Парзена нужен (k+1)-й чужой сосед."""
    n_neighbors = dist.shape[1] - 1  # диагональ бесконечна, это не сосед
    k_max = n_neighbors - 1 if mode == "parzen" else n_neighbors
    ks = np.arange(1, k_max + 1)
    risks = np.empty(len(ks), dtype=np.float64)
    for i, k in enumerate(ks):
        risks[i] = loo_risk(dist, y, int(k), mode=mode)
    return ks, risks


def predict_loo_parzen_subset(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    k: int,
) -> tuple[np.ndarray, int]:
    """LOO окна Парзена, когда обучение — подмножество mask.

    Для эталона x_i множество соседей — Ω без x_i. Для остальных — всё Ω.
    k уменьшается, если в Ω не хватает (k+1)-го соседа.
    """
    proto = np.flatnonzero(mask)
    if len(proto) < 3:
        raise ValueError("для окна Парзена на эталонах нужно хотя бы 3 объекта")
    k_used = min(int(k), len(proto) - 2)
    pred = np.empty(len(y), dtype=y.dtype)
    outside = np.flatnonzero(~mask)
    if len(outside):
        pred[outside] = predict(X[proto], y[proto], X[outside], k_used, mode="parzen")
    dist = loo_distance_matrix(X[proto])
    pred[proto] = loo_predictions(dist, y[proto], k_used, mode="parzen")
    return pred, k_used


def predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    k: int,
    mode: str = "parzen",
    p: float = 2.0,
) -> np.ndarray:
    dist = pairwise_minkowski(X_query, X_train, p=p)
    if mode == "parzen":
        return parzen_predict_from_distances(dist, y_train, k)
    if mode == "uniform":
        return uniform_predict_from_distances(dist, y_train, k)
    raise ValueError(mode)
