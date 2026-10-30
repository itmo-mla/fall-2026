import numpy as np


def pairwise_distances(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    squared = (
        np.sum(X**2, axis=1)[:, None] + np.sum(Z**2, axis=1)[None, :] - 2 * X @ Z.T
    )
    return np.sqrt(np.maximum(squared, 0))


def gaussian_kernel(values: np.ndarray) -> np.ndarray:
    return np.exp(-2 * values**2)


def _class_votes(
    labels: np.ndarray,
    weights: np.ndarray,
    class_count: int,
) -> np.ndarray:
    votes = np.zeros((weights.shape[0], class_count))
    for class_index in range(class_count):
        votes[:, class_index] = weights[:, labels == class_index].sum(axis=1)
    return votes


def _predict_knn_from_distances(
    distances: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> np.ndarray:
    if not 1 <= k <= distances.shape[1]:
        raise ValueError("k выходит за допустимый диапазон")

    neighbors = np.argsort(distances, axis=1)[:, :k]
    neighbor_labels = labels[neighbors]
    predictions = np.empty(len(distances), dtype=int)

    for index, row in enumerate(neighbor_labels):
        predictions[index] = np.bincount(row).argmax()
    return predictions


def _predict_parzen_from_distances(
    distances: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> np.ndarray:
    if not 1 <= k < distances.shape[1]:
        raise ValueError("Для окна Парзена нужен (k + 1)-й сосед")

    # Индекс k соответствует (k + 1)-му соседу при нумерации с единицы
    ordered = np.sort(distances, axis=1)
    bandwidth = ordered[:, k].copy()

    # При совпадающих объектах используем первое положительное расстояние
    zero_width = bandwidth <= 1e-12
    if np.any(zero_width):
        ordered_zero = ordered[zero_width]
        positive = np.where(
            (ordered_zero > 1e-12) & np.isfinite(ordered_zero),
            ordered_zero,
            np.inf,
        )
        first_positive = positive.min(axis=1)
        bandwidth[zero_width] = np.where(
            np.isfinite(first_positive), first_positive, 1.0
        )

    # Веса получают все объекты, а k определяет только ширину окна
    weights = gaussian_kernel(distances / bandwidth[:, None])
    votes = _class_votes(labels, weights, int(labels.max()) + 1)
    return votes.argmax(axis=1)


def predict_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    k: int,
) -> np.ndarray:
    distances = pairwise_distances(X_query, X_train)
    return _predict_knn_from_distances(distances, y_train, k)


def predict_parzen(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: np.ndarray,
    k: int,
) -> np.ndarray:
    distances = pairwise_distances(X_query, X_train)
    return _predict_parzen_from_distances(distances, y_train, k)


def calculate_loo_risks(
    X: np.ndarray,
    y: np.ndarray,
    method: str,
) -> np.ndarray:
    distances = pairwise_distances(X, X)

    # Бесконечность на диагонали исключает объект из собственных соседей
    np.fill_diagonal(distances, np.inf)
    risks = []

    for k in range(1, len(X) - 1):
        if method == "knn":
            prediction = _predict_knn_from_distances(distances, y, k)
        elif method == "parzen":
            prediction = _predict_parzen_from_distances(distances, y, k)
        else:
            raise ValueError(f"Неизвестный метод: {method}")
        risks.append(np.mean(prediction != y))

    return np.array(risks)


def calculate_compactness_profile(
    X: np.ndarray,
    y: np.ndarray,
    max_neighbors: int = 20,
    prototype_indices: np.ndarray | None = None,
) -> np.ndarray:
    if prototype_indices is None:
        reference_X = X
        reference_y = y
        distances = pairwise_distances(X, X)
        np.fill_diagonal(distances, np.inf)
        available_neighbors = len(X) - 1
    else:
        reference_X = X[prototype_indices]
        reference_y = y[prototype_indices]
        distances = pairwise_distances(X, reference_X)

        # Эталон тоже не должен становиться собственным соседом
        for column, source_index in enumerate(prototype_indices):
            distances[source_index, column] = np.inf
        available_neighbors = len(prototype_indices) - 1

    neighbor_count = min(max_neighbors, available_neighbors)
    neighbors = np.argsort(distances, axis=1)[:, :neighbor_count]
    neighbor_labels = reference_y[neighbors]
    return np.mean(neighbor_labels != y[:, None], axis=0)


def _calculate_ccv_q1(
    distances: np.ndarray,
    y: np.ndarray,
    prototype_indices: list[int],
) -> float:
    prototype_indices_array = np.array(prototype_indices, dtype=int)
    selected_distances = distances[:, prototype_indices_array].copy()

    for column, source_index in enumerate(prototype_indices_array):
        selected_distances[source_index, column] = np.inf

    nearest = selected_distances.argmin(axis=1)
    prediction = y[prototype_indices_array[nearest]]

    # При длине контроля q=1 критерий CCV равен Pi(1)
    return float(np.mean(prediction != y))


def select_prototypes(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distances = pairwise_distances(X, X)
    prototypes = []

    # Начинаем с объекта, ближайшего к центру каждого класса
    for class_index in np.unique(y):
        class_indices = np.flatnonzero(y == class_index)
        center = X[class_indices].mean(axis=0)
        center_distances = np.linalg.norm(X[class_indices] - center, axis=1)
        prototypes.append(int(class_indices[center_distances.argmin()]))

    prototypes.sort()
    current_error = _calculate_ccv_q1(distances, y, prototypes)
    history = [(len(prototypes), current_error)]

    while True:
        best_candidate = None
        best_error = current_error

        for candidate in range(len(X)):
            if candidate in prototypes:
                continue
            candidate_error = _calculate_ccv_q1(
                distances,
                y,
                [*prototypes, candidate],
            )
            if candidate_error < best_error - 1e-12:
                best_candidate = candidate
                best_error = candidate_error

        if best_candidate is None:
            break

        prototypes.append(best_candidate)
        prototypes.sort()
        current_error = best_error
        history.append((len(prototypes), current_error))

    return np.array(prototypes, dtype=int), np.array(history, dtype=float)
