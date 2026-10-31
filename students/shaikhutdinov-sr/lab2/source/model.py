from dataclasses import dataclass

import numpy as np


def parzen_weights(neighbor_distances: np.ndarray, k: int) -> np.ndarray:
    if not 1 <= k < len(neighbor_distances):
        raise ValueError("Для ширины окна нужны как минимум k + 1 соседей")

    window_width = float(np.partition(neighbor_distances, k)[k])
    if not np.isfinite(window_width):
        raise ValueError("После исключения объекта осталось меньше k + 1 соседей")

    if window_width == 0:
        return (neighbor_distances == 0).astype(float)
    return np.exp(-2 * (neighbor_distances / window_width) ** 2)


@dataclass
class ParzenKNN:
    x_neighbors: np.ndarray
    y_neighbors: np.ndarray
    k: int

    def predict(self, x_objects: np.ndarray) -> np.ndarray:
        classes, neighbor_class_indices = np.unique(self.y_neighbors, return_inverse=True)
        predicted_labels = np.empty(len(x_objects), dtype=self.y_neighbors.dtype)
        for i, x in enumerate(x_objects):
            neighbor_distances = np.sqrt(np.sum((self.x_neighbors - x) ** 2, axis=1))
            neighbor_weights = parzen_weights(neighbor_distances, self.k)
            class_votes = np.bincount(
                neighbor_class_indices, weights=neighbor_weights, minlength=len(classes)
            )
            predicted_labels[i] = classes[np.argmax(class_votes)]
        return predicted_labels


def loo_risks(
    x_train: np.ndarray,
    y_train: np.ndarray,
    k_values: list[int],
    neighbor_indices: np.ndarray | None = None,
) -> np.ndarray:
    if neighbor_indices is None:
        neighbor_indices = np.arange(len(y_train))

    x_neighbors = x_train[neighbor_indices]
    y_neighbors = y_train[neighbor_indices]
    classes, neighbor_class_indices = np.unique(y_neighbors, return_inverse=True)
    error_counts = np.zeros(len(k_values), dtype=int)

    for i, x in enumerate(x_train):
        neighbor_distances = np.sqrt(np.sum((x_neighbors - x) ** 2, axis=1))
        neighbor_distances[neighbor_indices == i] = np.inf
        for k_index, k in enumerate(k_values):
            neighbor_weights = parzen_weights(neighbor_distances, k)
            class_votes = np.bincount(
                neighbor_class_indices, weights=neighbor_weights, minlength=len(classes)
            )
            predicted_label = classes[np.argmax(class_votes)]
            if predicted_label != y_train[i]:
                error_counts[k_index] += 1
    return error_counts / len(y_train)
