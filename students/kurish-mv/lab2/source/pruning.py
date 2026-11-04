from dataclasses import dataclass

import numpy as np
import pandas as pd

from kernel_knn import pairwise_distances


@dataclass
class CondensationResult:
    keep_mask: np.ndarray
    base_risk: float
    final_risk: float
    influence: np.ndarray
    critical_mask: np.ndarray
    protected_mask: np.ndarray
    history: pd.DataFrame

    @property
    def keep_indices(self) -> np.ndarray:
        return np.flatnonzero(self.keep_mask)


def one_nn_risk(
    distances: np.ndarray, labels: np.ndarray, keep_mask: np.ndarray
) -> float:
    """Ошибка 1NN на всех объектах при классификации по эталонам"""
    indices = np.flatnonzero(keep_mask)
    if len(indices) == 0:
        return 1.0
    if len(indices) == 1:
        return float(np.mean(labels != labels[indices[0]]))

    submatrix = distances[:, indices]
    positions = np.full(len(labels), -1, dtype=int)
    positions[indices] = np.arange(len(indices))
    submatrix[indices, positions[indices]] = np.inf
    predictions = labels[indices][np.argmin(submatrix, axis=1)]
    return float(np.mean(predictions != labels))


def boundary_protection(
    distances: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Критические объекты 1NN и их ближайшие соседи того же класса"""
    nearest = np.argmin(distances, axis=1)
    critical = labels[nearest] != labels
    protected = critical.copy()

    for i in np.flatnonzero(critical):
        same_class = np.flatnonzero(labels == labels[i])
        same_class = same_class[same_class != i]
        if len(same_class) > 0:
            protected[same_class[np.argmin(distances[i, same_class])]] = True
    return critical, protected


def condense_training_set(
    X: np.ndarray,
    y: np.ndarray,
    min_class_fraction: float = 0.4,
    tolerance: float = 0.0,
    max_passes: int = 20,
) -> CondensationResult:
    """Прореживание: удаляем наименее влиятельные объекты без роста риска 1NN"""
    if not 0.0 <= min_class_fraction <= 1.0:
        raise ValueError("Доля класса должна лежать в диапазоне [0, 1]")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes, inverse = np.unique(y, return_inverse=True)
    n_objects = len(y)

    distances = pairwise_distances(X, X)
    np.fill_diagonal(distances, np.inf)

    nearest = np.argmin(distances, axis=1)
    influence = np.bincount(nearest, minlength=n_objects)
    critical, protected = boundary_protection(distances, inverse)

    counts = np.bincount(inverse, minlength=len(classes))
    minimum = np.maximum(2, np.floor(min_class_fraction * counts).astype(int))
    keep = np.ones(n_objects, dtype=bool)
    base_risk = one_nn_risk(distances, inverse, keep)

    history = [_snapshot(keep, inverse, classes, base_risk)]
    removal_order = np.argsort(influence, kind="stable")

    changed = True
    passes = 0
    while changed and passes < max_passes:
        changed = False
        passes += 1
        for index in removal_order:
            if not keep[index] or protected[index]:
                continue
            label = inverse[index]
            if counts[label] - 1 < minimum[label]:
                continue

            keep[index] = False
            risk = one_nn_risk(distances, inverse, keep)
            if risk > base_risk + tolerance:
                keep[index] = True
                continue

            counts[label] -= 1
            history.append(_snapshot(keep, inverse, classes, risk))
            changed = True

    final_risk = one_nn_risk(distances, inverse, keep)
    return CondensationResult(
        keep_mask=keep,
        base_risk=base_risk,
        final_risk=final_risk,
        influence=influence,
        critical_mask=critical,
        protected_mask=protected,
        history=pd.DataFrame(history),
    )


def _snapshot(
    keep: np.ndarray, labels: np.ndarray, classes: np.ndarray, risk: float
) -> dict:
    counts = np.bincount(labels[keep], minlength=len(classes))
    row = {"prototypes": int(keep.sum()), "risk": float(risk)}
    row.update({f"class_{label}": int(count) for label, count in zip(classes, counts)})
    return row
