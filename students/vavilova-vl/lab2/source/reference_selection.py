from dataclasses import dataclass
from math import comb

import numpy as np

from knn import pairwise_distances


def ccv_weights(n_samples: int, control_size: int, tol: float = 1e-12) -> np.ndarray:
    """R(m) = C(L-1-m, l-1) / C(L-1, l) for m = 1, 2, ...; the tail below `tol` is dropped.

    L = n_samples, l = L - control_size. R(m) is the probability that, for a control
    object, its nearest training neighbor is exactly its m-th neighbor in the whole sample.
    """
    if not 1 <= control_size < n_samples:
        raise ValueError("control_size must be in [1, n_samples - 1]")
    train_size = n_samples - control_size
    total = comb(n_samples - 1, train_size)
    weights = []
    for m in range(1, control_size + 1):
        weight = comb(n_samples - 1 - m, train_size - 1) / total
        if weight < tol:
            break
        weights.append(weight)
    return np.asarray(weights)


def _neighbor_errors(
    distances: np.ndarray, y: np.ndarray, references: np.ndarray, depth: int
) -> tuple[np.ndarray, np.ndarray]:
    """Distances to the first `depth` neighbors of every object among the references
    (the object itself excluded) and whether each of those neighbors is from another class.

    Missing neighbors (fewer references than `depth`) have distance +inf and no error.
    """
    reference_distances = distances[:, references].copy()
    reference_distances[references, np.arange(len(references))] = np.inf
    order = np.argsort(reference_distances, axis=1, kind="stable")[:, :depth]
    sorted_distances = np.take_along_axis(reference_distances, order, axis=1)
    errors = (y[references][order] != y[:, None]) & np.isfinite(sorted_distances)

    padding = depth - sorted_distances.shape[1]
    if padding > 0:
        sorted_distances = np.pad(sorted_distances, ((0, 0), (0, padding)), constant_values=np.inf)
        errors = np.pad(errors, ((0, 0), (0, padding)), constant_values=False)
    return sorted_distances, errors


def compactness_profile(
    X: np.ndarray, y: np.ndarray, max_m: int, references: np.ndarray | None = None
) -> np.ndarray:
    """Pi(m, Omega): share of objects whose m-th neighbor among the references is from another class."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if references is None:
        references = np.arange(len(y))
    _, errors = _neighbor_errors(pairwise_distances(X, X), y, np.asarray(references), max_m)
    return errors.mean(axis=0)


def ccv(profile: np.ndarray, weights: np.ndarray) -> float:
    """CCV = sum_m Pi(m) R(m) for the 1NN classifier."""
    depth = min(len(profile), len(weights))
    return float(np.dot(profile[:depth], weights[:depth]))


@dataclass
class ReferenceSelection:
    references: np.ndarray
    ccv_history: list[float]


def greedy_add_references(
    X: np.ndarray, y: np.ndarray, control_size: int
) -> ReferenceSelection:
    """Greedy addition of references for 1NN by the criterion CCV(Omega) -> min.

    Omega starts with one object per class (the one closest to the class mean).
    At every step the object x minimizing CCV(Omega + {x}) is added, while CCV decreases.
    `ccv_history[j]` is CCV of the first j + len(classes) references.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = np.unique(y)
    distances = pairwise_distances(X, X)
    weights = ccv_weights(len(y), control_size)
    depth = len(weights)
    # R(m) for positions m = 1..depth + 1; a neighbor pushed past `depth` stops counting.
    weights_padded = np.append(weights, 0.0)

    references = []
    for label in classes:
        members = np.flatnonzero(y == label)
        center = X[members].mean(axis=0)
        references.append(int(members[np.argmin(np.linalg.norm(X[members] - center, axis=1))]))

    rows = np.arange(len(y))
    history = []
    while True:
        neighbor_distances, errors = _neighbor_errors(
            distances, y, np.asarray(references), depth
        )
        weighted_errors = errors * weights
        # T(x_i, Omega): contribution of every object to CCV.
        contribution = weighted_errors.sum(axis=1)
        current = float(contribution.mean())
        history.append(current)

        candidates = np.setdiff1d(rows, references)
        if not len(candidates):
            break
        candidate_distances = distances[:, candidates]
        # Position p (0-based) at which a candidate enters the neighbor list of every object.
        position = (neighbor_distances[:, :, None] < candidate_distances[:, None, :]).sum(axis=1)

        # Neighbors before p keep R(m); neighbors from p on move one place down to R(m + 1).
        before = np.concatenate(
            [np.zeros((len(y), 1)), np.cumsum(weighted_errors, axis=1)], axis=1
        )
        shifted = errors * weights_padded[1:]
        after = np.concatenate(
            [np.cumsum(shifted[:, ::-1], axis=1)[:, ::-1], np.zeros((len(y), 1))], axis=1
        )
        candidate_error = y[:, None] != y[candidates][None, :]
        new_contribution = (
            np.take_along_axis(before, position, axis=1)
            + candidate_error * weights_padded[position]
            + np.take_along_axis(after, position, axis=1)
        )
        # An object is never its own neighbor, so adding x_i does not change T(x_i).
        is_self = candidates[None, :] == rows[:, None]
        new_contribution = np.where(is_self, contribution[:, None], new_contribution)

        candidate_ccv = new_contribution.mean(axis=0)
        best = int(np.argmin(candidate_ccv))
        if candidate_ccv[best] >= current:
            break
        references.append(int(candidates[best]))

    return ReferenceSelection(np.asarray(references, dtype=int), history)
