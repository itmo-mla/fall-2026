import numpy as np

from knn import euclidean_distances


def initial_prototypes(X, y):
    prototypes = []

    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        centroid = X[indices].mean(axis=0)
        distances = np.sum((X[indices] - centroid) ** 2, axis=1)
        prototypes.append(indices[int(np.argmin(distances))])

    return prototypes


def classify_by_nearest_prototype(X, y, prototype_indices, object_index):
    candidates = np.array([
        index for index in prototype_indices
        if index != object_index
    ], dtype=int)

    if len(candidates) == 0:
        return y[object_index]

    distances = euclidean_distances(X[object_index:object_index + 1], X[candidates])[0]
    return y[candidates[int(np.argmin(distances))]]


def ccv_risk(X, y, prototype_indices):
    errors = 0

    for index in range(len(X)):
        prediction = classify_by_nearest_prototype(X, y, prototype_indices, index)
        errors += prediction != y[index]

    return errors / len(X)


def predict_by_nearest_prototype(X_prototypes, y_prototypes, X):
    distances = euclidean_distances(X, X_prototypes)
    nearest = np.argmin(distances, axis=1)
    return y_prototypes[nearest]


def select_prototypes(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    prototypes = initial_prototypes(X, y)
    prototype_set = set(prototypes)
    current_risk = ccv_risk(X, y, prototypes)

    while True:
        best_index = None
        best_risk = current_risk

        for index in range(len(X)):
            if index in prototype_set:
                continue

            candidate = prototypes + [index]
            risk = ccv_risk(X, y, candidate)

            if risk < best_risk:
                best_risk = risk
                best_index = index

        if best_index is None:
            break

        prototypes.append(int(best_index))
        prototype_set.add(int(best_index))
        current_risk = best_risk

    return np.array(prototypes, dtype=int)
