import numpy as np


def calculate_distances(X, x):
    return np.linalg.norm(X - x, axis=1)


def predict_knn(X_train, y_train, X, k):
    if not 1 <= k <= len(X_train):
        raise ValueError("k must be between 1 and the number of training objects")

    predictions = []

    for x in X:
        distances = calculate_distances(X_train, x)
        nearest = np.argsort(distances)[:k]
        labels, counts = np.unique(y_train[nearest], return_counts=True)
        predictions.append(labels[np.argmax(counts)])

    return np.array(predictions)


def predict_parzen(X_train, y_train, X, k):
    if not 1 <= k < len(X_train):
        raise ValueError(
            "k must be between 1 and the number of training objects minus 1"
        )

    predictions = []

    for x in X:
        distances = calculate_distances(X_train, x)
        predictions.append(predict_parzen_from_distances(distances, y_train, k))

    return np.array(predictions)


def predict_parzen_from_distances(distances, y_train, k):
    classes = np.unique(y_train)
    # already k+1 (indexes from 0)
    h = np.partition(distances, k)[k]

    if h == 0:
        weights = (distances == 0).astype(float)
    else:
        weights = gaussian_kernel(distances / h)

    scores = np.array([np.sum(weights[y_train == cls]) for cls in classes])
    return choose_class(classes, scores, distances, y_train)


def choose_class(classes, scores, distances, y_train):
    winners = classes[scores == np.max(scores)]
    if len(winners) == 1:
        return winners[0]

    nearest_distances = [np.min(distances[y_train == cls]) for cls in winners]
    return winners[np.argmin(nearest_distances)]


def gaussian_kernel(r):
    return np.exp(-2 * r**2)


def loo(X, y, ks):
    ks = np.asarray(ks)
    if ks.size == 0 or np.any((ks < 1) | (ks > len(X) - 2)):
        raise ValueError("k must be between 1 and the number of objects minus 2")

    distances = []
    for i, x in enumerate(X):
        train = np.delete(X, i, axis=0)
        std = np.std(train, axis=0)
        std[std == 0] = 1
        distances.append(np.linalg.norm((X - x) / std, axis=1))
    distances = np.array(distances)
    np.fill_diagonal(distances, np.inf)

    risks = []
    for k in ks:
        predictions = np.array(
            [predict_parzen_from_distances(row, y, k) for row in distances]
        )
        risks.append(np.mean(predictions != y))

    return np.array(risks)


def select_prototypes(X, y):
    if len(X) < 2:
        raise ValueError("At least two objects are required for prototype selection")

    distances = np.array([calculate_distances(X, x) for x in X])
    np.fill_diagonal(distances, np.inf)

    prototypes = np.arange(len(X))
    classes = np.unique(y)
    history = []

    while True:
        current = distances[:, prototypes].copy()
        nearest = np.argmin(current, axis=1)
        wrong = y[prototypes[nearest]] != y
        errors = np.sum(wrong)
        history.append((len(prototypes), errors / len(X)))

        if len(prototypes) <= max(2, len(classes)):
            break

        current[np.arange(len(X)), nearest] = np.inf
        second = np.argmin(current, axis=1)
        wrong_second = y[prototypes[second]] != y
        candidates = errors + np.bincount(
            nearest,
            weights=wrong_second.astype(int) - wrong.astype(int),
            minlength=len(prototypes),
        )

        for cls in classes:
            positions = np.flatnonzero(y[prototypes] == cls)
            if len(positions) == 1:
                candidates[positions[0]] = np.inf

        best = np.argmin(candidates)
        if candidates[best] > errors:
            break

        prototypes = np.delete(prototypes, best)

    return prototypes, np.array(history)
