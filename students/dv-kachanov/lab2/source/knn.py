import numpy as np


def euclidean_distances(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    diff = X[:, None, :] - Y[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def gaussian_kernel(r):
    return np.exp(-2.0 * r * r)


class VariableParzenKNN:
    def __init__(self, k=5, kernel=gaussian_kernel):
        self.k = k
        self.kernel = kernel

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(self.y_train)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        distances = euclidean_distances(X, self.X_train)
        return np.array([self._predict_by_distances(row) for row in distances])

    def _predict_by_distances(self, distances):
        order = np.argsort(distances)
        sorted_distances = distances[order]
        sorted_labels = self.y_train[order]

        width_index = min(self.k, len(sorted_distances) - 1)
        h = sorted_distances[width_index]

        if h <= 1e-12:
            same = sorted_labels[sorted_distances <= 1e-12]
            return self._majority_vote(same)

        weights = self.kernel(sorted_distances / h)
        votes = np.array([
            weights[sorted_labels == label].sum()
            for label in self.classes_
        ])

        return self.classes_[np.argmax(votes)]

    def _majority_vote(self, labels):
        counts = np.array([
            np.sum(labels == label)
            for label in self.classes_
        ])
        return self.classes_[np.argmax(counts)]


def loo_risk(X, y, k):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    errors = 0

    for i in range(len(X)):
        mask = np.ones(len(X), dtype=bool)
        mask[i] = False

        prediction = (
            VariableParzenKNN(k=k)
            .fit(X[mask], y[mask])
            .predict(X[i:i + 1])[0]
        )

        errors += prediction != y[i]

    return errors / len(X)


def select_k_by_loo(X, y, k_values):
    risks = []

    for k in k_values:
        risks.append(loo_risk(X, y, k))

    risks = np.array(risks, dtype=float)
    best_index = int(np.argmin(risks))

    return int(k_values[best_index]), risks
