from functools import partial

import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from model import calculate_best_k

# для каких k LOO своих функций сверяется со sklearn, больше брать долго
K_CHECK = list(range(1, 51))


# Готового окна Парзена переменной ширины в sklearn нет. Можно через KNeighborsClassifier
def sklearn_parzen_weights(dists):
    window_width = dists[:, -1:]
    weights = np.exp(-2 * (dists / window_width) ** 2)
    weights[:, -1] = 0
    return weights

def sklearn_parzen_lecture_weights(dists, k):
    window_width = dists[:, k:k + 1]
    weights = np.exp(-2 * (dists / window_width) ** 2)
    return weights


# эталонные модели строятся по k и n_train, числу объектов в обучающей выборке
# uniform это обычный KNN, distance это штатное взвешивание sklearn с весом 1/расстояние
SKLEARN_MODELS = {
    "knn": lambda k, n_train: KNeighborsClassifier(n_neighbors=k),
    "knn_distance": lambda k, n_train: KNeighborsClassifier(n_neighbors=k, weights="distance"),
    "parzen": lambda k, n_train: KNeighborsClassifier(n_neighbors=k + 1, weights=sklearn_parzen_weights),
    "parzen_lecture": lambda k, n_train: KNeighborsClassifier(n_neighbors=n_train, weights=partial(sklearn_parzen_lecture_weights, k=k)),
}


def run_sklearn_baseline(X_train, y_train, X_test):
    results = {}
    for name in SKLEARN_MODELS:
        make_model = SKLEARN_MODELS[name]

        # в LOO обучаемся на всех объектах, кроме одного
        loo_errors = []
        for k in K_CHECK:
            model = make_model(k, len(X_train) - 1)
            accuracy = cross_val_score(model, X_train, y_train, cv=LeaveOneOut(), n_jobs=-1).mean()
            loo_errors.append(1 - accuracy)

        # k выбирается тем же правилом, что и для своих моделей
        best_k = calculate_best_k(K_CHECK, loo_errors)
        model = make_model(best_k, len(X_train))
        model.fit(X_train, y_train)
        results[name] = {"loo": np.array(loo_errors), "k": best_k, "pred": model.predict(X_test)}

    return results
