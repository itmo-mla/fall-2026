import numpy as np

from parzen_knn import ParzenKNN


def loo_risk(features, labels, k):
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels)
    if k <= 0 or k >= len(features):
        raise ValueError("k должно быть от 1 до n - 1")

    errors = 0
    for index in range(len(features)):
        mask = np.arange(len(features)) != index
        model = ParzenKNN(k=k).fit(features[mask], labels[mask])
        prediction = model.predict(features[index])[0]
        errors += prediction != labels[index]
    return errors / len(features)


def select_k_loo(features, labels, k_values):
    k_values = np.asarray(list(k_values), dtype=int)
    if k_values.ndim != 1 or len(k_values) == 0:
        raise ValueError("Нужен непустой список значений k")
    risks = np.array([loo_risk(features, labels, int(k)) for k in k_values])
    best_risk = risks.min()
    best_k = int(k_values[risks == best_risk].min())
    return best_k, risks
