"""
Подбор параметра k методом скользящего контроля LOO (пункт 3 задания).

Формула из лекции (слайд 8):
    LOO(k, X^l) = sum_{i=1}^{l} [a(x_i; X^l \\ {x_i}, k) != y_i] -> min_k
    LOO - leave-one-out

Эмпирический риск -- доля (0/1-функция потерь) ошибок классификации.
Реализовано самостоятельно на numpy, без sklearn.model_selection.
"""
import numpy as np
from sklearn.metrics import f1_score  # только вычисление метрики, не алгоритм

from knn_parzen import ParzenKNN


def loo_risk_curve(X, y, k_values):
    """Считает LOO-риск (0/1) и LOO-F1 (по классу 1) для каждого k из k_values.

    Матрица попарных расстояний считается один раз и переиспользуется для
    всех k (иначе пришлось бы её пересчитывать 40 раз). Диагональ выставлена
    в +inf, чтобы объект не мог быть сам себе соседом - это и реализует
    LOO без явного цикла "исключить объект i, переобучить, проверить".
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt(np.maximum((diff ** 2).sum(axis=2), 0.0))
    np.fill_diagonal(D, np.inf)
    '''
    np.fill_diagonal(D, np.inf) ->
    and we get
          x1    x2    x3
    x1    inf   1.2   3.5
    x2    1.2   inf   2.1
    x3    3.5   2.1   inf
    thus now Теперь при классификации x_i он физически присутствует в X, 
                но его расстояние до самого себя = ∞, поэтому predict_from_dists() не сможет выбрать его в качестве соседа.
    and now this preds = model.predict_from_dists(D, k) below for each xi does обучение на остальных объектах, проверка на x_i.
    '''

    model = ParzenKNN().fit(X, y)
    risks, f1s = [], []
    for k in k_values:
        preds = model.predict_from_dists(D, k) 
        risks.append(np.mean(preds != y))
        f1s.append(f1_score(y, preds, pos_label=1, zero_division=0))
    return np.array(risks), np.array(f1s)


def best_k(X, y, k_values, criterion="risk"):
    risks, f1s = loo_risk_curve(X, y, k_values)
    if criterion == "risk":
        best_idx = int(np.argmin(risks))
    elif criterion == "f1":
        best_idx = int(np.argmax(f1s))
    else:
        raise ValueError(criterion)
    return k_values[best_idx], risks, f1s
