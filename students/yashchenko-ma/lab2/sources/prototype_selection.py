"""
Собственная реализация алгоритма отбора эталонов (пункт 6 задания):
жадный отбор по критерию CCV -> min.

Точный алгоритм из лекции ("Жадная стратегия добавления эталонов", слайд 26):

    Omega := {по одному объекту от каждого класса}
    повторять:
        найти x из X^L, не входящий в Omega: CCV(Omega + {x}) -> min
        Omega := Omega + {x}
    пока CCV уменьшается

Точная комбинаторная формула критерия (слайд 25) требует биномиальных
коэффициентов. Лекция там же (слайд 24) показывает, что при длине контроля
k=1, CCV(Omega) = LOO(Omega), и отдельно отмечает, что "CCV слабо зависит
от длины контроля k" -- поэтому в качестве практического критерия здесь
используется доля ошибок классификатора 1NN(Omega) по всей выборке
(в LOO-постановке, без утечки): это в точности CCV(Omega) при k=1,
применение упрощения, обоснованного самой лекцией.

Реализовано самостоятельно, только на numpy.
"""
import numpy as np


def _pairwise_dist(X):
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.maximum((diff ** 2).sum(axis=2), 0.0))


'''
Here is весь жадный алгоритм целиком: инициализация Omega (по одному вину на класс, ближайшему к центроиду), 
затем цикл for _ in range(max_iter), который на каждой итерации перебирает кандидатов и 
добавляет того, кто сильнее всего снижает число ошибок (векторизованно через Dc = D[:, candidates] и mask), 
пока критерий не перестанет улучшаться. Запускается эта функция из sources/run_prototype_selection.py 
(в самом начале main()), результат (индексы omega и история сходимости history) 
сразу используется там же для визуализации (PCA-график) и для сравнения качества до/после отбора.
'''
def greedy_prototype_selection(X, y, max_iter=None, verbose=False):
    """Жадный отбор эталонов Omega по критерию CCV(Omega) -> min.

    Возвращает (omega, history), где omega - индексы отобранных эталонов
    в исходном X, history - число ошибок 1NN(Omega) на каждой итерации."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    L = len(y)

    D = _pairwise_dist(X)
    np.fill_diagonal(D, np.inf)  # объект не может быть сам себе соседом (LOO)

    # Omega := по одному объекту от каждого класса.
    # Лекция не уточняет способ выбора этого "одного объекта"; берём
    # ближайший к центроиду класса как разумного представителя.
    omega = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        centroid = X[idx].mean(axis=0)
        d_to_centroid = np.linalg.norm(X[idx] - centroid, axis=1)
        omega.append(int(idx[np.argmin(d_to_centroid)]))

    best_dist = np.full(L, np.inf)
    best_label = np.full(L, -1)
    for o in omega:
        mask = D[:, o] < best_dist
        best_dist = np.where(mask, D[:, o], best_dist)
        best_label = np.where(mask, y[o], best_label)
    correct = (best_label == y)
    total_errors = int(L - correct.sum())

    history = [total_errors]
    max_iter = max_iter or L
    for _ in range(max_iter):
        candidates = np.setdiff1d(np.arange(L), omega)
        if len(candidates) == 0:
            break
        Dc = D[:, candidates]                          # L x n_candidates
        mask = Dc < best_dist[:, None]                  # где кандидат ближе текущего эталона
        y_cand = y[candidates]
        correct_if_added = np.where(mask, y[:, None] == y_cand[None, :], correct[:, None])
        errors_per_candidate = L - correct_if_added.sum(axis=0)
        best_idx = int(np.argmin(errors_per_candidate))
        best_error = int(errors_per_candidate[best_idx])
        if best_error >= total_errors:
            break                                       # CCV(Omega) больше не уменьшается
        best_c = int(candidates[best_idx])
        mask_c = D[:, best_c] < best_dist
        best_dist = np.where(mask_c, D[:, best_c], best_dist)
        best_label = np.where(mask_c, y[best_c], best_label)
        correct = (best_label == y)
        total_errors = best_error
        omega.append(best_c)
        history.append(total_errors)
        if verbose:
            print(f"Добавлен эталон #{len(omega)}: объект {best_c}, "
                  f"ошибок={total_errors} ({total_errors/L:.4f})")

    return np.array(omega), np.array(history)
