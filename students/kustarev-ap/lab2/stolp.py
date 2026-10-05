import numpy as np

from knn import class_scores, weighted_vote, gaussian_kernel


def margins(X, y, k):
    dists = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dists, np.inf)
    classes = np.unique(y)
    scores = class_scores(dists, y, k, classes)

    idx = np.searchsorted(classes, y)
    own = scores[np.arange(len(y)), idx].copy()
    scores[np.arange(len(y)), idx] = -np.inf
    other = scores.max(axis=1)

    return own - other


def ccv_errors(X, y, ref_idx, k):
    ref_idx = list(ref_idx)
    ref_pos = {idx: pos for pos, idx in enumerate(ref_idx)}
    dists = np.sqrt(((X[:, None, :] - X[ref_idx][None, :, :]) ** 2).sum(axis=2))
    for row in range(len(X)):
        pos = ref_pos.get(row)
        if pos is not None:
            dists[row, pos] = np.inf

    classes = np.unique(y)
    pred = weighted_vote(dists, y[ref_idx], min(k, len(ref_idx) - 1), classes)
    return (pred != y).sum()


def greedy_add(X, y, k):
    # жадное добавление эталонов: старт с одного объекта на класс (с максимальным
    # отступом), дальше на каждом шаге добавляется объект, с которым CCV минимален
    classes = np.unique(y)
    m = margins(X, y, k)

    selected = [int(np.where(y == c)[0][np.argmax(m[y == c])]) for c in classes]
    current = ccv_errors(X, y, selected, k)
    history = [current]

    while True:
        best_c, best_ccv = None, current
        for c in range(len(X)):
            if c in selected:
                continue
            trial_ccv = ccv_errors(X, y, selected + [c], k)
            if trial_ccv < best_ccv:
                best_c, best_ccv = c, trial_ccv

        if best_c is None:
            break
        selected.append(best_c)
        current = best_ccv
        history.append(current)

    return np.array(selected), np.array(history)


def _dist_to_selected(X, selected):
    D = np.sqrt(((X[:, None, :] - X[selected][None, :, :]) ** 2).sum(axis=2))
    for j, idx in enumerate(selected):
        D[idx, j] = np.inf  # объект не считает себя своим соседом
    return D


def _best_removal(D, y, selected, classes):
    # k=1: ширина окна h = расстояние до ближайшего эталона. Для кандидата на
    # удаление пересчет нужен только тем объектам, у кого этот кандидат был
    # ближайшим соседом - остальным достаточно вычесть его вклад из суммы весов
    order = np.argsort(D, axis=1)
    nn_idx = order[:, 0]
    second_idx = order[:, 1]
    rows = np.arange(D.shape[0])
    nn_dist = D[rows, nn_idx]
    second_dist = D[rows, second_idx]

    sel_labels = y[selected]
    onehot_sel = (sel_labels[:, None] == classes[None, :]).astype(float)

    W = gaussian_kernel(D / nn_dist[:, None])
    base_scores = W @ onehot_sel
    current = int((classes[np.argmax(base_scores, axis=1)] != y).sum())

    m = D.shape[1]
    best_j, best_err = None, current

    for j0 in range(m):
        bucket = nn_idx == j0
        not_bucket = ~bucket

        scores_unaff = base_scores[not_bucket] - W[not_bucket, j0][:, None] * onehot_sel[j0][None, :]
        err_unaff = (classes[np.argmax(scores_unaff, axis=1)] != y[not_bucket]).sum()

        err_aff = 0
        if bucket.any():
            h_new = second_dist[bucket]
            W_new = gaussian_kernel(D[bucket] / h_new[:, None])
            W_new[:, j0] = 0.0
            scores_aff = W_new @ onehot_sel
            err_aff = (classes[np.argmax(scores_aff, axis=1)] != y[bucket]).sum()

        total_err = err_unaff + err_aff
        if total_err < best_err:
            best_j, best_err = j0, total_err

    return best_j, best_err, current


def _greedy_remove_brute(X, y, k):
    # честный перебор: на каждом шаге для каждого кандидата пересчитывается CCV
    # заново. Годится для небольших выборок, на сотнях-тысячах объектов слишком
    # медленно - тогда нужна векторизованная версия (см. _best_removal, k=1)
    classes = np.unique(y)
    selected = list(range(len(X)))
    current = ccv_errors(X, y, selected, k)
    history = [current]

    while len(selected) > len(classes):
        best_drop, best_ccv = None, current
        for c in selected:
            trial = [s for s in selected if s != c]
            trial_ccv = ccv_errors(X, y, trial, k)
            if trial_ccv < best_ccv:
                best_drop, best_ccv = c, trial_ccv

        if best_drop is None:
            break
        selected.remove(best_drop)
        current = best_ccv
        history.append(current)

    return np.array(selected), np.array(history)


def greedy_remove(X, y, k):
    # жадное удаление не-эталонов: старт со всей выборки, на каждом шаге удаляется
    # объект, без которого CCV минимален. Честный перебор на большой выборке
    # (O(N*m^2) за шаг) занимает часы, поэтому для k=1 используется векторизованная
    # версия (O(N*m) за шаг), для k>1 - честный перебор (годится на небольших train)
    if k != 1:
        return _greedy_remove_brute(X, y, k)

    classes = np.unique(y)
    selected = list(range(len(X)))
    history = []

    while len(selected) > len(classes):
        D = _dist_to_selected(X, selected)
        best_j, best_err, current = _best_removal(D, y, selected, classes)
        if not history:
            history.append(current)

        if best_j is None:
            break
        del selected[best_j]
        history.append(best_err)

    return np.array(selected), np.array(history)
