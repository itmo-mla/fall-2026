import numpy as np

# евклидово расстояние до объекта
def calculate_dists(x, x_query):
    sq_diff = np.square(x_query - x)
    sq_sum = np.sum(sq_diff, axis=1)
    dists = np.sqrt(sq_sum)
    return dists


def calculate_knn(x, y, x_query, k):
    dists = calculate_dists(x, x_query)

    # выбор k-ближайших
    nearest_idx = np.argsort(dists)
    knn_idx = nearest_idx[0:k]
    knn_labels = y[knn_idx]

    labels, votes = np.unique(knn_labels, return_counts=True)
    best_pos = np.argmax(votes)
    return labels[best_pos]


def calculate_parzen(x, y, x_query, k, method):
    dists = calculate_dists(x, x_query)

    nearest_idx = np.argsort(dists)
    window_width = dists[nearest_idx[k]]

    if method == "parzen":
        voter_idx = nearest_idx[0:k]
    elif method == "parzen_lecture":
        voter_idx = nearest_idx
    else:
        raise ValueError("method должен быть 'parzen' или 'parzen_lecture'")

    voter_dists = dists[voter_idx]
    voter_labels = y[voter_idx]

    weights = np.exp(-2 * (voter_dists / window_width) ** 2)
    class_scores = np.bincount(voter_labels, weights=weights, minlength=(y.max() + 1))

    pred_label = np.argmax(class_scores)
    return pred_label, class_scores


def split_leave_one_out(x, y, i):
    x_rest = x.copy()
    y_rest = y.copy()
    x_query = x_rest[i]
    y_true = y_rest[i]
    x_rest = np.delete(x_rest, i, 0)
    y_rest = np.delete(y_rest, i, 0)
    return x_rest, y_rest, x_query, y_true


def calculate_loo(x, y, k_values, method):
    loo_errors = []
    for k in k_values:
        n_errors = 0
        for i in range(0, len(x)):
            x_rest, y_rest, x_query, y_true = split_leave_one_out(x, y, i)
            if method == "knn":
                pred_label = calculate_knn(x_rest, y_rest, x_query, k)
            elif method == "parzen":
                pred_label, _ = calculate_parzen(x_rest, y_rest, x_query, k, method)
            elif method == "parzen_lecture":
                pred_label, _ = calculate_parzen(x_rest, y_rest, x_query, k, method)
            else:
                raise ValueError("method должен быть 'knn', 'parzen' или 'parzen_lecture'")

            if pred_label != y_true:
                n_errors += 1

        loo_errors.append(n_errors / len(x))

    return loo_errors


def calculate_margin(x, y, k):
    margins = []
    for i in range(0, len(x)):
        x_rest, y_rest, x_query, y_true = split_leave_one_out(x, y, i)
        _, class_scores = calculate_parzen(x_rest, y_rest, x_query, k, method="parzen")
        other_scores = np.delete(class_scores, y_true, 0)
        margins.append(class_scores[y_true] - np.max(other_scores))

    return np.array(margins)


def calculate_start_omega(margins, y):
    start_idx = []

    for c in np.unique(y):
        class_idx = np.flatnonzero(y == c)
        best_pos = np.argmax(margins[class_idx])
        start_idx.append(class_idx[best_pos])

    return np.array(start_idx)


def calculate_ccv(X, y, omega_idx):
    n_errors = 0
    for i in range(0, len(X)):
        rest_idx = omega_idx[omega_idx != i]
        pred_label = calculate_knn(X[rest_idx], y[rest_idx], X[i], 1)
        if pred_label != y[i]:
            n_errors += 1

    return n_errors / len(X)


def greedy_addition(X, y, omega_idx):
    ccv_current = calculate_ccv(X, y, omega_idx)
    history = [ccv_current]

    for _ in range(len(X)):
        candidates_ccv = []
        candidate_idx = np.setdiff1d(np.arange(len(X)), omega_idx)

        if len(candidate_idx) == 0:
            break

        for cand in candidate_idx:
            omega_with_cand = np.append(omega_idx, cand)
            ccv_with_cand = calculate_ccv(X, y, omega_with_cand)
            candidates_ccv.append(ccv_with_cand)

        best_pos = np.argmin(candidates_ccv)
        if candidates_ccv[best_pos] < ccv_current:
            omega_idx = np.append(omega_idx, candidate_idx[best_pos])
            ccv_current = candidates_ccv[best_pos]
            history.append(ccv_current)
        else:
            break

    return omega_idx, history


def calculate_best_k(k_values, loo_errors):
    # при равной ошибке берем наибольшее k
    min_error = np.min(loo_errors)
    best_k = 0
    for i in range(0, len(k_values)):
        if loo_errors[i] == min_error and k_values[i] > best_k:
            best_k = k_values[i]
    return best_k


def calculate_predict(x, y, x_test, k, method):
    pred_labels = []
    for i in range(0, len(x_test)):
        if method == "knn":
            pred_label = calculate_knn(x, y, x_test[i], k)
        else:
            pred_label, _ = calculate_parzen(x, y, x_test[i], k, method)
        pred_labels.append(pred_label)
    return np.array(pred_labels)
