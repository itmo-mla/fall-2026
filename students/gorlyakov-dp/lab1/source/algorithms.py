import numpy as np


def calculate_margin(X, y, w):
    return y * np.dot(X, w)


def plot_margins(M):
    import matplotlib.pyplot as plt

    margins = np.sort(M)
    colors = np.where(margins < 0, "#d95f02", "#1b9e77")
    plt.bar(np.arange(len(margins)), margins, color=colors)
    plt.axhline(0, color="black")
    plt.xlabel("Объекты (отсортированы по отступу)")
    plt.ylabel("Отступ")
    plt.title("Отступы объектов при корреляционной инициализации")
    plt.show()


def calculate_gradient(x, y, margin):
    return -2 * (1 - margin) * y * x


def train_classifier(X, y, w, choose_object):
    w = w.copy()
    velocity = np.zeros_like(w)

    lambd = 0.1
    h = 1e-7
    gamma = 0.9
    l2 = 0.001
    max_iter = 10000
    tol = 1e-8
    regularization_mask = np.ones_like(w)
    regularization_mask[-1] = 0

    sample_idx = np.random.choice(len(X), size=10, replace=False)
    M_sample = calculate_margin(X[sample_idx], y[sample_idx], w)
    Q = np.mean((1 - M_sample) ** 2)
    Q_history = [Q]

    for _ in range(max_iter):
        old_w = w.copy()
        old_Q = Q
        M = calculate_margin(X, y, w)
        i = choose_object(M)
        loss_i = (1 - M[i]) ** 2
        grad = calculate_gradient(X[i], y[i], M[i])

        velocity = gamma * velocity + (1 - gamma) * grad
        w = w * (1 - h * l2 * regularization_mask) - h * velocity
        Q = lambd * loss_i + (1 - lambd) * Q
        Q_history.append(Q)

        if np.linalg.norm(w - old_w) < tol and abs(Q - old_Q) < tol:
            break

    return w, Q, Q_history


def train_steepest(X, y, w, choose_object):
    w = w.copy()
    lambd = 0.1
    max_iter = 10000
    tol = 1e-8

    sample_idx = np.random.choice(len(X), size=10, replace=False)
    M_sample = calculate_margin(X[sample_idx], y[sample_idx], w)
    Q = np.mean((1 - M_sample) ** 2)
    Q_history = [Q]

    for _ in range(max_iter):
        old_w = w.copy()
        old_Q = Q
        M = calculate_margin(X, y, w)
        i = choose_object(M)
        loss_i = (1 - M[i]) ** 2
        grad = calculate_gradient(X[i], y[i], M[i])
        h_star = 1 / (2 * np.dot(X[i], X[i]))

        w = w - h_star * grad
        Q = lambd * loss_i + (1 - lambd) * Q
        Q_history.append(Q)

        if np.linalg.norm(w - old_w) < tol and abs(Q - old_Q) < tol:
            break

    return w, Q, Q_history


def choose_random(M):
    return np.random.choice(len(M))


def choose_by_margin(M):
    eps = 1e-8
    sample_weights = 1 / (np.abs(M) + eps)
    probabilities = sample_weights / np.sum(sample_weights)
    return np.random.choice(len(M), p=probabilities)


def initialize_by_correlation(X, y):
    w = np.empty(X.shape[1])

    for j in range(X.shape[1]):
        w[j] = np.dot(y, X[:, j]) / np.dot(X[:, j], X[:, j])

    return w


def initialize_random(X):
    return np.random.uniform(
        -1 / (2 * X.shape[1]),
        1 / (2 * X.shape[1]),
        size=X.shape[1],
    )


