import os

import numpy as np

import plots
from classifier import empirical_risk, loss_gradient, margins, quadratic_loss, sgd
from data import add_bias, load_binary_iris, standardize

IMAGES = os.path.join(os.path.dirname(__file__), "..", "images")


def numeric_gradient(w, x, y, eps=1e-6):
    """Central differences: (L(w + eps * e_j) - L(w - eps * e_j)) / (2 * eps) for every weight j."""
    grad = np.zeros_like(w)
    for j in range(len(w)):
        step = np.zeros_like(w)
        step[j] = eps
        grad[j] = (quadratic_loss(y * (x @ (w + step))) - quadratic_loss(y * (x @ (w - step)))) / (2 * eps)
    return grad


def margin_analysis(X, y, w):
    print("## Отступы при случайных весах")
    m = margins(w, X, y)
    print(f"отступы: min {m.min():.3f}, max {m.max():.3f}, ошибок (M < 0): {(m < 0).sum()}\n")
    plots.plot_sorted_margins(m, "Отступы объектов при случайных весах",
                              os.path.join(IMAGES, "margins_random.png"))


def gradient_check(X, y, w):
    print("## Градиент функции потерь")
    i = 0
    grad = loss_gradient(w, X[i], y[i])
    print(f"объект {i}: M = {margins(w, X[i], y[i]):.3f}, L = {quadratic_loss(margins(w, X[i], y[i])):.3f}")
    print("  аналитический градиент:", np.round(grad, 4))
    print("  численный градиент:    ", np.round(numeric_gradient(w, X[i], y[i]), 4))

    diff = max(np.abs(loss_gradient(w, X[k], y[k]) - numeric_gradient(w, X[k], y[k])).max() for k in range(len(y)))
    print(f"максимальное расхождение по всем объектам: {diff:.2e}")

    h = 0.05
    w_new = w - h * grad
    print(f"шаг против градиента (h = {h}): L = {quadratic_loss(margins(w, X[i], y[i])):.3f} -> "
          f"{quadratic_loss(margins(w_new, X[i], y[i])):.3f}\n")


def training(X, y, w):
    print("## Обучение (SGD без инерции)")
    w_trained, history = sgd(X, y, w, lr=0.01, n_epochs=50)
    m = margins(w_trained, X, y)
    print("веса после обучения:", np.round(w_trained, 3))
    print(f"Q: {history[0]:.3f} -> {history[-1]:.3f}")
    print(f"ошибок (M < 0): {(margins(w, X, y) < 0).sum()} -> {(m < 0).sum()}\n")
    plots.plot_sorted_margins(m, "Отступы объектов после обучения",
                              os.path.join(IMAGES, "margins_trained.png"))
    plots.plot_risk(history, "Эмпирический риск по эпохам", os.path.join(IMAGES, "risk.png"))
    return w_trained


def main():
    X, y = load_binary_iris()
    X = add_bias(standardize(X))
    print(f"объектов: {X.shape[0]}, признаков с bias: {X.shape[1]}")

    rng = np.random.default_rng(0)
    d = X.shape[1]
    w = rng.uniform(-1 / (2 * d), 1 / (2 * d), size=d)
    print("случайные веса:", np.round(w, 3), "\n")

    margin_analysis(X, y, w)
    gradient_check(X, y, w)
    training(X, y, w)


if __name__ == "__main__":
    main()
