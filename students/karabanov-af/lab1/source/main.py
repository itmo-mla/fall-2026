import os

import numpy as np

import plots
from classifier import bias_mask, empirical_risk, loss_gradient, margins, quadratic_loss, sgd
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
    print("## Обучение")
    histories, trained = {}, {}
    for label, momentum in [("SGD без инерции", 0.0), ("SGD с инерцией", 0.9)]:
        trained[label], histories[label] = sgd(X, y, w, lr=0.01, momentum=momentum, n_epochs=50)
        m = margins(trained[label], X, y)
        print(f"{label}: Q {histories[label]['risk'][0]:.3f} -> {histories[label]['risk'][-1]:.3f}, "
              f"ошибок {(margins(w, X, y) < 0).sum()} -> {(m < 0).sum()}")
        print("  веса:", np.round(trained[label], 3))

    best = trained["SGD с инерцией"]
    plots.plot_sorted_margins(margins(best, X, y), "Отступы объектов после обучения",
                              os.path.join(IMAGES, "margins_trained.png"))
    plots.plot_risk({label: h["risk"] for label, h in histories.items()},
                    "Эмпирический риск по эпохам", os.path.join(IMAGES, "risk.png"))

    best_history = histories["SGD с инерцией"]
    print(f"рекуррентная оценка Q после обучения: {best_history['Q'][-1]:.3f}, "
          f"риск по всей выборке: {best_history['risk'][-1]:.3f}")
    plots.plot_risk({"риск по всей выборке": best_history["risk"], "рекуррентная оценка Q": best_history["Q"]},
                    "Рекуррентная оценка функционала качества", os.path.join(IMAGES, "recurrent_q.png"))
    print()
    return best


def optimizer_experiment(X, y, w):
    print("## Скорейший градиентный спуск")
    histories = {}
    for label, steepest in [("постоянный шаг", False), ("скорейший спуск", True)]:
        w_opt, history = sgd(X, y, w, lr=0.01, momentum=0.9, l2=0.01, n_epochs=50, steepest=steepest)
        histories[label] = history["risk"]
        print(f"{label}: Q {history['risk'][-1]:.3f}, ошибок {(margins(w_opt, X, y) < 0).sum()}, "
              f"минимум Q за обучение {min(history['risk']):.3f}")
    plots.plot_risk(histories, "Постоянный шаг и скорейший спуск", os.path.join(IMAGES, "steepest.png"))
    print()


def l2_experiment(X, y, w):
    print("## L2-регуляризация")
    taus = np.logspace(-4, 1, 11)
    risk, errors, norm = [], [], []
    for tau in taus:
        w_tau, _ = sgd(X, y, w, lr=0.01, momentum=0.9, l2=tau, n_epochs=50)
        risk.append(empirical_risk(w_tau, X, y, tau))
        errors.append((margins(w_tau, X, y) < 0).sum())
        norm.append(np.linalg.norm(bias_mask(len(w_tau)) * w_tau))
        print(f"tau = {tau:<8.4g} Q = {risk[-1]:.3f}  ошибок = {errors[-1]:3d}  ||w|| = {norm[-1]:.3f}")
    plots.plot_l2(taus, risk, errors, norm, "Влияние L2-регуляризации", os.path.join(IMAGES, "l2.png"))
    print()


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
    l2_experiment(X, y, w)
    optimizer_experiment(X, y, w)


if __name__ == "__main__":
    main()
