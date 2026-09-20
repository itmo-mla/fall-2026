import os

import numpy as np

import plots
from classifier import (bias_mask, correlation_weights, empirical_risk, loss_gradient, margins,
                        multistart, predict, quadratic_loss, random_weights, sgd)
from data import add_bias, load_binary_iris, standardize, train_test_split
from metrics import accuracy, confusion_matrix, f1, precision, recall

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


def sampling_experiment(X, y, w):
    print("## Предъявление объектов по модулю отступа")
    histories = {}
    for label, sampling in [("случайное предъявление", "random"), ("по модулю отступа", "margin")]:
        w_s, history = sgd(X, y, w, lr=0.01, momentum=0.9, l2=0.01, n_epochs=50, sampling=sampling)
        histories[label] = history["risk"]
        m = margins(w_s, X, y)
        print(f"{label}: Q {history['risk'][-1]:.3f}, ошибок {(m < 0).sum()}, "
              f"пограничных |M| < 0.3: {(np.abs(m) < 0.3).sum()}")
    plots.plot_risk(histories, "Порядок предъявления объектов", os.path.join(IMAGES, "sampling.png"))
    print()


def init_experiment(X, y, w):
    print("## Инициализация весов")
    params = dict(lr=0.01, momentum=0.9, l2=0.01, n_epochs=50)
    histories = {}

    w_corr = correlation_weights(X, y)
    print("веса через корреляцию:", np.round(w_corr, 3))
    for label, w0 in [("случайные веса", w), ("через корреляцию", w_corr)]:
        w_init, history = sgd(X, y, w0, **params)
        histories[label] = history["risk"]
        print(f"{label}: Q до обучения {history['risk'][0]:.3f} -> {history['risk'][-1]:.3f}, "
              f"ошибок {(margins(w_init, X, y) < 0).sum()}")
    plots.plot_risk(histories, "Инициализация весов", os.path.join(IMAGES, "init.png"))

    best, runs = multistart(X, y, n_starts=10, **params)
    finals = [run[1]["risk"][-1] for run in runs]
    print(f"мультистарт (10 запусков): Q от {min(finals):.4f} до {max(finals):.4f}, "
          f"ошибок у лучшего {(margins(best[0], X, y) < 0).sum()}")
    plots.plot_multistart(runs, best, "Мультистарт: 10 случайных инициализаций",
                          os.path.join(IMAGES, "multistart.png"))
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


def evaluation(X_train, X_test, y_train, y_test, w):
    print("## Качество классификации на отложенной выборке")
    w_trained, _ = sgd(X_train, y_train, w, lr=0.01, momentum=0.9, l2=0.01, n_epochs=50)
    for name, X_part, y_part in [("обучение", X_train, y_train), ("тест", X_test, y_test)]:
        y_pred = predict(w_trained, X_part)
        print(f"{name}: accuracy {accuracy(y_part, y_pred):.3f}, precision {precision(y_part, y_pred):.3f}, "
              f"recall {recall(y_part, y_pred):.3f}, f1 {f1(y_part, y_pred):.3f}")
    cm = confusion_matrix(y_test, predict(w_trained, X_test))
    print("матрица ошибок на тесте (строки — истинный класс, столбцы — предсказанный):")
    print(f"  versicolor: {cm[0]}")
    print(f"  virginica:  {cm[1]}\n")
    return w_trained


def main():
    X, y = load_binary_iris()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, seed=42)
    X_test = add_bias(standardize(X_test, reference=X_train))
    X_train = add_bias(standardize(X_train))
    print(f"обучение: {X_train.shape}, тест: {X_test.shape}")

    rng = np.random.default_rng(0)
    d = X_train.shape[1]
    w = random_weights(rng, d)
    print("случайные веса:", np.round(w, 3), "\n")

    margin_analysis(X_train, y_train, w)
    gradient_check(X_train, y_train, w)
    training(X_train, y_train, w)
    l2_experiment(X_train, y_train, w)
    optimizer_experiment(X_train, y_train, w)
    sampling_experiment(X_train, y_train, w)
    init_experiment(X_train, y_train, w)
    evaluation(X_train, X_test, y_train, y_test, w)


if __name__ == "__main__":
    main()
