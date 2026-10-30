import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CURVES = ["#2a78d6", "#eb6834", "#1baf7a", "#9a56c7"]


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_convergence(losses, divergences, title, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

    styles = ["-", "--", ":", "-"]
    for (name, values), color, style in zip(losses.items(), CURVES, styles):
        # Q* is the best value any of the methods reached; an exact zero gap is clipped
        # so that the logarithmic axis can still show it
        gap = np.maximum(values - min(v.min() for v in losses.values()), 1e-16)
        left.plot(np.arange(len(gap)), gap, style, marker="o", markersize=3, linewidth=2,
                  color=color, label=name)
    left.set_yscale("log")
    # the first three methods finish in ten steps, the fourth is still far away at the right edge
    left.set_xlim(-0.5, 40)
    left.set_xlabel("номер итерации")
    left.set_ylabel("Q(w) - Q(w*), логарифмическая шкала")
    left.set_title("Скорость сходимости")
    left.legend()
    left.grid(alpha=0.3)

    for (name, values), color in zip(divergences.items(), CURVES[1:]):
        right.plot(np.arange(len(values)), np.maximum(values, 1e-17), marker="o", color=color, label=name)
    right.axhline(2.2e-16, color="black", linestyle="--", linewidth=1, label="машинная точность")
    right.set_yscale("log")
    right.set_xlabel("номер итерации")
    right.set_ylabel("max|разность весов|")
    right.set_title("Расхождение с методом Ньютона-Рафсона")
    right.legend()
    right.grid(alpha=0.3)

    fig.suptitle(title)
    save(fig, path)


def plot_regularization(taus, cv_losses, test_losses, norms, conditions, title, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    best = int(np.argmin(cv_losses))

    left.plot(taus, cv_losses, marker="o", color=CURVES[0], label="кросс-валидация на обучении")
    left.plot(taus, test_losses, marker="o", color=CURVES[2], label="тест")
    left.axvline(taus[best], color="black", linestyle=":", linewidth=1.5,
                 label=f"выбор по кросс-валидации: tau = {taus[best]:.3g}")
    left.set_xscale("log")
    left.set_xlabel("коэффициент регуляризации tau")
    left.set_ylabel("log-loss на объект")
    left.set_title("Качество против регуляризации")
    left.legend()
    left.grid(alpha=0.3)

    right.plot(taus, norms, marker="o", color=CURVES[0], label="норма весов ||w||")
    right.plot(taus, conditions, marker="o", color=CURVES[1], label="число обусловленности гессиана")
    right.axvline(taus[best], color="black", linestyle=":", linewidth=1.5, label="выбранное tau")
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("коэффициент регуляризации tau")
    right.set_ylabel("логарифмическая шкала")
    right.set_title("Чем платим за отсутствие регуляризации")
    right.legend()
    right.grid(alpha=0.3)

    fig.suptitle(title)
    save(fig, path)


def plot_calibration(predicted, observed, sizes, probabilities, labels, title, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))

    left.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="идеальная калибровка")
    left.plot(predicted, observed, marker="o", color=CURVES[0],
              label="наблюдаемая частота по группам")
    for x, y_value, size in zip(predicted, observed, sizes):
        left.annotate(str(size), (x, y_value), textcoords="offset points", xytext=(0, 7), fontsize=8)
    left.set_xlabel("предсказанная вероятность класса 1")
    left.set_ylabel("доля класса 1 в группе")
    left.set_title("Калибровка вероятностей на тесте (подписи — размер группы)")
    left.legend(loc="upper left")
    left.grid(alpha=0.3)

    bins = np.linspace(0, 1, 21)
    right.hist(probabilities[labels < 0], bins=bins, color=CURVES[1], alpha=0.75, label="злокачественная")
    right.hist(probabilities[labels > 0], bins=bins, color=CURVES[0], alpha=0.75, label="доброкачественная")
    right.axvline(0.5, color="black", linestyle="--", linewidth=1, label="порог решения")
    right.set_yscale("log")
    right.set_xlabel("предсказанная вероятность класса 1")
    right.set_ylabel("число объектов, логарифмическая шкала")
    right.set_title("Распределение вероятностей по классам")
    right.legend()
    right.grid(alpha=0.3)

    fig.suptitle(title)
    save(fig, path)
