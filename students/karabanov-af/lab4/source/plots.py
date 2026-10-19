import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CURVES = ["#2a78d6", "#eb6834", "#1baf7a"]


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_scree(variance, residual, slope, title, path):
    axes_numbers = np.arange(1, len(variance) + 1)
    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(16, 4.5))

    left.bar(axes_numbers, variance, color=CURVES[0], label="дисперсия компоненты")
    left.axhline(1, color="black", linestyle="--", linewidth=1, label="критерий Кайзера: дисперсия = 1")
    left.set_xlabel("номер главной компоненты")
    left.set_ylabel("дисперсия")
    left.set_title("Спектр: дисперсия по компонентам")
    left.legend()
    left.grid(alpha=0.3)

    # E(d) = 0 exactly, a logarithmic axis cannot show it
    middle.plot(axes_numbers[:-1], residual[:-1], marker="o", color=CURVES[0],
                label="потерянная доля нормы E(m)")
    for eps, color in zip([0.05, 0.01], [CURVES[1], CURVES[2]]):
        middle.axhline(eps, color=color, linestyle="--", linewidth=1,
                       label=f"порог {eps:.0%}: размерность {np.searchsorted(-residual, -eps) + 1}")
    middle.set_yscale("log")
    middle.set_xlabel("число оставленных компонент m")
    middle.set_ylabel("E(m), логарифмическая шкала")
    middle.set_title("Эффективная размерность по порогу")
    middle.legend()
    middle.grid(alpha=0.3)

    break_point = int(np.argmax(slope > 2)) + 1
    right.bar(np.arange(1, len(slope) + 1), slope, color=CURVES[0], label="отношение E(m-1) / E(m)")
    right.bar(break_point, slope[break_point - 1], color=CURVES[1],
              label=f"склон ломается на m = {break_point}")
    right.axhline(2, color="black", linestyle="--", linewidth=1, label="двукратное падение потери")
    right.set_xlabel("число оставленных компонент m")
    right.set_ylabel("во сколько раз упала потеря")
    right.set_title("Критерий крутого склона")
    right.legend()
    right.grid(alpha=0.3)

    fig.suptitle(title)
    save(fig, path)


def plot_regression(components, pcr_cv, pcr_test, alphas, ridge_cv, ridge_test, ols_test, title, path):
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    best_k, best_alpha = int(np.argmax(pcr_cv)), int(np.argmax(ridge_cv))

    left.plot(components, pcr_cv, marker="o", color=CURVES[0], label="кросс-валидация на обучении")
    left.plot(components, pcr_test, marker="o", color=CURVES[2], label="тест")
    left.axhline(ols_test, color=CURVES[1], linestyle="--", linewidth=1.5,
                 label=f"МНК на тесте, R² = {ols_test:.3f}")
    left.axvline(components[best_k], color="black", linestyle=":", linewidth=1.5,
                 label=f"выбор по кросс-валидации: k = {components[best_k]}")
    left.set_xlabel("число оставленных компонент k")
    left.set_ylabel("R²")
    left.set_title("Качество против числа компонент")
    left.legend(loc="lower right")
    left.grid(alpha=0.3)

    right.plot(alphas, ridge_cv, marker="o", markersize=3, color=CURVES[0],
               label="кросс-валидация на обучении")
    right.plot(alphas, ridge_test, marker="o", markersize=3, color=CURVES[2], label="тест")
    right.axhline(ols_test, color=CURVES[1], linestyle="--", linewidth=1.5, label="МНК на тесте")
    right.axvline(alphas[best_alpha], color="black", linestyle=":", linewidth=1.5,
                  label=f"выбор по кросс-валидации: alpha = {alphas[best_alpha]:.3g}")
    right.set_xscale("log")
    right.set_xlabel("коэффициент регуляризации alpha")
    right.set_ylabel("R²")
    right.set_title("Качество против регуляризации")
    right.legend(loc="lower left")
    right.grid(alpha=0.3)

    fig.suptitle(title)
    save(fig, path)
