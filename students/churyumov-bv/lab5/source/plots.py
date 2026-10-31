import numpy as np
import matplotlib.pyplot as plt

import logreg

FLOOR = 1e-17  # на логарифмической шкале ноль не нарисовать


def plot_convergence(hists, gd_loss, q_star, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {"Ньютон–Рафсон": ("o-", "tab:blue"), "IRLS (слайд 9)": ("s--", "tab:orange"), "IRLS для GLM (слайд 19)": ("^:", "tab:green")}
    for name, h in hists.items():
        fmt, color = styles[name]
        t = np.arange(len(h.loss))
        a1.semilogy(t, np.maximum(np.array(h.loss) - q_star, FLOOR), fmt, color=color, label=name, ms=5)
        a2.semilogy(t, np.maximum(h.grad_norm, FLOOR), fmt, color=color, label=name, ms=5)
    a1.semilogy(np.arange(len(gd_loss)), np.maximum(gd_loss - q_star, FLOOR), "-", color="tab:gray", label="градиентный спуск, шаг 1/L")
    a1.set_xlim(-0.5, 60)
    a1.set_xlabel("итерация t")
    a1.set_ylabel("Q(wᵗ) − Q(w*)")
    a1.set_title("Невязка функционала")
    a2.set_xlabel("итерация t")
    a2.set_ylabel("maxⱼ |∂Q/∂wⱼ|")
    a2.set_title("Норма градиента")
    a1.legend()
    for ax in (a1, a2):
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_weights(F, y, w_start, w_star, path):
    # слайд 8: вес объекта (1 - sigma) sigma и модуль модифицированного ответа |y~| = e^{-M/2}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    grid = np.linspace(-8, 8, 400)
    s = logreg.sigmoid(grid)
    for w, label, color, marker in ((w_start, "w⁰ (МНК)", "tab:red", "x"), (w_star, "w* (сходимость)", "tab:blue", "o")):
        M = logreg.margins(F, y, w)
        a1.scatter(M, logreg.sigmoid(M) * (1 - logreg.sigmoid(M)), s=14, alpha=0.5, color=color, marker=marker, label=label)
        a2.scatter(M, np.exp(-M / 2), s=14, alpha=0.5, color=color, marker=marker, label=label)
    a1.plot(grid, s * (1 - s), "k-", lw=1, label="σ(M)(1 − σ(M))")
    a2.plot(grid, np.exp(-grid / 2), "k-", lw=1, label="e^(−M/2)")
    a1.set_ylabel("вес γᵢ² = (1 − σᵢ)σᵢ")
    a1.set_title("Веса объектов")
    a2.set_ylabel("|ỹᵢ| = e^(−Mᵢ/2)")
    a2.set_title("Модифицированные ответы")
    a2.set_yscale("log")
    for ax in (a1, a2):
        ax.axvline(0, color="gray", ls=":", lw=1)
        ax.set_xlim(-8, 8)
        ax.set_xlabel("отступ Mᵢ = yᵢ wᵀxᵢ")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_equivalence(names, w_own, w_ref, p_own, p_ref, path):
    j = np.arange(len(names))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    a1.bar(j - 0.2, w_ref, width=0.4, label="sklearn (newton-cg)")
    a1.bar(j + 0.2, w_own, width=0.4, label="своя реализация")
    a1.set_xticks(j, names, rotation=30)
    a1.set_ylabel("вес wⱼ")
    a1.set_title("Коэффициенты")
    a1.legend()
    a2.plot([0, 1], [0, 1], "k-", lw=1)
    a2.plot(p_ref, p_own, "o", ms=4, alpha=0.6, color="tab:blue")
    a2.set_xlabel("P(y = +1 | x), sklearn")
    a2.set_ylabel("P(y = +1 | x), своя реализация")
    a2.set_title("Вероятности на тестовой выборке")
    for ax in (a1, a2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_separable(h_magic, h_sep, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for h, label, color in ((h_magic, "MAGIC (неразделима)", "tab:blue"), (h_sep, "Breast Cancer (разделима)", "tab:red")):
        t = np.arange(len(h.w))
        a1.plot(t, [np.linalg.norm(w) for w in h.w], "o-", color=color, label=label, ms=4)
        a2.semilogy(t, np.maximum(h.loss, FLOOR), "o-", color=color, label=label, ms=4)
    a1.set_ylabel("‖wᵗ‖")
    a1.set_title("Норма вектора весов")
    a2.set_ylabel("Q(wᵗ)")
    a2.set_title("Функционал")
    for ax in (a1, a2):
        ax.set_xlabel("итерация t")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ols(p_log, p_ols, y01, path):
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(p_log[y01 == 0], p_ols[y01 == 0], s=14, alpha=0.6, color="tab:blue", label="класс 0")
    ax.scatter(p_log[y01 == 1], p_ols[y01 == 1], s=14, alpha=0.6, color="tab:red", label="класс 1")
    ax.axhspan(-1, 0, color="gray", alpha=0.15)
    ax.axhspan(1, 2, color="gray", alpha=0.15)
    ax.set_ylim(min(-0.3, p_ols.min() - 0.05), max(1.3, p_ols.max() + 0.05))
    ax.set_xlabel("σ(wᵀx), логистическая регрессия")
    ax.set_ylabel("wᵀx, МНК на ответах {0, 1}")
    ax.set_title("Предсказания МНК выходят за [0, 1]")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
