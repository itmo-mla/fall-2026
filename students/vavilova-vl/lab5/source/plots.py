"""Графики для отчёта."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "lines.linewidth": 2,
})


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def convergence(histories: dict[str, list[float]], path: Path) -> None:
    """‖Q′(w_t)‖∞ по итерациям в логарифмической шкале."""
    fig, ax = plt.subplots(figsize=(7, 4))
    styles = [(BLUE, "o", "-"), (ORANGE, "s", "--"), (AQUA, "^", ":")]
    for (label, values), (color, marker, ls) in zip(histories.items(), styles):
        values = np.maximum(values, 1e-16)
        ax.semilogy(range(len(values)), values, color=color, marker=marker, ls=ls, ms=7, label=label)
    ax.set(xlabel="итерация t", ylabel="‖Q′(w_t)‖∞", title="Сходимость методов второго порядка")
    ax.legend(frameon=False)
    _save(fig, path)


def coefficients(names: list[str], ours: np.ndarray, ref: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pos = np.arange(len(names))
    ax.barh(pos, ours, color=BLUE, height=0.6, label="собственная реализация (IRLS)")
    ax.plot(ref, pos, "x", color=ORANGE, ms=10, mew=2.5, label="sklearn newton-cholesky")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set(yticks=pos, yticklabels=names, xlabel="w_j (признаки стандартизованы)",
           title="Коэффициенты логистической регрессии")
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right")
    _save(fig, path)


def irls_weights(margin: np.ndarray, weight: np.ndarray, path: Path) -> None:
    """Вес объекта (1 − σ_i)σ_i на последней итерации IRLS против отступа M_i (слайд 8)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    wrong = margin < 0
    ax.scatter(margin[~wrong], weight[~wrong], s=14, color=BLUE, alpha=0.6, label="верно классифицирован")
    ax.scatter(margin[wrong], weight[wrong], s=14, color=ORANGE, alpha=0.8, label="ошибка (M_i < 0)")
    grid = np.linspace(margin.min(), margin.max(), 300)
    s = 0.5 * (1 + np.tanh(0.5 * grid))
    ax.plot(grid, (1 - s) * s, color=MUTED, lw=1.2)
    ax.set(xlabel="отступ M_i = y_i⟨w, x_i⟩", ylabel="вес (1 − σ_i)σ_i",
           title="Веса объектов в IRLS")
    ax.legend(frameon=False)
    _save(fig, path)


def roc(fpr: np.ndarray, tpr: np.ndarray, auc: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    # Точки соединяются отрезками: группа равных score даёт диагональ, как трапеция в алгоритме AUC.
    ax.plot(fpr, tpr, color=BLUE, label=f"тест, AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC-кривая", xlim=(0, 1), ylim=(0, 1.01))
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, path)
