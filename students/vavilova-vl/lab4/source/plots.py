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


def spectrum(eigenvalues: np.ndarray, E: np.ndarray, eps: float, m_eff: int, path: Path) -> None:
    """Слева — спектр FᵀF, справа — доля потерянной дисперсии E_m."""
    n = len(eigenvalues)
    ks = np.arange(1, n + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.bar(ks, eigenvalues, color=BLUE, width=0.6)
    ax1.set(xlabel="номер компоненты j", ylabel="λ_j", title="Собственные значения FᵀF", xticks=ks)

    ms = np.arange(0, n + 1)
    ax2.plot(ms, E, color=BLUE, marker="o", ms=6)
    ax2.axhline(eps, color=ORANGE, ls="--", lw=1.5, label=f"ε = {eps}")
    ax2.plot([m_eff], [E[m_eff]], "o", ms=11, mfc="none", mec=ORANGE, mew=2)
    ax2.annotate(f"m = {m_eff}", (m_eff, E[m_eff]), xytext=(10, 18), textcoords="offset points", color=INK)
    ax2.set(xlabel="число компонент m", ylabel="E_m", title="Доля потерянной дисперсии", xticks=ms)
    ax2.legend(frameon=False)
    _save(fig, path)


def projection(G: np.ndarray, y: np.ndarray, ratio: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sc = ax.scatter(G[:, 0], G[:, 1], c=y, cmap="Blues", s=22, edgecolors=MUTED, linewidths=0.3)
    fig.colorbar(sc, ax=ax, label="y = log(1 + Fare)")
    ax.set(xlabel=f"PC1 ({ratio[0]:.1%} дисперсии)", ylabel=f"PC2 ({ratio[1]:.1%} дисперсии)",
           title="Проекция выборки на первые две компоненты")
    _save(fig, path)


def loadings(components: np.ndarray, feature_names: list[str], path: Path) -> None:
    n = components.shape[0]
    lim = np.abs(components).max()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(components.T, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set(xticks=range(n), xticklabels=[f"PC{j + 1}" for j in range(n)],
           yticks=range(len(feature_names)), yticklabels=feature_names,
           title="Главные направления (столбцы U)")
    ax.grid(False)
    for i in range(components.shape[1]):
        for j in range(n):
            v = components[j, i]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > 0.6 * lim else INK)
    fig.colorbar(im, ax=ax)
    _save(fig, path)


def reconstruction(ours: np.ndarray, ref: np.ndarray, E: np.ndarray, path: Path) -> None:
    ms = np.arange(1, len(ours) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ms, E[1:], color=MUTED, lw=6, alpha=0.35, label="E_m по спектру")
    ax.plot(ms, ours, color=BLUE, marker="o", ms=8, label="собственный PCA")
    ax.plot(ms, ref, color=ORANGE, marker="x", ms=8, ls="none", mew=2, label="sklearn PCA")
    ax.set(xlabel="число компонент m", ylabel="‖F − F̂‖² / ‖F‖²",
           title="Относительная ошибка восстановления", xticks=ms)
    ax.legend(frameon=False)
    _save(fig, path)


def pcr(rmse: np.ndarray, m_eff: int, path: Path) -> None:
    ms = np.arange(1, len(rmse) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ms, rmse, color=BLUE, marker="o", ms=7)
    ax.axvline(m_eff, color=ORANGE, ls="--", lw=1.5, label=f"эффективная размерность m = {m_eff}")
    best = int(np.argmin(rmse))
    ax.annotate(f"min RMSE = {rmse[best]:.2f}\n(m = {best + 1})", (best + 1, rmse[best]),
                xytext=(best + 1 - 1.2, rmse[best] + 0.35 * (rmse.max() - rmse.min())),
                textcoords="data", ha="right", color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set(xlabel="число компонент m", ylabel="RMSE (5-fold CV)",
           title="Регрессия на главных компонентах", xticks=ms)
    ax.legend(frameon=False)
    _save(fig, path)
