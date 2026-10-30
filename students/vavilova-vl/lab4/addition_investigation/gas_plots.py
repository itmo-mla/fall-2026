"""Графики дополнительного исследования (оформление — как в plots.py лабораторной)."""

from pathlib import Path

import numpy as np
from plots import AQUA, BLUE, GRID, INK, MUTED, ORANGE, _save, plt


def correlation(X: np.ndarray, names: list[str], path: Path) -> None:
    C = np.corrcoef(X, rowvar=False)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set(xticks=range(len(names)), yticks=range(len(names)), title="Корреляции признаков")
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticklabels(names)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(C[i, j]) > 0.7 else INK)
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, path)


def projection(G: np.ndarray, y: np.ndarray, groups: np.ndarray, ratio: np.ndarray, path: Path) -> None:
    """Слева цвет — дебит газа, справа — серия испытаний (линии соединяют режимы одной скважины)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    sc = ax1.scatter(G[:, 0], G[:, 1], c=y, cmap="Blues", s=30, edgecolors=MUTED, linewidths=0.4)
    fig.colorbar(sc, ax=ax1, label="дебит газа, тыс. м³/сут")
    ax1.set(title="Цвет — дебит газа")

    cmap = plt.get_cmap("tab20")
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        idx = idx[np.argsort(G[idx, 0])]
        ax2.plot(G[idx, 0], G[idx, 1], "-o", color=cmap(g % 20), ms=4, lw=1)
    ax2.set(title="Цвет — серия испытаний")
    for ax in (ax1, ax2):
        ax.set(xlabel=f"PC1 ({ratio[0]:.1%} дисперсии)", ylabel=f"PC2 ({ratio[1]:.1%} дисперсии)")
    _save(fig, path)


def pcr_curves(rmse_group: np.ndarray, rmse_random: np.ndarray, baseline: float,
               m_eff: int, path: Path) -> None:
    ms = np.arange(1, len(rmse_group) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ms, rmse_group, color=BLUE, marker="o", ms=7, label="контроль по сериям (новая скважина)")
    ax.plot(ms, rmse_random, color=AQUA, marker="s", ms=5, label="контроль по замерам")
    ax.axhline(baseline, color=MUTED, ls=":", lw=1.5, label="константа ȳ (по сериям)")
    ax.axvline(m_eff, color=ORANGE, ls="--", lw=1.5, label=f"эффективная размерность m = {m_eff}")
    best = int(np.argmin(rmse_group))
    ax.plot([best + 1], [rmse_group[best]], "o", ms=13, mfc="none", mec=ORANGE, mew=2)
    ax.annotate(f"min = {rmse_group[best]:.1f} (m = {best + 1})", (best + 1, rmse_group[best]),
                xytext=(8, -18), textcoords="offset points", color=INK)
    ax.set(xlabel="число компонент m", ylabel="RMSE, тыс. м³/сут",
           title="Регрессия на главных компонентах, 10×5-fold CV", xticks=ms)
    ax.legend(frameon=False, fontsize=9)
    _save(fig, path)


def ridge_curve(taus: np.ndarray, rmse: np.ndarray, n_eff: np.ndarray, rmse_ols: float,
                best_pcr: float, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.semilogx(taus, rmse, color=BLUE)
    ax1.axhline(rmse_ols, color=MUTED, ls=":", lw=1.5, label=f"МНК, τ = 0 ({rmse_ols:.1f})")
    ax1.axhline(best_pcr, color=ORANGE, ls="--", lw=1.5, label=f"лучший PCR ({best_pcr:.1f})")
    best = int(np.argmin(rmse))
    ax1.plot([taus[best]], [rmse[best]], "o", ms=11, mfc="none", mec=ORANGE, mew=2)
    ax1.annotate(f"τ* = {taus[best]:.3g}\nRMSE = {rmse[best]:.1f}", (taus[best], rmse[best]),
                 xytext=(10, 20), textcoords="offset points", color=INK)
    ax1.set(xlabel="τ", ylabel="RMSE, тыс. м³/сут", title="Гребневая регрессия, контроль по сериям")
    ax1.legend(frameon=False, fontsize=9)

    ax2.semilogx(taus, n_eff, color=BLUE)
    ax2.plot([taus[best]], [n_eff[best]], "o", ms=11, mfc="none", mec=ORANGE, mew=2)
    ax2.annotate(f"n_eff(τ*) = {n_eff[best]:.1f}", (taus[best], n_eff[best]),
                 xytext=(10, 10), textcoords="offset points", color=INK)
    ax2.set(xlabel="τ", ylabel="Σ λ_j / (λ_j + τ)", title="Эффективная размерность гребневой регрессии")
    _save(fig, path)


def coefficients(boot: np.ndarray, labels: list[str], names: list[str], path: Path) -> None:
    """Медиана и 90%-интервал коэффициентов по бутстрепу серий испытаний."""
    n_methods, n = boot.shape[1], boot.shape[2]
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = [MUTED, BLUE, ORANGE]
    width = 0.8 / n_methods
    x = np.arange(n)
    for k in range(n_methods):
        lo, med, hi = np.percentile(boot[:, k], [5, 50, 95], axis=0)
        pos = x + (k - (n_methods - 1) / 2) * width
        ax.errorbar(pos, med, yerr=[med - lo, hi - med], fmt="o", ms=4, color=colors[k],
                    capsize=2, lw=1.2, label=labels[k])
    ax.axhline(0, color=GRID, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60)
    ax.set(ylabel="коэффициент (признаки стандартизованы)",
           title="Устойчивость коэффициентов: медиана и 90% бутстреп-интервал")
    ax.legend(frameon=False)
    _save(fig, path)
