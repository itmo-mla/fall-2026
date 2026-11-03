from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PICTURES_DIR = Path(__file__).resolve().parent / "pics"


def _save(fig, filename: str) -> Path:
    PICTURES_DIR.mkdir(exist_ok=True)
    path = PICTURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scree(explained_variance: np.ndarray, kaiser_k: int, variance_k: int) -> Path:
    k = len(explained_variance)
    x = np.arange(1, k + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, explained_variance, alpha=0.6, label="Дисперсия компонент")
    ax.plot(x, explained_variance, "o-", color="darkred", ms=3,
            label="Профиль (scree)")
    ax.axhline(1.0, color="green", ls="--", lw=2,
               label="Порог Кайзера (уровень 1)")
    ax.axvline(kaiser_k, color="green", ls=":", lw=2,
               label=f"k_Кайзера = {kaiser_k}")
    ax.axvline(variance_k, color="orange", ls=":", lw=2,
               label=f"k (95% дисперсии) = {variance_k}")
    ax.set_xlabel("Номер главной компоненты")
    ax.set_ylabel("Объяснённая дисперсия")
    ax.set_title("Scree plot: спектр дисперсии главных компонент")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, "scree_plot.png")


def plot_cumulative_variance(explained_variance_ratio: np.ndarray,
                             variance_k: int, effective_rank: float) -> Path:
    k = len(explained_variance_ratio)
    x = np.arange(1, k + 1)
    cumulative = np.cumsum(explained_variance_ratio)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, cumulative, "o-", ms=3, color="darkblue",
            label="Накопленная доля дисперсии")
    for threshold in (0.90, 0.95, 0.99):
        ax.axhline(threshold, color="gray", ls="--", lw=1)
        n_thr = int(np.searchsorted(cumulative, threshold)) + 1
        ax.annotate(f"{int(threshold * 100)}%: k = {n_thr}",
                    xy=(k, threshold), ha="right", va="bottom", fontsize=9)
    ax.axvline(variance_k, color="orange", ls=":", lw=2,
               label=f"k_eff (95%) = {variance_k}")
    ax.axvline(round(effective_rank), color="purple", ls=":", lw=2,
               label=f"Эффективный ранг ≈ {effective_rank:.1f}")
    ax.set_xlabel("Число компонент")
    ax.set_ylabel("Накопленная объяснённая дисперсия")
    ax.set_title("Накопленная объяснённая дисперсия vs число компонент")
    ax.set_ylim(0, 1.03)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return _save(fig, "cumulative_variance.png")
