import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CURVES = ["#2a78d6", "#eb6834", "#1baf7a"]
CLASSES = ["#2a78d6", "#eb6834", "#1baf7a"]
NOISE = "#d03b3b"


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_loo(ks, risks, title, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (label, risk), color in zip(risks.items(), CURVES):
        best = ks[risk.argmin()]
        ax.plot(ks, risk, marker="o", markersize=4, color=color, label=f"{label}, лучшее k = {best}")
        ax.plot(best, risk.min(), marker="*", markersize=16, color=color)
    ax.set_xlabel("число соседей k")
    ax.set_ylabel("доля ошибок LOO")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, path)


def plot_prototypes(P, y, prototypes, noise, class_names, title, path):
    """P is the 2D projection of the training sample, prototypes and noise are index arrays."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for c, color in zip(range(y.max() + 1), CLASSES):
        mask = y == c
        ax.scatter(P[mask, 0], P[mask, 1], color=color, alpha=0.35, s=35, label=f"{class_names[c]} ({mask.sum()})")
    ax.scatter(P[prototypes, 0], P[prototypes, 1], facecolors="none", edgecolors="black", s=190, linewidths=1.8,
               label=f"эталоны ({len(prototypes)})")
    ax.scatter(P[noise, 0], P[noise, 1], color=NOISE, marker="x", s=90, linewidths=2,
               label=f"шум, отсеян ({len(noise)})")
    ax.set_xlabel("главная компонента 1")
    ax.set_ylabel("главная компонента 2")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, path)


def plot_margins(margins, title, path):
    order = np.argsort(margins)
    values = margins[order]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(np.arange(len(values))[values <= 0], values[values <= 0], color=NOISE, width=0.9,
           label=f"шум, M <= 0 ({(values <= 0).sum()})")
    ax.bar(np.arange(len(values))[values > 0], values[values > 0], color=CURVES[2], width=0.9,
           label=f"оставлены, M > 0 ({(values > 0).sum()})")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("номер объекта после сортировки")
    ax.set_ylabel("отступ M")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, path)


def plot_comparison(results, title, path):
    labels = list(results)
    values = [results[name] for name in labels]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, [np.mean(v) for v in values], yerr=[np.std(v) for v in values], capsize=5,
           color=CURVES[:len(labels)], width=0.5, label="среднее ± std по разбиениям")
    for x, v in enumerate(values):
        ax.scatter([x] * len(v), v, color="black", s=18, zorder=3,
                   label="отдельные разбиения" if x == 0 else None)
    ax.set_ylim(min(min(v) for v in values) - 0.05, 1.02)
    ax.set_ylabel("accuracy на тесте")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="y")
    save(fig, path)


def plot_confusions(matrices, class_names, title, path):
    fig, axes = plt.subplots(1, len(matrices), figsize=(5.5 * len(matrices), 4.5))
    for ax, (name, cm) in zip(np.atleast_1d(axes), matrices.items()):
        ax.imshow(cm, cmap="Blues")
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(len(cm)), class_names)
        ax.set_yticks(range(len(cm)), class_names)
        ax.set_xlabel("предсказанный класс")
        ax.set_ylabel("истинный класс")
        ax.set_title(name)
        ax.grid(False)
    fig.suptitle(title)
    save(fig, path)
