import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

CORRECT, WRONG = "#2e8b57", "#d03b3b"


def plot_sorted_margins(m, title, path):
    m = np.sort(m)
    idx = np.arange(len(m))
    wrong = m < 0

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(idx[wrong], m[wrong], color=WRONG, width=0.8)
    ax.bar(idx[~wrong], m[~wrong], color=CORRECT, width=0.8)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("номер объекта после сортировки")
    ax.set_ylabel("отступ M")
    ax.set_title(title)
    ax.legend(handles=[
        Patch(color=WRONG, label=f"ошибка, M < 0 ({wrong.sum()})"),
        Patch(color=CORRECT, label=f"верно, M > 0 ({(~wrong).sum()})"),
    ], loc="upper left")
    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
