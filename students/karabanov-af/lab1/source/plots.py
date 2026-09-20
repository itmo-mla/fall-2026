import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

CORRECT, WRONG = "#2e8b57", "#d03b3b"
CURVES = ["#2a78d6", "#eb6834"]


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


def plot_risk(histories, title, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for (label, history), color in zip(histories.items(), CURVES):
        ax.plot(history, color=color, linewidth=2, label=label)
    ax.set_xlabel("эпоха")
    ax.set_ylabel("Q")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_l2(taus, risk, errors, norm, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    panels = [
        (risk, "Q", "Риск на обучении", "эмпирический риск Q"),
        (errors, "ошибок (M < 0)", "Число ошибок", "ошибок из 100"),
        (norm, "||w||", "Норма весов", "||w|| без bias"),
    ]
    for ax, (values, ylabel, title_ax, label), color in zip(axes, panels, CURVES + ["#1baf7a"]):
        ax.plot(taus, values, marker="o", color=color, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("коэффициент регуляризации tau")
        ax.set_ylabel(ylabel)
        ax.set_title(title_ax)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(title)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
