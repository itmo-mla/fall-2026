"""Графики для классификации."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from .core import margin, log

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "results" / "figures"


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log(f"[plot] {path}")


"""
def plot_margin_hist(X, y, w, b, name="margin.png"):
    m = margin(X, y, w, b)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(m, bins=60, color="steelblue")
    ax.axvline(0, color="red", ls="--", label="margin=0")
    ax.axvline(1, color="orange", ls=":", label="margin=1")
    ax.set_xlabel("margin  (y * (w^T x + b))")
    ax.set_ylabel("частота")
    ax.legend()
    _save(fig, name)
"""

def plot_margin_hist(X, y, w, b, name="margin.png"):
    m = margin(X, y, w, b)
    ms = np.sort(m)
    idx = np.arange(len(ms))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(idx, ms, color="steelblue", lw=1.5)
    ax.axhline(0, color="red", ls="--", label="margin = 0")
    ax.axhline(1, color="orange", ls=":", label="margin = 1")
    ax.fill_between(idx, ms, 0, where=(ms < 0),
                    color="red", alpha=0.2, label="ошибки (m < 0)")
    ax.set_xlabel("объект (отсортирован по отступу)")
    ax.set_ylabel("margin  y·(wᵀx + b)")
    ax.set_xlim(0, len(ms) - 1)
    ax.set_ylim(ms.min(), ms.max())
    ax.legend()
    _save(fig, name)


def _extract(hist, key):
    out = []
    for h in hist:
        if isinstance(h, dict):
            out.append(h.get(key, np.nan))
        else:
            out.append(h if key == "loss" else np.nan)
    return np.asarray(out, dtype=float)


def plot_loss_curves(histories, name="loss_curves.png", keys=("loss",)):
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, h in histories.items():
        for key in keys:
            y = _extract(h, key)
            if np.all(np.isnan(y)):
                continue
            tag = label if key == "loss" else f"{label} [{key}]"
            style = "-" if key == "loss" else "--"
            ax.plot(y, style, label=tag)
    ax.set_xlabel("эпоха"); ax.set_ylabel("loss")
    ax.grid(alpha=0.3); ax.legend()
    _save(fig, name)


def plot_ema_curves(history, name="ema_curves.png",
                    alphas=(0.9, 0.99, 0.999), loss_key="loss"):
    epochs = np.arange(1, len(history) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, [h[loss_key] for h in history], "k-", lw=2,
            label="full-batch loss")
    styles = [("--", "tab:orange"), (":", "tab:green"), ("-.", "tab:red")]
    for a, (ls, c) in zip(alphas, styles):
        key = f"ema_{a}"
        if key not in history[0]:
            continue
        ax.plot(epochs, [h[key] for h in history], ls, color=c, lw=1.6,
                label=f"EMA α={a}")
    ax.set_xlabel("эпоха"); ax.set_ylabel("loss")
    ax.grid(alpha=0.3); ax.legend()
    _save(fig, name)


def plot_confusion(y_true, y_pred, name="confusion.png"):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == -1) & (y_true == -1)))
    fp = int(np.sum((y_pred == 1) & (y_true == -1)))
    fn = int(np.sum((y_pred == -1) & (y_true == 1)))
    mat = np.array([[tp, fn], [fp, tn]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["+1", "-1"]); ax.set_yticklabels(["+1", "-1"])
    ax.set_xlabel("pred"); ax.set_ylabel("true")
    _save(fig, name)


def plot_decision_boundary(X, y, w, b, name="boundary.png"):
    mean = X.mean(axis=0)
    Xc = X - mean
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T
    w2 = Vt[:2] @ w
    b2 = float(w @ mean + b)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=6, alpha=0.5,
               c="tab:green", label="+1")
    ax.scatter(Z[y == -1, 0], Z[y == -1, 1], s=6, alpha=0.5,
               c="tab:red", label="-1")
    xs = np.linspace(Z[:, 0].min(), Z[:, 0].max(), 200)
    if abs(w2[1]) > 1e-9:
        ax.plot(xs, -(w2[0] * xs + b2) / w2[1], "k-", lw=2)
    ax.legend()
    _save(fig, name)


def plot_bar_by_axis(results, axis_idx, axis_label,
                     metric="accuracy", name=None):   # <<< дефолт исправлен: было "test_acc"
    groups = defaultdict(list)
    for k, res in results.items():
        rep = res.get("test_rep", {})
        if metric not in rep:
            raise KeyError(
                f"Метрика '{metric}' не найдена в test_rep. "
                f"Доступные: {list(rep.keys())}"
            )
        groups[k[axis_idx]].append(rep[metric])

    labels = list(groups.keys())
    means = [float(np.mean(groups[l])) for l in labels]
    stds = [float(np.std(groups[l])) for l in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(labels)), means, yerr=stds, capsize=5, color="steelblue")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([str(l) for l in labels])
    ax.set_ylabel(metric)
    ax.set_title(f"Влияние: {axis_label}")
    ax.grid(alpha=0.3, axis="y")
    _save(fig, name or f"axis_{axis_label}.png")