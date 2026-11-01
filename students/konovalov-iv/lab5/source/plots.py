import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(models_by_label, Q_optimal, path):
    fig, (ax_q, ax_grad) = plt.subplots(1, 2, figsize=(12, 4))
    for label, model in models_by_label.items():
        Q = np.array(model.history["Q"])
        iterations = np.arange(len(Q))
        ax_q.plot(iterations, np.maximum(Q - Q_optimal, 1e-16), marker=".", label=label)
        ax_grad.plot(iterations, model.history["grad_norm"], marker=".", label=label)
    for ax in (ax_q, ax_grad):
        ax.set_yscale("log")
        ax.set_xscale("symlog", linthresh=10)
        ax.set_xlim(left=0)
        ax.set_xlabel("итерация")
        ax.grid(alpha=0.3)
    ax_q.set_ylabel("Q(w) - Q*")
    ax_q.set_title("Сходимость функционала")
    ax_grad.set_ylabel("‖Q'(w)‖")
    ax_grad.set_title("Норма градиента")
    ax_q.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_weights(weights_by_label, feature_names, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(feature_names))
    width = 0.8 / len(weights_by_label)
    for k, (label, w) in enumerate(weights_by_label.items()):
        ax.bar(x + (k - (len(weights_by_label) - 1) / 2) * width, w, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_ylabel("вес w_j")
    ax.set_title("Веса логистической регрессии")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_irls_weights(M, gamma, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(M, gamma, s=12, alpha=0.6)
    grid = np.linspace(M.min(), M.max(), 300)
    sigma = 1 / (1 + np.exp(-grid))
    ax.plot(grid, sigma * (1 - sigma), color="black", linewidth=1, label="γ = σ(M)(1 - σ(M))")
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlabel("отступ M = y⟨w, x⟩")
    ax.set_ylabel("вес объекта γ в IRLS")
    ax.set_title("Веса объектов в IRLS на последней итерации")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_separable(models_by_label, path):
    fig, (ax_norm, ax_q) = plt.subplots(1, 2, figsize=(12, 4))
    for label, (norms, Q) in models_by_label.items():
        ax_norm.plot(norms, label=label)
        ax_q.plot(Q, label=label)
    ax_norm.set_xlabel("итерация")
    ax_norm.set_ylabel("‖w‖")
    ax_norm.set_title("Норма весов")
    ax_q.set_yscale("log")
    ax_q.set_xlabel("итерация")
    ax_q.set_ylabel("Q(w)")
    ax_q.set_title("Функционал")
    for ax in (ax_norm, ax_q):
        ax.grid(alpha=0.3)
    ax_norm.legend()
    fig.suptitle("Линейно разделимая выборка без регуляризации")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
