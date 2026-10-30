import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import logistic

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=130)
    plt.close(fig)


def convergence(hists, gd_loss, q_star, name="convergence.png"):
    """hists: {метка: History}. Слева Q - Q*, справа длина шага (квадратичная сходимость)."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for label, h in hists.items():
        ax[0].semilogy(np.maximum(np.array(h.loss) - q_star, 1e-16), "o-", label=label)
        ax[1].semilogy(h.step_norm, "o-", label=label)
    ax[0].semilogy(np.maximum(gd_loss - q_star, 1e-16), "--", label="градиентный спуск")
    ax[0].set_xlim(0, 30)
    ax[0].set(xlabel="итерация", ylabel="Q(w_t) - Q*", title="Сходимость по функционалу")
    ax[1].set(xlabel="итерация", ylabel="||w_{t+1} - w_t||", title="Длина шага")
    for a in ax:
        a.grid(alpha=.3)
        a.legend()
    save(fig, name)


def boundary(F2, y, w, names, name="boundary_2d.png"):
    """Область вероятностей и граница P = 0.5 для модели на двух признаках."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    xs = np.linspace(F2[:, 1].min() - .5, F2[:, 1].max() + .5, 300)
    ys = np.linspace(F2[:, 2].min() - .5, F2[:, 2].max() + .5, 300)
    gx, gy = np.meshgrid(xs, ys)
    P = logistic.probability(np.column_stack([np.ones(gx.size), gx.ravel(), gy.ravel()]), w).reshape(gx.shape)
    cs = ax.contourf(gx, gy, P, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=.6)
    fig.colorbar(cs, ax=ax, label="P(y=+1 | x)")
    ax.contour(gx, gy, P, levels=[.5], colors="k", linewidths=2)
    for c, m, lab in [(-1, "o", "нет диабета"), (1, "^", "диабет")]:
        s = y == c
        ax.scatter(F2[s, 1], F2[s, 2], marker=m, s=18, edgecolor="k", linewidth=.3, label=lab)
    ax.plot([], [], "k-", label="граница P = 0.5")
    ax.set(xlabel=f"{names[0]} (станд.)", ylabel=f"{names[1]} (станд.)", title="Логистическая регрессия на двух признаках")
    ax.legend(loc="lower right")
    save(fig, name)


def separable(norms, losses, name="separable.png"):
    """Без регуляризации на разделимой выборке: ||w|| растёт, Q -> 0."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(norms, "o-", label="||w||, tau = 0")
    ax[0].set(xlabel="итерация Ньютона", ylabel="||w||", title="Норма весов не ограничена")
    ax[1].semilogy(losses, "o-", color="C1", label="Q(w), tau = 0")
    ax[1].set(xlabel="итерация Ньютона", ylabel="Q(w)", title="Функционал стремится к нулю")
    for a in ax:
        a.grid(alpha=.3)
        a.legend()
    save(fig, name)


def odds(names, w_own, w_ref, name="odds_ratios.png"):
    """Коэффициенты (на стандартизованных признаках): собственные и sklearn."""
    idx = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(idx + .2, w_own, height=.4, label="собственная реализация")
    ax.barh(idx - .2, w_ref, height=.4, label="sklearn")
    ax.set_yticks(idx)
    ax.set_yticklabels(names)
    ax.axvline(0, color="k", lw=.8)
    ax.set(xlabel="вес w_j (стандартизованный признак)", title="Веса модели на Pima")
    ax.legend()
    save(fig, name)


def calibration(pred, obs, sizes, name="calibration.png"):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="идеальная калибровка")
    ax.plot(pred, obs, "o-", label="модель (тест)")
    ax.set(xlabel="предсказанная вероятность", ylabel="доля класса +1", title="Калибровка вероятностей")
    ax.grid(alpha=.3)
    ax.legend()
    save(fig, name)
