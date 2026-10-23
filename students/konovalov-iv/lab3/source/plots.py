import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = np.array(["#d95f02", "#1b9e77"])


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _grid(X, pad=0.6, n=250):
    xs = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, n)
    ys = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, n)
    xx, yy = np.meshgrid(xs, ys)
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def _draw_surface(ax, decision, X):
    xx, yy, pts = _grid(X)
    z = decision(pts).reshape(xx.shape)
    ax.contourf(xx, yy, z, levels=[-1e9, 0, 1e9], colors=COLORS, alpha=0.18)
    ax.contour(xx, yy, z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"], linewidths=[1, 1.8, 1])


def draw_boundary(ax, decision, X, y, sv_mask, title):
    _draw_surface(ax, decision, X)
    ax.scatter(X[:, 0], X[:, 1], c=COLORS[(y > 0).astype(int)], s=22, edgecolor="white", linewidth=0.4)
    ax.scatter(X[sv_mask, 0], X[sv_mask, 1], s=90, facecolors="none", edgecolors="k", linewidth=1.2)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def _mask(n, idx):
    m = np.zeros(n, bool)
    m[idx] = True
    return m


def plot_kernels(rows, X, y, path):
    fig, axes = plt.subplots(2, len(rows), figsize=(4.4 * len(rows), 8))
    for j, (name, own, ref) in enumerate(rows):
        draw_boundary(axes[0, j], own.decision_function, X, y, _mask(len(y), own.sv_index_),
                      f"своя: {name}, опорных {len(own.sv_index_)}")
        draw_boundary(axes[1, j], ref.decision_function, X, y, _mask(len(y), ref.support_),
                      f"sklearn.SVC: {name}, опорных {len(ref.support_)}")
    save(fig, path)


def plot_C(models, X, y, path):
    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 4.4))
    for ax, (C, m) in zip(axes, models):
        draw_boundary(ax, m.decision_function, X, y, _mask(len(y), m.sv_index_),
                      f"RBF, C = {C}, опорных {len(m.sv_index_)}")
    save(fig, path)


def plot_sv_types(model, X, y, path):
    types = model.sv_types()
    fig, ax = plt.subplots(figsize=(6, 5.5))
    _draw_surface(ax, model.decision_function, X)
    names = ["периферийные (λ = 0)", "опорные-граничные (0 < λ < C)", "опорные-нарушители (λ = C)"]
    marks = ["o", "s", "X"]
    sizes = [20, 55, 70]
    for t in range(3):
        m = types == t
        ax.scatter(X[m, 0], X[m, 1], c=COLORS[(y[m] > 0).astype(int)], marker=marks[t], s=sizes[t],
                   edgecolor="k", linewidth=0.6, label=f"{names[t]}: {m.sum()}")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Типизация объектов (RBF, своя реализация)")
    save(fig, path)


def plot_cancer_C(Cs, own_acc, ref_acc, n_sv, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(Cs, own_acc, marker="o", label="своя")
    axes[0].plot(Cs, ref_acc, marker="x", linestyle="--", label="sklearn.SVC")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("C")
    axes[0].set_ylabel("accuracy на тесте")
    axes[0].set_title("Breast Cancer, RBF: качество от C")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    axes[1].plot(Cs, n_sv, marker="o", color="tab:red")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("C")
    axes[1].set_ylabel("число опорных векторов")
    axes[1].set_title("Опорные векторы от C")
    axes[1].grid(alpha=0.3)
    save(fig, path)
