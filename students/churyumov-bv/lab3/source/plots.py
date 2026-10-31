import numpy as np
import matplotlib.pyplot as plt

TYPE_NAMES = ["периферийный", "опорный-граничный", "опорный-нарушитель"]
TYPE_COLORS = ["tab:gray", "tab:green", "tab:red"]


def plot_boundaries(items, Z, y, path, n=200, xlabel="PC1", ylabel="PC2"):
    """
    items - список (заголовок, модель SVM, обученная на Z), Z - объекты в 2D, y - метки из {-1, +1}.
    Сплошная линия - разделяющая гиперплоскость f(x) = 0, пунктир - границы полосы f(x) = +-1,
    обведены опорные объекты.
    """
    fig, axes = plt.subplots(1, len(items), figsize=(5 * len(items), 4.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    mx, my = 0.1 * np.ptp(Z[:, 0]), 0.1 * np.ptp(Z[:, 1])
    xx, yy = np.meshgrid(np.linspace(Z[:, 0].min() - mx, Z[:, 0].max() + mx, n),
                         np.linspace(Z[:, 1].min() - my, Z[:, 1].max() + my, n))
    grid = np.c_[xx.ravel(), yy.ravel()]

    for ax, (title, model) in zip(axes, items):
        f = model.decision_function(grid).reshape(xx.shape)
        ax.contourf(xx, yy, np.sign(f), levels=[-2, 0, 2], cmap="coolwarm", alpha=0.2)
        ax.contour(xx, yy, f, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"], linewidths=1)
        ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="coolwarm", vmin=-1, vmax=1, s=20)
        ax.scatter(Z[model.sv, 0], Z[model.sv, 1], facecolors="none", edgecolors="k", s=80,
                   label=f"опорные: {model.sv.sum()}")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.legend(loc="upper right")
    axes[0].set_ylabel(ylabel)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cv(results, path):
    """results - {подпись: (список C, список CV accuracy)}"""
    plt.figure()
    for name, (Cs, acc) in results.items():
        plt.plot(Cs, acc, marker="o", label=name)
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("accuracy (5-fold CV)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_margins(model, path):
    """Гистограмма отступов M_i обучающих объектов по типам (слайд 10)"""
    M, t = model.margins(), model.types()
    bins = np.linspace(M.min(), M.max(), 40)
    plt.figure()
    plt.hist([M[t == k] for k in range(3)], bins=bins, stacked=True, color=TYPE_COLORS,
             label=[f"{TYPE_NAMES[k]}: {(t == k).sum()}" for k in range(3)])
    plt.axvline(1, linestyle="--", color="k", label="M = 1")
    plt.axvline(0, linestyle="-", color="k", label="M = 0")
    plt.xlabel("отступ $M_i$")
    plt.ylabel("число объектов")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_compare(f_own, f_sk, lam_own, lam_sk, path):
    """Слева: решающая функция на тесте, справа: двойственные переменные lambda_i; идеальное совпадение - диагональ"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, a, b, xl, yl in ((axes[0], f_sk, f_own, "sklearn: f(x)", "своя реализация: f(x)"),
                             (axes[1], lam_sk, lam_own, r"sklearn: $\lambda_i$", r"своя реализация: $\lambda_i$")):
        ax.scatter(a, b, s=18)
        lim = [min(a.min(), b.min()), max(a.max(), b.max())]
        ax.plot(lim, lim, "k--", linewidth=1)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
    axes[0].set_title("решающая функция на тестовой выборке")
    axes[1].set_title("двойственные переменные на обучающей выборке")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
