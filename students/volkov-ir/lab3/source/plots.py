import numpy as np
import matplotlib.pyplot as plt


def make_grid(X):
    # сетка 200 x 200 по двум признакам
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200), np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200))
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return xx, yy, points


def plot_boundaries(panels, xx, yy, X, y, feature_names, title, path):
    # panels - список (подпись, score на сетке, lam, C, отступы train)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (label, Z, lam, C, M) in zip(axes, panels):
        # фон только вне полосы, полоса |score| < 1 белая
        ax.contourf(xx, yy, Z, levels=[Z.min(), -1, 1, Z.max()], colors=["#e3eef9", "white", "#fbe3e3"])
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors="black", linestyles=["--", "-", "--"])
        ax.scatter(X[y == -1, 0], X[y == -1, 1], s=12, color="tab:blue", label="доброкачественная, -1")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=12, color="tab:red", label="злокачественная, +1")
        # граничные обводим, они лежат на пунктире
        border = (lam > 1e-6) & (lam < C - 1e-6)
        ax.scatter(X[border, 0], X[border, 1], s=60, facecolors="none", edgecolors="black", linewidths=1.2, label="граничные")
        # явные нарушители: M < -1, лежат в зоне чужого класса
        wrong = M < -1
        ax.scatter(X[wrong, 0], X[wrong, 1], s=60, marker="s", facecolors="none", edgecolors="black", linewidths=1.2, label="в чужой зоне")

        ax.set_title(f"{label}, граничных {np.sum(border)}, в чужой зоне {np.sum(wrong)}", fontsize=10)
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_margins(M, lam, C, path):
    # отступы по возрастанию, цвет - тип объекта
    order = np.argsort(M)
    M = M[order]
    lam = lam[order]
    idx = np.arange(len(M))
    periphery = lam <= 1e-6
    border = (lam > 1e-6) & (lam < C - 1e-6)
    violators = lam >= C - 1e-6

    plt.figure(figsize=(8, 4.5))
    plt.axhline(0, color="black")
    plt.axhline(1, color="black", ls="--")
    plt.scatter(idx[periphery], M[periphery], s=10, color="tab:gray", label=f"периферийные ({np.sum(periphery)})")
    plt.scatter(idx[border], M[border], s=22, color="tab:green", label=f"граничные ({np.sum(border)})")
    plt.scatter(idx[violators], M[violators], s=22, marker="s", color="tab:red", label=f"нарушители ({np.sum(violators)})")
    plt.xlabel("объекты train по возрастанию отступа")
    plt.ylabel("отступ M")
    plt.title("Отступы объектов train, линейное ядро, C = 0.1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
