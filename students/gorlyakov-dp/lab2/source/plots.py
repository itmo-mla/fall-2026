import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_data_projection(X, y, path, prototypes=None):
    pca = PCA(n_components=2, svd_solver="full")
    projection = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls, name, color in [
        (-1, "Злокачественные (-1)", "tab:orange"),
        (1, "Доброкачественные (+1)", "tab:blue"),
    ]:
        mask = y == cls
        ax.scatter(
            projection[mask, 0],
            projection[mask, 1],
            s=25,
            alpha=0.65 if prototypes is None else 0.15,
            color=color,
            label=name if prototypes is None else None,
        )
        if prototypes is not None:
            selected = prototypes[y[prototypes] == cls]
            ax.scatter(
                projection[selected, 0],
                projection[selected, 1],
                s=70,
                color=color,
                edgecolors="black",
                linewidths=0.8,
                label=f"{name}, эталоны",
            )

    ax.set(
        xlabel=f"PC1 ({explained[0]:.1%} дисперсии)",
        ylabel=f"PC2 ({explained[1]:.1%} дисперсии)",
        title="Обучающая выборка, PCA",
    )
    ax.set_aspect("equal", adjustable="box")
    if prototypes is not None:
        ax.set_title(f"Эталоны: {len(prototypes)} из {len(X)}, PCA")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_selection_history(history, path):
    removed = history[0, 0] - history[:, 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(removed, history[:, 1], linewidth=1.5)
    ax.scatter(
        removed[[0, -1]], history[[0, -1], 1], color="tab:red", zorder=3,
    )
    ax.set(
        xlabel="Число удаленных объектов",
        ylabel="Доля ошибок LOO для 1NN",
        title="Изменение ошибки при отборе эталонов",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_loo_risk(ks, risks, k, path):
    best_risk = risks[ks == k][0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, limit in zip(axes, [ks[-1], 50]):
        mask = ks <= limit
        ax.plot(ks[mask], risks[mask], linewidth=1.5)
        ax.scatter(k, best_risk, color="tab:red", zorder=3, label=f"Минимум, k = {k}")
        ax.set(xlabel="k", ylabel="Доля ошибок LOO", title=f"k от 1 до {limit}")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
