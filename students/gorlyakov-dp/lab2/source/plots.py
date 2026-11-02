import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_data_projection(X, y, path):
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
            alpha=0.65,
            color=color,
            label=name,
        )

    ax.set(
        xlabel=f"PC1 ({explained[0]:.1%} дисперсии)",
        ylabel=f"PC2 ({explained[1]:.1%} дисперсии)",
        title="Обучающая выборка, PCA",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend()
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
