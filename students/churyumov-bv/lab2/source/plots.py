import numpy as np
import matplotlib.pyplot as plt


def plot_loo_risk(ks, risks, best_k, path):
    plt.figure()
    plt.plot(ks, risks, label="LOO")
    plt.axvline(best_k, linestyle="--", color="gray", label=f"k = {best_k}")
    plt.xlabel("k")
    plt.ylabel("эмпирический риск LOO")
    plt.legend()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_pca_predictions(Z, y_true, y_pred, path, name, pca=None, predict_fn=None, n=120):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    if predict_fn is not None:
        mx, my = 0.05 * np.ptp(Z[:, 0]), 0.05 * np.ptp(Z[:, 1])
        xx, yy = np.meshgrid(np.linspace(Z[:, 0].min() - mx, Z[:, 0].max() + mx, n),
                             np.linspace(Z[:, 1].min() - my, Z[:, 1].max() + my, n))
        regions = predict_fn(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
        for ax in axes:
            ax.pcolormesh(xx, yy, regions, cmap="tab10", vmin=0, vmax=9, alpha=0.25, shading="auto")

    for ax, labels, title in zip(axes, (y_true, y_pred),
                                 ("Истинные классы", f"Предсказание: {name}")):
        ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", vmin=0, vmax=9, s=25)
        ax.set_title(title)
        ax.set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    wrong = y_true != y_pred
    axes[1].scatter(Z[wrong, 0], Z[wrong, 1], facecolors="none",
                    edgecolors="black", s=90, label="ошибки")
    axes[1].legend()

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ccv_history(history, stop, path):
    plt.figure()
    plt.plot(np.arange(len(history)), history, label="CCV(Ω)")
    plt.axvline(stop, linestyle="--", color="gray", label=f"остановка: удалено {stop}")
    plt.xlabel("число удалённых объектов")
    plt.ylabel("CCV(Ω)")
    plt.legend()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_prototypes(Z, y, omega, path):
    is_proto = np.zeros(len(y), dtype=bool)
    is_proto[omega] = True
    colors = plt.get_cmap("tab10")

    plt.figure(figsize=(7, 6))
    for c in np.unique(y):
        rest = (y == c) & ~is_proto
        plt.scatter(Z[rest, 0], Z[rest, 1], s=12, alpha=0.3, color=colors(c),
                    label=f"класс {c}, не эталоны")
    for c in np.unique(y):
        proto = (y == c) & is_proto
        plt.scatter(Z[proto, 0], Z[proto, 1], s=60, color=colors(c),
                    edgecolors="black", label=f"класс {c}, эталоны")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
