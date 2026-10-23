import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_loo(ks, risks, risks_biased, k_best, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ks, risks, marker="o", markersize=4, label="LOO (несмещённая оценка)")
    ax.plot(ks, risks_biased, marker="o", markersize=4, linestyle="--",
            label="объект считается соседом самого себя (смещённая)")
    ax.scatter([k_best], [risks[list(ks).index(k_best)]], s=120, color="red", zorder=3,
               label=f"оптимум k = {k_best}")
    ax.set_xlabel("число соседей k")
    ax.set_ylabel("эмпирический риск (доля ошибок)")
    ax.set_title("Выбор k методом скользящего контроля (LOO)")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, path)


def plot_compactness(profiles, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, profile in profiles.items():
        m = np.arange(1, len(profile) + 1)
        ax.plot(m, profile, marker="o", markersize=4, label=label)
    ax.set_xlabel("номер соседа m")
    ax.set_ylabel("Π(m)")
    ax.set_title("Профиль компактности: доля объектов, у которых m-й сосед из другого класса")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, path)


def plot_ccv_paths(remove_path, n_removed, add_path, n_classes, path):
    total = len(remove_path) - 1 + n_classes  # путь идёт до одного эталона на класс
    fig, (ax_r, ax_a) = plt.subplots(1, 2, figsize=(12, 4))
    ax_r.plot(np.arange(len(remove_path)), remove_path, label="CCV(Ω)")
    ax_r.scatter([n_removed], [remove_path[n_removed]], s=100, color="red", zorder=3,
                 label=f"выбрано: удалено {n_removed}, осталось {total - n_removed}")
    ax_r.set_xlabel("число удалённых объектов")
    ax_r.set_ylabel("CCV(Ω)")
    ax_r.set_title("Жадное удаление не-эталонов")
    ax_r.grid(alpha=0.3)
    ax_r.legend()

    sizes = np.arange(n_classes, n_classes + len(add_path))
    ax_a.plot(sizes, add_path, marker="o", markersize=3, label="CCV(Ω)")
    ax_a.set_xlabel("число эталонов |Ω|")
    ax_a.set_ylabel("CCV(Ω)")
    ax_a.set_title("Жадное добавление эталонов")
    ax_a.grid(alpha=0.3)
    ax_a.legend()
    save(fig, path)


def pca_2d(X):
    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def plot_prototypes(X, y, masks, class_names, path):
    Z = pca_2d(X)
    fig, axes = plt.subplots(1, len(masks), figsize=(6 * len(masks), 5))
    for ax, (title, mask) in zip(np.atleast_1d(axes), masks.items()):
        for c, name in enumerate(class_names):
            idx = y == c
            ax.scatter(*Z[idx & ~mask].T, s=20, color=f"C{c}", alpha=0.25,
                       label=f"{name}: не-эталон")
            ax.scatter(*Z[idx & mask].T, s=90, color=f"C{c}", edgecolors="black",
                       label=f"{name}: эталон ({np.sum(idx & mask)})")
        ax.set_title(f"{title}: {mask.sum()} эталонов из {len(y)}")
        ax.set_xlabel("главная компонента 1")
        ax.set_ylabel("главная компонента 2")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    save(fig, path)
