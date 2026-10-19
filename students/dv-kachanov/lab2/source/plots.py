import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_LOO(k_values, LOOs, k_star, output_path):
    plt.figure(figsize=(9, 5))
    plt.plot(k_values, LOOs, marker="o", label="LOO(k, X^l)")
    plt.axvline(k_star, linestyle="--", label=f"k* = {k_star}")
    plt.xlabel("Число соседей k")
    plt.ylabel("LOO(k, X^l) --- число ошибок (сумма)")
    plt.title("Подбор k методом скользящего контроля LOO(k, X^l)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_prototypes(X_L, y_L, Omega, output_path):
    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(X_L)

    plt.figure(figsize=(9, 6))

    for label in np.unique(y_L):
        mask = y_L == label
        plt.scatter(
            points[mask, 0],
            points[mask, 1],
            s=35,
            alpha=0.45,
            label=f"Класс {label}"
        )

    prototype_points = points[np.asarray(Omega)]
    plt.scatter(
        prototype_points[:, 0],
        prototype_points[:, 1],
        s=140,
        facecolors="none",
        edgecolors="black",
        linewidths=1.8,
        label="Эталоны Omega"
    )

    plt.xlabel("Первая главная компонента")
    plt.ylabel("Вторая главная компонента")
    plt.title("Отбор эталонов Omega (PCA-проекция X^L)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_metric_comparison(names, values, output_path):
    x = np.arange(len(names))

    plt.figure(figsize=(9, 5))
    plt.bar(x, values, label="Accuracy")
    plt.xticks(x, names, rotation=15)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Сравнение качества: a(x; X^l, k, K), 1NN на Omega, эталон")
    plt.grid(axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
