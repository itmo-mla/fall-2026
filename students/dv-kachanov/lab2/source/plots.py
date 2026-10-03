import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_loo_risks(k_values, risks, best_k, output_path):
    plt.figure(figsize=(9, 5))
    plt.plot(k_values, risks, marker="o", label="LOO(k, X^ell)")
    plt.axvline(best_k, linestyle="--", label=f"best k = {best_k}")
    plt.xlabel("Число соседей k")
    plt.ylabel("Эмпирический риск LOO")
    plt.title("Подбор k методом скользящего контроля")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_prototypes(X, y, prototype_indices, output_path):
    pca = PCA(n_components=2, random_state=42)
    points = pca.fit_transform(X)

    plt.figure(figsize=(9, 6))

    for label in np.unique(y):
        mask = y == label
        plt.scatter(
            points[mask, 0],
            points[mask, 1],
            s=35,
            alpha=0.45,
            label=f"Класс {label}"
        )

    prototype_points = points[prototype_indices]
    plt.scatter(
        prototype_points[:, 0],
        prototype_points[:, 1],
        s=140,
        facecolors="none",
        edgecolors="black",
        linewidths=1.8,
        label="Эталоны"
    )

    plt.xlabel("Первая главная компонента")
    plt.ylabel("Вторая главная компонента")
    plt.title("Отбор эталонных объектов")
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
    plt.title("Сравнение качества KNN")
    plt.grid(axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
