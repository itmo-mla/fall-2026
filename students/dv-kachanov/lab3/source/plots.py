import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(model, X, y, output_path, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    values = model.decision_function(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, values, levels=25, cmap="coolwarm", alpha=0.25)
    plt.contour(xx, yy, values, levels=[-1, 0, 1], colors=["gray", "black", "gray"], linestyles=["--", "-", "--"])

    for label, marker in [(-1.0, "o"), (1.0, "s")]:
        mask = y == label
        plt.scatter(
            X[mask, 0],
            X[mask, 1],
            marker=marker,
            s=35,
            label=f"Класс {int(label)}"
        )

    plt.scatter(
        model.support_vectors[:, 0],
        model.support_vectors[:, 1],
        s=130,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Опорные векторы"
    )

    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_margins(margins, output_path):
    sorted_margins = np.sort(margins)

    plt.figure(figsize=(8, 5))
    plt.plot(sorted_margins, label="M_i(w, w0)")
    plt.axhline(0, linestyle="--", label="Граница ошибки")
    plt.axhline(1, linestyle="--", label="M = 1")
    plt.xlabel("Объекты")
    plt.ylabel("Отступ")
    plt.title("Отступы тестовых объектов")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_quality(names, accuracies, output_path):
    x = np.arange(len(names))

    plt.figure(figsize=(8, 5))
    plt.bar(x, accuracies, label="Accuracy")
    plt.xticks(x, names, rotation=15)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Сравнение SVM")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
