from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from data import load_iris_split
from svm import LinearSVM


def main():
    x_train, x_test, y_train, y_test = load_iris_split()
    x_train = x_train[:, 2:4]
    x_test = x_test[:, 2:4]
    model = LinearSVM(c=1.0).fit(x_train, y_train)

    points = np.vstack((x_train, x_test))
    low = points.min(axis=0) - 0.5
    high = points.max(axis=0) + 0.5
    xx, yy = np.meshgrid(
        np.linspace(low[0], high[0], 300),
        np.linspace(low[1], high[1], 300),
    )
    scores = model.decision_function(np.column_stack((xx.ravel(), yy.ravel())))
    scores = scores.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, scores, levels=[-1e9, 0, 1e9], colors=["#e3effa", "#fff0df"])
    ax.contour(
        xx, yy, scores, levels=[-1, 0, 1], colors="black",
        linestyles=["--", "-", "--"], linewidths=1.2,
    )

    for label, name, color in ((-1, "versicolor", "#2878b5"), (1, "virginica", "#d97719")):
        train = x_train[y_train == label]
        test = x_test[y_test == label]
        ax.scatter(train[:, 0], train[:, 1], c=color, s=35, label=name)
        ax.scatter(test[:, 0], test[:, 1], c=color, marker="^", s=65, edgecolors="black")

    support = x_train[model.support]
    ax.scatter(
        support[:, 0], support[:, 1], s=110, facecolors="none",
        edgecolors="black", label="Опорные векторы",
    )
    handles, _ = ax.get_legend_handles_labels()
    handles.extend([
        Line2D([], [], marker="^", color="black", linestyle="none", label="Тестовые объекты"),
        Line2D([], [], color="black", label="Граница: f(x) = 0"),
        Line2D([], [], color="black", linestyle="--", label="Отступы: f(x) = ±1"),
    ])
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    ax.set(
        xlabel="Длина лепестка (стандартизованная)",
        ylabel="Ширина лепестка (стандартизованная)",
        title="Линейный SVM на двух признаках Iris, C = 1",
        xlim=(low[0], high[0]), ylim=(low[1], high[1]),
    )
    fig.tight_layout()

    path = Path(__file__).resolve().parents[1] / "figures" / "linear_svm.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"График сохранён: {path.name}")
    print(f"Опорных векторов: {len(model.support)}")
    print(f"Accuracy на двух признаках: {np.mean(model.predict(x_test) == y_test):.2%}")


if __name__ == "__main__":
    main()
