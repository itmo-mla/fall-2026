from matplotlib.lines import Line2D
from pathlib import Path
from time import perf_counter
from scipy.optimize import minimize
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import numpy as np


class SVM:
    def __init__(self, C=1.0, kernel="linear", gamma=1.0, degree=3):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree

    def kernel_matrix(self, X1, X2):
        if self.kernel == "linear":
            return X1 @ X2.T
        if self.kernel == "poly":
            return (X1 @ X2.T + 1) ** self.degree
        if self.kernel == "rbf":
            distances = (
                np.sum(X1 ** 2, axis=1)[:, None]
                + np.sum(X2 ** 2, axis=1)[None, :]
                - 2 * X1 @ X2.T
            )
            return np.exp(-self.gamma * np.maximum(distances, 0))
        
        raise ValueError("Непредусмотренный тип ядра: " + self.kernel)

    @staticmethod
    def lagrangian(lambdas, Q):
        return 0.5 * lambdas @ Q @ lambdas - np.sum(lambdas)

    @staticmethod
    def lagrangian_gradient(lambdas, Q):
        return Q @ lambdas - np.ones_like(lambdas)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        K = self.kernel_matrix(X, X)
        Q = y[:, None] * y[None, :] * K
        result = minimize(
            self.lagrangian,
            np.zeros(len(y)),
            args=(Q,),
            jac=self.lagrangian_gradient,
            bounds=[(0, self.C)] * len(y),
            constraints={
                "type": "eq",
                "fun": lambda lambdas: lambdas @ y,
                "jac": lambda lambdas: y
            },
            options={"maxiter": 2000, "ftol": 1e-9},
        )
        if not result.success:
            raise RuntimeError(result.message)

        lambdas = result.x

        support = lambdas > 1e-6
        boundary = support & (lambdas < self.C - 1e-6)
        values = K[:, support] @ (lambdas[support] * y[support])
        w0 = np.mean(values[boundary] - y[boundary])

        self.support_vectors = X[support]
        self.support_labels = y[support]
        self.support_lambdas = lambdas[support]
        self.lambdas = lambdas
        self.w0 = w0
        return self

    def decision_function(self, X):
        K = self.kernel_matrix(np.asarray(X, dtype=float), self.support_vectors)
        return K @ (self.support_lambdas * self.support_labels) - self.w0

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


def print_metrics(name, y_true, y_pred, elapsed, support_count):
    print(
        f"{name:<14} {accuracy_score(y_true, y_pred):>9.4f}"
        f" {precision_score(y_true, y_pred, zero_division=0):>10.4f}"
        f" {recall_score(y_true, y_pred, zero_division=0):>9.4f}"
        f" {f1_score(y_true, y_pred, zero_division=0):>9.4f}"
        f" {support_count:>8} {elapsed:>10.4f}"
    )


def plot_solution(ax, X_train, y_train, X_test, y_test, support_vectors, grid, values, title):
    grid_x, grid_y = grid
    values = values.reshape(grid_x.shape)
    regions = np.where(values >= 0, 1, -1)
    ax.contourf(grid_x, grid_y, regions, levels=[-2, 0, 2], colors=["#8fc2de", "#eda978"], alpha=0.5)
    ax.contour(grid_x, grid_y, values, levels=[-1, 0, 1], colors="black", linestyles=["--", "-", "--"], linewidths=1.1)
    for label, color in [(-1, "#246896"), (1, "#be5928")]:
        train = y_train == label
        test = y_test == label
        ax.scatter(X_train[train, 0], X_train[train, 1], c=color, s=24, edgecolors="white", linewidths=0.4)
        ax.scatter(X_test[test, 0], X_test[test, 1], c=color, s=32, marker="x")
    ax.scatter(support_vectors[:, 0], support_vectors[:, 1], s=85, facecolors="none", edgecolors="black", linewidths=0.9)
    ax.set(title=title, xlabel="$x_1$", ylabel="$x_2$")
    ax.set_xlim(grid_x.min(), grid_x.max())
    ax.set_ylim(grid_y.min(), grid_y.max())
    ax.set_aspect("equal")


def visualize(X_train, y_train, X_test, y_test, models, references):
    X = np.vstack([X_train, X_test])
    grid = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 250),
        np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 250),
    )
    points = np.column_stack([grid[0].ravel(), grid[1].ravel()])

    titles = ["Линейное", "Полиномиальное, d = 3", "RBF"]
    own_values = [model.decision_function(points) for model in models]
    reference_values = [
        model.decision_function(points) for model in references
    ]

    legend = [
        Line2D([], [], marker="o", color="none",
               markerfacecolor="#246896", label="Класс −1"),
        Line2D([], [], marker="o", color="none",
               markerfacecolor="#be5928", label="Класс +1"),
        Line2D([], [], marker="x", color="gray",
               linestyle="none", label="Тестовые объекты"),
        Line2D([], [], marker="o", color="black",
               markerfacecolor="none", linestyle="none",
               label="Опорные векторы"),
        Line2D([], [], color="black", label="$g(x)=0$"),
        Line2D([], [], color="black", linestyle="--",
               label=r"$g(x)=\pm1$"),
    ]

    output = Path(__file__).resolve().parent.parent / "img"
    output.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for i in range(3):
        plot_solution(
            axes[0, i],
            X_train, y_train, X_test, y_test,
            models[i].support_vectors,
            grid, own_values[i],
            "Собственный SVM: " + titles[i],
        )
        plot_solution(
            axes[1, i],
            X_train, y_train, X_test, y_test,
            references[i].support_vectors_,
            grid, reference_values[i],
            "sklearn SVC: " + titles[i],
        )

    fig.suptitle("Сравнение с эталонной реализацией")
    fig.legend(handles=legend, loc="lower center", ncol=6)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(output / "reference.png", dpi=180)
    plt.close(fig)

    print("\nИзображение сохранено в:", output / "reference.png")


def main():
    C = 1.0
    gamma = 1.0
    degree = 3

    X, y = make_moons(n_samples=300, noise=0.15, random_state=42)
    y = 2 * y - 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models, references = [], []

    for kernel in ["linear", "poly", "rbf"]:
        start = perf_counter()
        model = SVM(C=C, kernel=kernel, gamma=gamma, degree=degree).fit(X_train, y_train)
        own_time = perf_counter() - start
        start = perf_counter()
        reference = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, coef0=1.0, tol=1e-7, shrinking=False).fit(X_train, y_train)
        reference_time = perf_counter() - start
        own_predictions = model.predict(X_test)
        reference_predictions = reference.predict(X_test)
        print(f"\nЯдро: {kernel}")
        print(f"{'Модель':<14} {'Accuracy':>9} {'Precision':>10} {'Recall':>9} {'F1':>9} {'Опорных':>8} {'Время, с':>10}")
        print_metrics("Собственный", y_test, own_predictions, own_time, len(model.support_vectors))
        print_metrics("sklearn SVC", y_test, reference_predictions, reference_time, len(reference.support_))
        models.append(model)
        references.append(reference)

    visualize(X_train, y_train, X_test, y_test, models, references)


if __name__ == "__main__":
    main()
