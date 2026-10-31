import numpy as np

from scipy.optimize import minimize

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

def load_data(test_size=0.2, random_state=42):
    """
    Загружает Banknote Authentication,
    преобразует классы в {-1, +1},
    делит данные на train/test и стандартизует признаки.
    """

    data = fetch_openml(
        name="banknote-authentication",
        version=1,
        as_frame=True,
    )

    X = data.data.to_numpy(dtype=float)
    y = data.target.to_numpy()

    classes = np.unique(y)

    if len(classes) != 2:
        raise ValueError("Для SVM требуется бинарная классификация.")

    # Преобразуем классы в -1 и +1
    y = np.where(y == classes[0], -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Стандартизация
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, data.feature_names


def make_train_subset(
    X_train,
    y_train,
    n_samples=100,
    random_state=42,
):

    if n_samples >= len(X_train):
        return X_train, y_train

    X_subset, _, y_subset, _ = train_test_split(
        X_train,
        y_train,
        train_size=n_samples,
        random_state=random_state,
        stratify=y_train,
    )

    return X_subset, y_subset


def linear_kernel(X1, X2):
    """
    Линейное ядро:
    K(x, z) = <x, z>
    """

    return X1 @ X2.T


def rbf_kernel(X1, X2, gamma=1.0):
    """
    RBF-ядро:
    K(x, z) = exp(-gamma * ||x - z||^2)
    """

    X1_sq = np.sum(X1 ** 2, axis=1)[:, np.newaxis]
    X2_sq = np.sum(X2 ** 2, axis=1)[np.newaxis, :]

    squared_distances = (
        X1_sq
        + X2_sq
        - 2 * X1 @ X2.T
    )

    # Из-за погрешностей вычислений значение теоретически
    # может оказаться совсем немного меньше нуля.
    squared_distances = np.maximum(
        squared_distances,
        0.0,
    )

    return np.exp(-gamma * squared_distances)


def solve_dual(X, y, C=1.0, kernel=linear_kernel):
    """
    Решает двойственную задачу SVM по lambda.

    min  1/2 * lambda^T Q lambda - sum(lambda)

    при ограничениях:
        0 <= lambda_i <= C
        sum(lambda_i * y_i) = 0
    """

    n_samples = X.shape[0]

    # Матрица ядра:
    # K_ij = K(x_i, x_j)
    K = kernel(X, X)

    # Q_ij = y_i * y_j * K(x_i, x_j)
    Q = np.outer(y, y) * K

    # scipy.optimize.minimize минимизирует функцию,
    # поэтому используем отрицание двойственной задачи.
    def objective(lambdas):
        return (
            0.5 * lambdas @ Q @ lambdas
            - np.sum(lambdas)
        )

    # Градиент целевой функции
    def gradient(lambdas):
        return Q @ lambdas - np.ones(n_samples)

    # Ограничение:
    # sum(lambda_i * y_i) = 0
    constraints = {
        "type": "eq",
        "fun": lambda lambdas: np.dot(lambdas, y),
        "jac": lambda lambdas: y,
    }

    # Ограничения:
    # 0 <= lambda_i <= C
    bounds = [(0.0, C)] * n_samples

    # Начальное приближение
    initial_lambdas = np.zeros(n_samples)

    result = minimize(
        objective,
        initial_lambdas,
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={
            "maxiter": 500,
            "ftol": 1e-7,
            "disp": True,
        },
    )

    if not result.success:
        raise RuntimeError(
            f"Оптимизация не сошлась: {result.message}"
        )

    return result.x

class SVM:
    def __init__(
        self,
        C=1.0,
        kernel=linear_kernel,
        tol=1e-5,
    ):
        self.C = C
        self.kernel = kernel
        self.tol = tol

        self.lambdas = None
        self.support_vectors = None
        self.support_labels = None
        self.support_lambdas = None

        self.b = None
        self.w = None

    def fit(self, X, y):
        """
        Обучает SVM посредством решения
        двойственной задачи.
        """

        self.lambdas = solve_dual(
            X,
            y,
            C=self.C,
            kernel=self.kernel,
        )

        # Опорные векторы имеют lambda_i > 0
        support_mask = self.lambdas > self.tol

        self.support_vectors = X[support_mask]
        self.support_labels = y[support_mask]
        self.support_lambdas = self.lambdas[support_mask]

        # Для линейного ядра можно явно восстановить:
        # w = sum(lambda_i * y_i * x_i)
        if self.kernel == linear_kernel:
            self.w = (
                self.support_lambdas
                * self.support_labels
            ) @ self.support_vectors

        # Для вычисления b используем в первую очередь
        # опорные векторы, для которых:
        # 0 < lambda_i < C
        margin_mask = (
            (self.lambdas > self.tol)
            & (self.lambdas < self.C - self.tol)
        )

        margin_vectors = X[margin_mask]
        margin_labels = y[margin_mask]

        # Если таких точек нет,
        # используем все опорные векторы.
        if len(margin_vectors) == 0:
            margin_vectors = self.support_vectors
            margin_labels = self.support_labels

        # Значения функции без свободного члена b
        K = self.kernel(
            self.support_vectors,
            margin_vectors,
        )

        decision_without_b = (
            self.support_lambdas
            * self.support_labels
        ) @ K

        # Для опорных векторов на границе зазора,
        # для которых 0 < lambda_i < C:
        #
        # y_i = sum(lambda_j * y_j * K(x_j, x_i)) + b
        #
        # поэтому:
        # b = y_i - sum(lambda_j * y_j * K(x_j, x_i))
        self.b = np.mean(
            margin_labels - decision_without_b
        )

        return self

    def decision_function(self, X):
        """
        Возвращает значение решающей функции:
        f(x) = sum(lambda_i * y_i * K(x_i, x)) + b
        """

        K = self.kernel(
            self.support_vectors,
            X,
        )

        return (
            (self.support_lambdas * self.support_labels) @ K
            + self.b
        )

    def predict(self, X):
        """
        Возвращает класс -1 или +1.
        """

        scores = self.decision_function(X)

        return np.where(scores >= 0, 1, -1)


def print_metrics(
    name,
    y_true,
    y_pred,
    scores,
):
    print(f"\n{name}:")

    print(
        "Accuracy:",
        round(accuracy_score(y_true, y_pred), 4),
    )

    print(
        "Precision:",
        round(precision_score(y_true, y_pred), 4),
    )

    print(
        "Recall:",
        round(recall_score(y_true, y_pred), 4),
    )

    print(
        "F1:",
        round(f1_score(y_true, y_pred), 4),
    )

    print(
        "ROC-AUC:",
        round(roc_auc_score(y_true, scores), 4),
    )
def plot_decision_boundary(
    model,
    X,
    y,
    feature_names,
    title,
    save_path,
):
    """
    Визуализирует решающую границу SVM для двух признаков.

    Показывает:
    - объекты двух классов;
    - границу f(x) = 0;
    - границы зазора f(x) = -1 и f(x) = 1;
    - опорные векторы.
    """

    x_min = X[:, 0].min() - 0.5
    x_max = X[:, 0].max() + 0.5

    y_min = X[:, 1].min() - 0.5
    y_max = X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    grid = np.c_[
        xx.ravel(),
        yy.ravel(),
    ]

    scores = model.decision_function(grid)

    scores = scores.reshape(xx.shape)

    plt.figure(figsize=(8, 6))

    plt.scatter(
        X[y == -1, 0],
        X[y == -1, 1],
        label="Класс -1",
        alpha=0.7,
    )

    plt.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        label="Класс +1",
        alpha=0.7,
    )

    contours = plt.contour(
        xx,
        yy,
        scores,
        levels=[-1, 0, 1],
        linestyles=["--", "-", "--"],
    )

    plt.clabel(
        contours,
        inline=True,
        fontsize=9,
    )
    plt.scatter(
        model.support_vectors[:, 0],
        model.support_vectors[:, 1],
        s=120,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5,
        label="Опорные векторы",
    )

    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title)

    plt.legend()
    plt.grid(alpha=0.2)

    save_path = Path(save_path)
    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=150,
    )

    plt.close()

def main():
    X_train, X_test, y_train, y_test, feature_names = load_data()

    print("Признаки:", list(feature_names))

    print("\nРазмеры выборок:")
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)

    print("\nБаланс классов train:")

    for cls in [-1, 1]:
        count = np.sum(y_train == cls)

        print(
            f"Класс {cls}: "
            f"{count} объектов "
            f"({count / len(y_train):.1%})"
        )

    print("\nПосле стандартизации:")
    print(
        "Средние:",
        np.round(X_train.mean(axis=0), 4),
    )
    print(
        "Std:",
        np.round(X_train.std(axis=0), 4),
    )


    K_linear = linear_kernel(
        X_train[:3],
        X_train[:3],
    )

    K_rbf = rbf_kernel(
        X_train[:3],
        X_train[:3],
    )

    print("\nЛинейное ядро:")
    print(np.round(K_linear, 3))

    print("\nRBF-ядро:")
    print(np.round(K_rbf, 3))


    X_train_small, y_train_small = make_train_subset(
        X_train,
        y_train,
        n_samples=100,
        random_state=42,
    )

    print(
        "\nРазмер подвыборки для собственной SVM:",
        X_train_small.shape,
    )

    print("Баланс классов подвыборки:")

    for cls in [-1, 1]:
        count = np.sum(y_train_small == cls)

        print(
            f"Класс {cls}: "
            f"{count} объектов "
            f"({count / len(y_train_small):.1%})"
        )


    print("\nОбучение собственной линейной SVM:")

    svm = SVM(
        C=1.0,
        kernel=linear_kernel,
    )

    svm.fit(
        X_train_small,
        y_train_small,
    )

    print(
        "Ограничение sum(lambda_i * y_i):",
        np.dot(svm.lambdas, y_train_small),
    )

    print(
        "Опорных векторов:",
        len(svm.support_vectors),
    )

    print("b:", svm.b)

    print(
        "w:",
        np.round(svm.w, 4),
    )


    kernel_scores = svm.decision_function(
        X_train_small
    )

    linear_scores = (
        X_train_small @ svm.w
        + svm.b
    )

    max_difference = np.max(
        np.abs(kernel_scores - linear_scores)
    )

    print(
        "Максимальная разница между "
        "kernel-формой и линейной формой:",
        max_difference,
    )


    reference_svm = SVC(
        C=1.0,
        kernel="linear",
    )

    reference_svm.fit(
        X_train_small,
        y_train_small,
    )


    custom_pred = svm.predict(X_test)
    custom_scores = svm.decision_function(X_test)

    reference_pred = reference_svm.predict(X_test)
    reference_scores = reference_svm.decision_function(
        X_test
    )

    print_metrics(
        "Наша SVM",
        y_test,
        custom_pred,
        custom_scores,
    )

    print_metrics(
        "sklearn SVC",
        y_test,
        reference_pred,
        reference_scores,
    )

    print(
        "\nСовпадение предсказаний:",
        round(
            np.mean(
                custom_pred == reference_pred
            ),
            4,
        ),
    )

    print(
        "Опорных векторов:",
        len(svm.support_vectors),
        "(наша) /",
        len(reference_svm.support_),
        "(sklearn)",
    )


    print("\nОбучение собственной RBF SVM:")

    rbf_svm = SVM(
        C=1.0,
        kernel=rbf_kernel,
    )

    rbf_svm.fit(
        X_train_small,
        y_train_small,
    )

    print(
        "Ограничение sum(lambda_i * y_i):",
        np.dot(rbf_svm.lambdas, y_train_small),
    )

    print(
        "Опорных векторов:",
        len(rbf_svm.support_vectors),
    )

    print("b:", rbf_svm.b)

    reference_rbf = SVC(
        C=1.0,
        kernel="rbf",
        gamma=1.0,
    )

    reference_rbf.fit(
        X_train_small,
        y_train_small,
    )

    custom_rbf_pred = rbf_svm.predict(X_test)
    custom_rbf_scores = rbf_svm.decision_function(X_test)

    reference_rbf_pred = reference_rbf.predict(X_test)
    reference_rbf_scores = reference_rbf.decision_function(
        X_test
    )

    print_metrics(
        "Наша RBF SVM",
        y_test,
        custom_rbf_pred,
        custom_rbf_scores,
    )

    print_metrics(
        "sklearn RBF SVC",
        y_test,
        reference_rbf_pred,
        reference_rbf_scores,
    )

    print(
        "\nСовпадение предсказаний RBF:",
        round(
            np.mean(
                custom_rbf_pred == reference_rbf_pred
            ),
            4,
        ),
    )

    print(
        "Опорных векторов RBF:",
        len(rbf_svm.support_vectors),
        "(наша) /",
        len(reference_rbf.support_),
        "(sklearn)",
    )


    print("\nПостроение визуализаций...")

    X_visual = X_train_small[:, :2]

    visual_feature_names = feature_names[:2]

    linear_visual_svm = SVM(
        C=1.0,
        kernel=linear_kernel,
    )

    linear_visual_svm.fit(
        X_visual,
        y_train_small,
    )

    plot_decision_boundary(
        model=linear_visual_svm,
        X=X_visual,
        y=y_train_small,
        feature_names=visual_feature_names,
        title="Linear SVM",
        save_path=(
            "students/ostapenko_na/lab3/"
            "images/linear_boundary.png"
        ),
    )

    rbf_visual_svm = SVM(
        C=1.0,
        kernel=rbf_kernel,
    )

    rbf_visual_svm.fit(
        X_visual,
        y_train_small,
    )

    plot_decision_boundary(
        model=rbf_visual_svm,
        X=X_visual,
        y=y_train_small,
        feature_names=visual_feature_names,
        title="RBF SVM",
        save_path=(
            "students/ostapenko_na/lab3/"
            "images/rbf_boundary.png"
        ),
    )

    print(
        "Сохранено:",
        "images/linear_boundary.png",
    )

    print(
        "Сохранено:",
        "images/rbf_boundary.png",
    )

    results = pd.DataFrame([
        {
            "model": "Custom Linear SVM",
            "accuracy": accuracy_score(y_test, custom_pred),
            "precision": precision_score(y_test, custom_pred),
            "recall": recall_score(y_test, custom_pred),
            "f1": f1_score(y_test, custom_pred),
            "roc_auc": roc_auc_score(y_test, custom_scores),
            "support_vectors": len(svm.support_vectors),
        },
        {
            "model": "sklearn Linear SVC",
            "accuracy": accuracy_score(y_test, reference_pred),
            "precision": precision_score(y_test, reference_pred),
            "recall": recall_score(y_test, reference_pred),
            "f1": f1_score(y_test, reference_pred),
            "roc_auc": roc_auc_score(y_test, reference_scores),
            "support_vectors": len(reference_svm.support_),
        },
        {
            "model": "Custom RBF SVM",
            "accuracy": accuracy_score(y_test, custom_rbf_pred),
            "precision": precision_score(y_test, custom_rbf_pred),
            "recall": recall_score(y_test, custom_rbf_pred),
            "f1": f1_score(y_test, custom_rbf_pred),
            "roc_auc": roc_auc_score(y_test, custom_rbf_scores),
            "support_vectors": len(rbf_svm.support_vectors),
        },
        {
            "model": "sklearn RBF SVC",
            "accuracy": accuracy_score(y_test, reference_rbf_pred),
            "precision": precision_score(y_test, reference_rbf_pred),
            "recall": recall_score(y_test, reference_rbf_pred),
            "f1": f1_score(y_test, reference_rbf_pred),
            "roc_auc": roc_auc_score(y_test, reference_rbf_scores),
            "support_vectors": len(reference_rbf.support_),
        },
    ])

    results_path = Path(
        "students/ostapenko_na/lab3/results/metrics.csv"
    )

    results_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        results_path,
        index=False,
    )

    print("\nРезультаты:")
    print(results.round(4))

    print(
        "\nСохранено:",
        results_path,
    )
if __name__ == "__main__":
    main()