"""Эксперименты лабораторной работы № 3. SVM."""

from pathlib import Path
from time import perf_counter

import numpy as np
from kernels import (  # noqa: F401
    ConstantKernel,
    LinearKernel,
    PolynomialKernel,
    RBFKernel,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC
from svm import SVM as MY_SVM
from visualization import (
    kernel_name,
    plot_confusion_matrices,
    plot_decision_slice,
    plot_metrics,
)

OUTPUT_DIR = Path(__file__).with_name("report-images")
RANDOM_STATE = 42
C = 1
PREDICT_REPEATS = 10
DATASET = "airline"  # "moons" или "airline"

kernels_grid = [
    ConstantKernel(1),
    LinearKernel(),
    PolynomialKernel(),
    PolynomialKernel(degree=2, coef0=0),
    PolynomialKernel(degree=2, coef0=3),
    PolynomialKernel(degree=3, coef0=1),
    RBFKernel(),
    RBFKernel(gamma=2),
    RBFKernel(gamma=6),
    RBFKernel(gamma=0.5),
    LinearKernel() + PolynomialKernel(degree=3) + 10 * RBFKernel(),
]


def main():
    # 1. выбрать датасет для бинарной классификации;
    from data import load_data, load_simple_data

    if DATASET == "moons":
        X_train, X_test, y_train, y_test, feature_names = load_simple_data(
            random_state=RANDOM_STATE
        )
        plot_features = ["x1", "x2"]
    elif DATASET == "airline":
        X_train, X_test, y_train, y_test, feature_names = load_data(
            sample_fraction=0.01,
            test_size=0.3,
            random_state=RANDOM_STATE,
            return_feature_names=True,
        )
        # Лучшая пара по перебору в find_airline_features.py.
        plot_features = ["Inflight wifi service", "Online boarding"]
    else:
        raise ValueError(f"Unknown dataset: {DATASET}")
    # 2. реализовать решение двойственной задачи по лямбда; для решения задачи использовать
    # [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize)
    # или любую другую библиотеку;
    #
    # Сама двойственная задача и ее градиент записаны в
    # SVM._objective_and_gradient
    # А ограничения - в методе SVM._optimize

    # 3. провернуть трюк с ядром;
    # Для ядер и приколов с ними был даже написан отдельный класс Kernel.
    # В нем я реализовал поддержку синтеза новых ядер путем суммирования ядер с положительными коэффицентами
    # Или перемножении двух ядер
    custom_kernel = LinearKernel() + PolynomialKernel(degree=2)
        # RBFKernel(gamma=2)
      # + ConstantKernel(1)

    # Метод SVM_optimize выдает нам вектор решения - Lambdas.
    # Используя его мы можем найти по формуле веса W и w0
    # W = np.sum(y[i] * x[i] * lambdas[i])

    feature_indices = [feature_names.index(name) for name in plot_features]
    if DATASET == "moons":
        for index, kernel in enumerate(kernels_grid, start=1):
            folder_name = f"{index:02d}_{type(kernel).__name__}"
            output_dir = OUTPUT_DIR / DATASET / folder_name
            run_experiment(
                kernel,
                X_train,
                X_test,
                y_train,
                y_test,
                feature_indices,
                plot_features,
                output_dir,
            )
    else:
        output_dir = OUTPUT_DIR / DATASET
        run_experiment(
            custom_kernel,
            X_train,
            X_test,
            y_train,
            y_test,
            feature_indices,
            plot_features,
            output_dir,
        )


def run_experiment(
    kernel, X_train, X_test, y_train, y_test, feature_indices, plot_features, output_dir
):
    # 4. Построим классификаторы с одним и тем же ядром.
    my_model = MY_SVM(C=C, kernel=kernel)
    reference_model = SVC(C=C, kernel=kernel)

    # 5. Сравним метрики и время работы, потом построим графики.
    models = {"My SVM": my_model, "sklearn SVC": reference_model}
    predictions = {}
    results = {}
    print(
        f"Dataset: {DATASET}\n"
        f"Kernel: {kernel}\n"
        f"Train: {len(y_train)}\n"
        f"Test: {len(y_test)}\n"
        f"C: {C}\n"
        f"Predict repeats: {PREDICT_REPEATS}\n"
    )
    for name, model in models.items():
        started = perf_counter()
        model.fit(X_train, y_train)
        fit_time = perf_counter() - started

        started = perf_counter()
        for _ in range(PREDICT_REPEATS):
            y_pred = model.predict(X_test)
        predict_time = (perf_counter() - started) / PREDICT_REPEATS
        # Среднее на объект при предсказании всей тестовой выборки за один вызов.
        predict_time_per_sample = predict_time / len(X_test)
        predictions[name] = y_pred

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary", pos_label=1, zero_division=0
        )
        accuracy = accuracy_score(y_test, y_pred)
        support_vectors = (
            len(model.support_indices) if model is my_model else model.n_support_.sum()
        )
        results[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Fit time (s)": fit_time,
            "Mean predict time, test set (s)": predict_time,
            "Mean predict time per sample (us)": predict_time_per_sample * 1e6,
            "Support vectors": support_vectors,
        }
        print(
            f"{name}\n"
            f"Accuracy: {accuracy:.4f}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall: {recall:.4f}\n"
            f"F1: {f1:.4f}\n"
            f"Fit time: {fit_time:.4f} s\n"
            f"Mean predict time (test set): {predict_time:.6f} s\n"
            f"Mean predict time per sample: {predict_time_per_sample * 1e6:.2f} us\n"
            f"Support vectors: {support_vectors}\n"
        )
    # Посмотрим совпадут ли результаты нашего SVM, и импортного
    agreement = np.mean(predictions["My SVM"] == predictions["sklearn SVC"])
    print(
        f"My SVM support vectors: {len(my_model.support_indices)}\n"
        f"sklearn SVC support vectors: {reference_model.n_support_.sum()}\n"
        f"My SVM margin width: {my_model.margin_width():.4f}\n"
        f"My SVM w0: {my_model.w0:.4f}\n"
        f"Prediction agreement: {agreement:.4f}\n"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    title = f"Kernel: {kernel_name(kernel)}; C = {C}"
    # 6. Для moons это полная граница, для airline — срез по выбранной паре.
    plot_decision_slice(
        models,
        X_train,
        X_test,
        y_test,
        feature_indices,
        plot_features,
        output_dir / "decision_slice.png",
        title=title,
    )
    plot_confusion_matrices(
        y_test, predictions, output_dir / "confusion_matrices.png", title=title
    )
    plot_metrics(results, output_dir / "metrics.png", title=title)
    print(f"Plots saved to: {output_dir}\n\n\n\n")


if __name__ == "__main__":
    main()
