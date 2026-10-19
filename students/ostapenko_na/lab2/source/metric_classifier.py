import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.2

def prepare_data():
    data = load_breast_cancer()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Инвертируем классы после разбиения:
    # 0 = benign, 1 = malignant
    y_train = 1 - y_train
    y_test = 1 - y_test

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, data.feature_names

def euclidean_distances(x, X):
    return np.sqrt(np.sum((X - x) ** 2, axis=1))

def gaussian_kernel(r):
    return np.exp(-2 * r ** 2)

def parzen_predict(X, X_train, y_train, k):
    if k < 1 or k >= len(X_train):
        raise ValueError(
            "k must satisfy 1 <= k < len(X_train)"
        )

    X = np.atleast_2d(X)
    predictions = []
    for x in X:
        distances = euclidean_distances(x, X_train)
        sorted_indices = np.argsort(distances)
        # Ширина окна = расстояние до (k + 1)-го соседа
        h = distances[sorted_indices[k]]
        if h == 0:
            h = 1e-10
        weights = gaussian_kernel(distances / h)
        classes = np.unique(y_train)
        class_scores = []
        for cls in classes:
            score = np.sum(weights[y_train == cls])
            class_scores.append(score)
        predictions.append(classes[np.argmax(class_scores)])
    return np.array(predictions)

def loo_score(X, y, k):
    errors = 0
    for i in range(len(X)):
        # Исключаем i-й объект
        X_train_loo = np.delete(X, i, axis=0)
        y_train_loo = np.delete(y, i)
        prediction = parzen_predict(
    X[i],
    X_train_loo,
    y_train_loo,
    k,
)[0]
        if prediction != y[i]:
            errors += 1
    return errors / len(X)
def find_best_k(X, y, k_values):
    risks = []
    for k in k_values:
        risk = loo_score(X, y, k)
        risks.append(risk)
        print(f"k={k:2d}, LOO risk={risk:.4f}")
    best_index = np.argmin(risks)
    best_k = k_values[best_index]
    return best_k, risks

def plot_loo_risk(k_values, risks, best_k):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, risks, marker="o", markersize=4)
    plt.axvline(
        best_k,
        linestyle="--",
        label=f"Best k = {best_k}",
    )
    plt.xlabel("k")
    plt.ylabel("LOO risk")
    plt.title("LOO empirical risk depending on k")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "loo_risk.png", dpi=150)
    plt.close()

 ## эталоны   
def nearest_neighbor_predict(x, X_prototypes, y_prototypes):
    distances = euclidean_distances(x, X_prototypes)
    nearest_index = np.argmin(distances)
    return y_prototypes[nearest_index]

def prototype_ccv(X, y, prototype_indices):
    errors = 0

    for i in range(len(X)):
        # При LOO проверяемый объект
        # не может использовать сам себя
        available_indices = [
            idx
            for idx in prototype_indices
            if idx != i
        ]

        if len(available_indices) == 0:
            errors += 1
            continue

        prediction = nearest_neighbor_predict(
            X[i],
            X[available_indices],
            y[available_indices],
        )

        if prediction != y[i]:
            errors += 1

    return errors / len(X)

def select_prototypes(X, y):
    classes = np.unique(y)
    prototype_indices = []

    for cls in classes:
        first_index = np.where(y == cls)[0][0]
        prototype_indices.append(first_index)

    current_ccv = prototype_ccv(
        X,
        y,
        prototype_indices,
    )

    print(
        f"Начало: {len(prototype_indices)} эталона, "
        f"CCV = {current_ccv:.4f}"
    )

    while True:
        best_index = None
        best_ccv = current_ccv

        for i in range(len(X)):
            if i in prototype_indices:
                continue

            candidate_indices = (
                prototype_indices + [i]
            )

            candidate_ccv = prototype_ccv(
                X,
                y,
                candidate_indices,
            )

            if candidate_ccv < best_ccv:
                best_ccv = candidate_ccv
                best_index = i

        # Останавливаемся, если CCV
        # больше нельзя уменьшить
        if best_index is None:
            break

        prototype_indices.append(best_index)
        current_ccv = best_ccv

        print(
            f"Эталонов: {len(prototype_indices)}, "
            f"CCV = {current_ccv:.4f}"
        )

    return np.array(prototype_indices)

def plot_prototypes(X, y, prototype_indices):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    classes = np.unique(y)

    plt.figure(figsize=(8, 6))

    # Все объекты обучающей выборки
    for cls in classes:
        mask = y == cls

        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            alpha=0.3,
            s=25,
            label=f"Class {cls}",
        )

    # Выбранные эталоны
    plt.scatter(
        X_2d[prototype_indices, 0],
        X_2d[prototype_indices, 1],
        facecolors="none",
        edgecolors="black",
        s=100,
        linewidths=1.5,
        label="Prototypes",
    )

    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.title("Selected prototypes (PCA projection)")
    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(
        IMAGES_DIR / "prototypes.png",
        dpi=150,
    )
    plt.close()

def plot_metrics_comparison(
    custom_metrics,
    sklearn_metrics,
    prototype_metrics,
):
    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
    ]

    x = np.arange(len(metric_names))
    width = 0.25

    plt.figure(figsize=(9, 6))

    plt.bar(
        x - width,
        custom_metrics,
        width,
        label="Parzen (all objects)",
    )

    plt.bar(
        x,
        sklearn_metrics,
        width,
        label="sklearn KNN",
    )

    plt.bar(
        x + width,
        prototype_metrics,
        width,
        label="Parzen (prototypes)",
    )

    plt.xticks(x, metric_names)
    plt.ylabel("Score")
    plt.ylim(0.85, 1.0)
    plt.title("Classification quality comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        IMAGES_DIR / "metrics_comparison.png",
        dpi=150,
    )
    plt.close()

def save_metrics(
    custom_metrics,
    sklearn_metrics,
    prototype_metrics,
):
    rows = [
        ["Parzen (all objects)", *custom_metrics],
        ["sklearn KNN", *sklearn_metrics],
        ["Parzen (prototypes)", *prototype_metrics],
    ]

    with open(
        RESULTS_DIR / "metrics.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]
        )

        writer.writerows(rows)

def main():
    X_train, X_test, y_train, y_test, feature_names = prepare_data()
    print("Признаки:", feature_names)
    print("Train:", X_train.shape)
    print("Test:", X_test.shape)
    classes, counts = np.unique(y_train, return_counts=True)
    class_names = {
    0: "benign",
    1: "malignant",
}

    print("\nБаланс классов:")
    for cls, count in zip(classes, counts):
        print(
            f"Класс {cls} ({class_names[cls]}): "
            f"{count} объектов "
            f"({count / len(y_train) * 100:.1f}%)"
        )
    # Подбираем k с помощью LOO
    k_values = list(range(1, 31))
    print("\nПодбор k с помощью LOO:")
    best_k, loo_risks = find_best_k(
        X_train,
        y_train,
        k_values,
    )
    print(f"\nЛучшее k: {best_k}")
    print(f"Минимальный LOO risk: {min(loo_risks):.4f}")
    # Строим график LOO-риска
    plot_loo_risk(k_values, loo_risks, best_k)
    # Классифицируем всю тестовую выборку
    y_pred = parzen_predict(
        X_test,
        X_train,
        y_train,
        best_k,
    )
    # Считаем метрики
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("\nМетрики собственного классификатора:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
        # Эталонный KNN из sklearn
    sklearn_knn = KNeighborsClassifier(
        n_neighbors=best_k,
        metric="euclidean",
    )
    sklearn_knn.fit(X_train, y_train)
    y_pred_sklearn = sklearn_knn.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, y_pred_sklearn)
    sklearn_precision = precision_score(y_test, y_pred_sklearn)
    sklearn_recall = recall_score(y_test, y_pred_sklearn)
    sklearn_f1 = f1_score(y_test, y_pred_sklearn)
    print("\nМетрики sklearn KNN:")
    print(f"Accuracy:  {sklearn_accuracy:.4f}")
    print(f"Precision: {sklearn_precision:.4f}")
    print(f"Recall:    {sklearn_recall:.4f}")
    print(f"F1:        {sklearn_f1:.4f}")

    prototype_indices = select_prototypes(
        X_train,
        y_train,
    )
    X_prototypes = X_train[prototype_indices]
    y_prototypes = y_train[prototype_indices]
    print("\nОтбор эталонов:")
    print(f"Объектов до отбора: {len(X_train)}")
    print(f"Эталонов после отбора: {len(X_prototypes)}")
    print(
        f"Осталось: "
        f"{len(X_prototypes) / len(X_train) * 100:.1f}%"
    )
    y_pred_prototypes = parzen_predict(
    X_test,
    X_prototypes,
    y_prototypes,
    best_k,
)
    prototype_accuracy = accuracy_score(
        y_test,
        y_pred_prototypes,
    )
    prototype_precision = precision_score(
        y_test,
        y_pred_prototypes,
    )
    prototype_recall = recall_score(
        y_test,
        y_pred_prototypes,
    )
    prototype_f1 = f1_score(
        y_test,
        y_pred_prototypes,
    )

    plot_prototypes(
    X_train,
    y_train,
    prototype_indices,
)
    
    print("\nМетрики после отбора эталонов:")
    print(f"Accuracy:  {prototype_accuracy:.4f}")
    print(f"Precision: {prototype_precision:.4f}")
    print(f"Recall:    {prototype_recall:.4f}")
    print(f"F1:        {prototype_f1:.4f}")

    custom_metrics = [
    accuracy,
    precision,
    recall,
    f1,
]

    sklearn_metrics = [
        sklearn_accuracy,
        sklearn_precision,
        sklearn_recall,
        sklearn_f1,
    ]

    prototype_metrics = [
        prototype_accuracy,
        prototype_precision,
        prototype_recall,
        prototype_f1,
    ]

    plot_metrics_comparison(
        custom_metrics,
        sklearn_metrics,
        prototype_metrics,
    )

    save_metrics(
    custom_metrics,
    sklearn_metrics,
    prototype_metrics,
)
if __name__ == "__main__":
    main()