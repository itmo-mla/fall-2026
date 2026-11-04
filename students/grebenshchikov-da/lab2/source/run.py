from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from data import FEATURES, load_data
from metric_methods import (
    calculate_compactness_profile,
    calculate_loo_risks,
    predict_knn,
    predict_parzen,
    select_prototypes,
)
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"
RANDOM_STATE = 42


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def measure_time(function, repeats: int = 200) -> float:
    for _ in range(5):
        function()

    measurements = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        measurements.append((perf_counter() - start) * 1000)
    return float(np.median(measurements))


def plot_loo_risks(
    knn_risks: np.ndarray,
    parzen_risks: np.ndarray,
    best_knn_k: int,
    best_parzen_k: int,
) -> None:
    k_values = np.arange(1, len(knn_risks) + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(k_values, knn_risks, label="Обычный kNN", color="#4C78A8")
    ax.plot(k_values, parzen_risks, label="Окно Парзена", color="#F58518")
    ax.scatter(best_knn_k, knn_risks[best_knn_k - 1], color="#4C78A8", zorder=3)
    ax.scatter(
        best_parzen_k,
        parzen_risks[best_parzen_k - 1],
        color="#F58518",
        zorder=3,
    )
    ax.annotate(
        f"k={best_knn_k}",
        (best_knn_k, knn_risks[best_knn_k - 1]),
        xytext=(12, 8),
        textcoords="offset points",
        color="#4C78A8",
    )
    ax.annotate(
        f"k={best_parzen_k}",
        (best_parzen_k, parzen_risks[best_parzen_k - 1]),
        xytext=(36, 20),
        textcoords="offset points",
        color="#F58518",
    )
    ax.set(
        title="Ошибка LOO для разных значений k",
        xlabel="Количество соседей k",
        ylabel="Доля ошибок",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "loo_risk.png", dpi=180)
    plt.close(fig)


def plot_compactness(
    full_profile: np.ndarray,
    prototype_profile: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        np.arange(1, len(full_profile) + 1),
        full_profile,
        marker="o",
        label="Вся обучающая выборка",
        color="#4C78A8",
    )
    ax.plot(
        np.arange(1, len(prototype_profile) + 1),
        prototype_profile,
        marker="s",
        label="Выбранные эталоны",
        color="#F58518",
    )
    ax.set(
        title="Профиль компактности",
        xlabel="Номер ближайшего соседа m",
        ylabel="Доля соседей другого класса",
        xticks=np.arange(1, len(full_profile) + 1),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "compactness_profile.png", dpi=180)
    plt.close(fig)


def plot_prototypes(
    X_train: np.ndarray,
    y_train: np.ndarray,
    prototype_indices: np.ndarray,
    class_names: list[str],
) -> None:
    projection = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_train)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    for class_index, class_name in enumerate(class_names):
        class_mask = y_train == class_index
        class_prototypes = prototype_indices[y_train[prototype_indices] == class_index]

        ax.scatter(
            projection[class_mask, 0],
            projection[class_mask, 1],
            s=28,
            alpha=0.35,
            color=colors[class_index],
            label=class_name,
        )
        ax.scatter(
            projection[class_prototypes, 0],
            projection[class_prototypes, 1],
            s=105,
            facecolors=colors[class_index],
            edgecolors="black",
            linewidths=1.2,
        )

    ax.set(
        title="Выбранные эталоны в проекции PCA",
        xlabel="Первая главная компонента",
        ylabel="Вторая главная компонента",
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "prototypes.png", dpi=180)
    plt.close(fig)


def plot_comparison(results: list[dict]) -> None:
    names = [result["name"] for result in results]
    x = np.arange(len(names))
    balanced = [result["metrics"]["balanced_accuracy"] for result in results]
    macro_f1 = [result["metrics"]["macro_f1"] for result in results]
    objects = [result["objects"] for result in results]
    times = [result["time_ms"] for result in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    width = 0.36
    axes[0].bar(
        x - width / 2,
        balanced,
        width,
        label="Сбалансированная точность",
    )
    axes[0].bar(x + width / 2, macro_f1, width, label="F1 по классам")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Качество")
    axes[0].legend(fontsize=8)

    axes[1].bar(x, objects, color="#72B7B2")
    axes[1].set_title("Объектов для предсказания")
    axes[1].set_ylabel("Количество")

    axes[2].bar(x, times, color="#E45756")
    axes[2].set_title("Медианное время")
    axes[2].set_ylabel("мс")

    for ax in axes:
        ax.set_xticks(x, names, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test, class_names = load_data()

    knn_risks = calculate_loo_risks(X_train, y_train, method="knn")
    parzen_risks = calculate_loo_risks(X_train, y_train, method="parzen")
    best_knn_k = int(knn_risks.argmin()) + 1
    best_parzen_k = int(parzen_risks.argmin()) + 1

    prototype_indices, selection_history = select_prototypes(X_train, y_train)
    full_profile = calculate_compactness_profile(X_train, y_train)
    prototype_profile = calculate_compactness_profile(
        X_train,
        y_train,
        prototype_indices=prototype_indices,
    )

    reference = KNeighborsClassifier(
        n_neighbors=best_knn_k,
        weights="uniform",
        algorithm="brute",
        metric="euclidean",
    )
    reference.fit(X_train, y_train)

    methods = [
        (
            "Парзен",
            lambda: predict_parzen(X_train, y_train, X_test, best_parzen_k),
            len(X_train),
        ),
        (
            "Ручной kNN, все",
            lambda: predict_knn(X_train, y_train, X_test, best_knn_k),
            len(X_train),
        ),
        ("sklearn kNN", lambda: reference.predict(X_test), len(X_train)),
        (
            "1NN, эталоны",
            lambda: predict_knn(
                X_train[prototype_indices],
                y_train[prototype_indices],
                X_test,
                1,
            ),
            len(prototype_indices),
        ),
    ]

    results = []
    for name, prediction_function, object_count in methods:
        prediction = prediction_function()
        results.append(
            {
                "name": name,
                "prediction": prediction,
                "metrics": calculate_metrics(y_test, prediction),
                "objects": object_count,
                "time_ms": measure_time(prediction_function),
            }
        )

    manual_knn = next(
        result for result in results if result["name"] == "Ручной kNN, все"
    )
    sklearn_knn = next(result for result in results if result["name"] == "sklearn kNN")
    agreement = np.mean(manual_knn["prediction"] == sklearn_knn["prediction"])

    plot_loo_risks(knn_risks, parzen_risks, best_knn_k, best_parzen_k)
    plot_compactness(full_profile, prototype_profile)
    plot_prototypes(X_train, y_train, prototype_indices, class_names)
    plot_comparison(results)

    print(f"Признаки: {len(FEATURES)}")
    print(f"Train: {len(X_train)}, test: {len(X_test)}")
    print(f"Классы train: {np.bincount(y_train).tolist()}")
    print(f"Классы test: {np.bincount(y_test).tolist()}")
    print(f"Обычный kNN: k={best_knn_k}, LOO={knn_risks[best_knn_k - 1]:.4f}")
    print(f"Парзен: k={best_parzen_k}, LOO={parzen_risks[best_parzen_k - 1]:.4f}")
    print(
        f"Эталоны: {len(prototype_indices)} из {len(X_train)}, "
        f"CCV(q=1) {selection_history[0, 1]:.4f} -> "
        f"{selection_history[-1, 1]:.4f}"
    )
    print(f"Совпадение ручного kNN и sklearn: {agreement:.4f}")

    for result in results:
        metrics = result["metrics"]
        print(
            f"{result['name']}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}, "
            f"objects={result['objects']}, "
            f"time={result['time_ms']:.4f} ms"
        )


if __name__ == "__main__":
    main()
