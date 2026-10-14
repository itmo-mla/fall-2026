import math
import os
import time

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2


K_VALUES = range(1, 21)

CCV_TRAIN_FRACTION = 0.5

CCV_TAIL = 1e-4


CCV_TOLERANCE = 0.01

CCV_EPS = 1e-12

NEIGHBORS_DEPTH = 1000



print("=" * 70)
print("ЗАГРУЗКА DRY BEAN DATASET")
print("=" * 70)

dataset_path = kagglehub.dataset_download(
    "muratkokludataset/dry-bean-dataset"
)

dataset_file = None

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".xlsx"):
            dataset_file = os.path.join(root, file)
            break

    if dataset_file is not None:
        break

if dataset_file is None:
    raise FileNotFoundError("Файл Dry Bean Dataset не найден")

data = pd.read_excel(dataset_file)

print("Размерность данных:", data.shape)
print("\nПервые 5 строк:")
print(data.head())

print("\nРаспределение классов:")
print(data["Class"].value_counts())



def stratified_split(y, test_size, random_state):
    rng = np.random.default_rng(random_state)

    train_indices = []
    test_indices = []

    for label in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == label))
        n_test = int(round(test_size * len(indices)))

        test_indices.append(indices[:n_test])
        train_indices.append(indices[n_test:])

    return (
        np.sort(np.concatenate(train_indices)),
        np.sort(np.concatenate(test_indices)),
    )


X = data.drop(columns=["Class"]).to_numpy(dtype=float)
y = data["Class"].to_numpy()


classes, y_encoded = np.unique(y, return_inverse=True)

train_indices, test_indices = stratified_split(
    y_encoded,
    TEST_SIZE,
    RANDOM_STATE,
)

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y_encoded[train_indices]
y_test = y_encoded[test_indices]


train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)

if np.any(train_std == 0):
    raise ValueError(
        "Постоянные признаки: "
        f"{np.flatnonzero(train_std == 0).tolist()}"
    )

X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std

print("\nОбучающая выборка:", X_train.shape)
print("Тестовая выборка:", X_test.shape)
print("Классы:", classes)




def confusion_matrix(y_true, y_pred, n_classes):

    cm = np.zeros((n_classes, n_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1

    return cm


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def balanced_accuracy_score(y_true, y_pred):

    recalls = [
        np.mean(y_pred[y_true == label] == label)
        for label in np.unique(y_true)
    ]

    return np.mean(recalls)



def gaussian_kernel(r):

    return np.exp(-2 * r**2)



def euclidean_distances(x, X):
    return np.sqrt(np.sum((X - x) ** 2, axis=1))


def predict_one(x, X_train, y_train, k):
    distances = euclidean_distances(x, X_train)

    neighbors_count = min(k + 1, len(X_train))

    nearest_indices = np.argpartition(
        distances,
        neighbors_count - 1,
    )[:neighbors_count]

    nearest_distances = distances[nearest_indices]

    order = np.argsort(nearest_distances)

    nearest_indices = nearest_indices[order]
    nearest_distances = nearest_distances[order]

    if len(nearest_indices) > k:
        voting_indices = nearest_indices[:k]
        voting_distances = nearest_distances[:k]
        h = nearest_distances[k]
    else:
        voting_indices = nearest_indices
        voting_distances = nearest_distances
        h = nearest_distances[-1]

    class_weights = {}

    for index, distance in zip(
        voting_indices,
        voting_distances,
    ):
        if h > 0:
            r = distance / h
        else:
            r = 0

        weight = gaussian_kernel(r)
        label = y_train[index]

        class_weights[label] = (
            class_weights.get(label, 0) + weight
        )

    predicted_class = max(
        class_weights.items(),
        key=lambda item: item[1],
    )[0]

    return predicted_class


def predict(X, X_train, y_train, k):
    predictions = []

    for x in X:
        predicted_class = predict_one(
            x,
            X_train,
            y_train,
            k,
        )

        predictions.append(predicted_class)

    return np.array(predictions)


def loo_risk(X, y, k):
    errors = 0

    for i in range(len(X)):
        X_without_i = np.delete(X, i, axis=0)
        y_without_i = np.delete(y, i)

        predicted_class = predict_one(
            X[i],
            X_without_i,
            y_without_i,
            k,
        )

        if predicted_class != y[i]:
            errors += 1

    return errors / len(X)


print("\n" + "=" * 70)
print("ПОДБОР k МЕТОДОМ LOO")
print("=" * 70)

loo_risks = []

for k in K_VALUES:
    print(f"k = {k:2d}", end=" | ")

    risk = loo_risk(
        X_train,
        y_train,
        k,
    )

    loo_risks.append(risk)

    print(
        f"эмпирический риск = {risk:.4f}, "
        f"accuracy = {1 - risk:.4f}"
    )


best_k_index = np.argmin(loo_risks)
best_k = list(K_VALUES)[best_k_index]
best_risk = loo_risks[best_k_index]

print("\nЛучший параметр:")
print(f"k = {best_k}")
print(f"LOO risk = {best_risk:.4f}")
print(f"LOO accuracy = {1 - best_risk:.4f}")


plt.figure(figsize=(10, 6))

plt.plot(
    list(K_VALUES),
    loo_risks,
    "bo-",
    linewidth=2,
    markersize=6,
    label="Эмпирический риск LOO(k)",
)

plt.axvline(
    best_k,
    color="red",
    linestyle="--",
    label=f"Лучшее k = {best_k}",
)

plt.xlabel("Количество соседей k")
plt.ylabel("Эмпирический риск LOO")
plt.title("Зависимость эмпирического риска от k")
plt.xticks(list(K_VALUES))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(PLOTS_DIR, "1_loo_risk.png"),
    dpi=150,
    bbox_inches="tight",
)

plt.close()


print("\n" + "=" * 70)
print("СОБСТВЕННАЯ РЕАЛИЗАЦИЯ KNN")
print("=" * 70)

start_time = time.time()

my_predictions = predict(
    X_test,
    X_train,
    y_train,
    best_k,
)

my_time = time.time() - start_time

my_accuracy = accuracy_score(
    y_test,
    my_predictions,
)

my_balanced_accuracy = balanced_accuracy_score(
    y_test,
    my_predictions,
)

print(f"Accuracy:          {my_accuracy:.4f}")
print(
    f"Balanced accuracy: "
    f"{my_balanced_accuracy:.4f}"
)
print(f"Время:             {my_time:.4f} сек")



print("\n" + "=" * 70)
print("СРАВНЕНИЕ С SKLEARN KNN")
print("=" * 70)

sklearn_knn = KNeighborsClassifier(
    n_neighbors=best_k,
    metric="euclidean",
    weights="distance",
)

sklearn_knn.fit(
    X_train,
    y_train,
)

start_time = time.time()

sklearn_predictions = sklearn_knn.predict(X_test)

sklearn_time = time.time() - start_time

sklearn_accuracy = accuracy_score(
    y_test,
    sklearn_predictions,
)

sklearn_balanced_accuracy = balanced_accuracy_score(
    y_test,
    sklearn_predictions,
)

print(
    f"{'Метрика':<25} | "
    f"{'Наш KNN':<12} | "
    f"{'Sklearn':<12}"
)

print("-" * 56)

print(
    f"{'Accuracy':<25} | "
    f"{my_accuracy:<12.4f} | "
    f"{sklearn_accuracy:<12.4f}"
)

print(
    f"{'Balanced accuracy':<25} | "
    f"{my_balanced_accuracy:<12.4f} | "
    f"{sklearn_balanced_accuracy:<12.4f}"
)

print(
    f"{'Время, сек':<25} | "
    f"{my_time:<12.4f} | "
    f"{sklearn_time:<12.4f}"
)


my_confusion = confusion_matrix(
    y_test,
    my_predictions,
    len(classes),
)

sklearn_confusion = confusion_matrix(
    y_test,
    sklearn_predictions,
    len(classes),
)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6),
)

image1 = axes[0].imshow(my_confusion)

axes[0].set_title("Собственный KNN")
axes[0].set_xlabel("Предсказанный класс")
axes[0].set_ylabel("Истинный класс")

for i in range(len(my_confusion)):
    for j in range(len(my_confusion)):
        axes[0].text(
            j,
            i,
            my_confusion[i, j],
            ha="center",
            va="center",
        )

image2 = axes[1].imshow(sklearn_confusion)

axes[1].set_title("Sklearn KNN")
axes[1].set_xlabel("Предсказанный класс")
axes[1].set_ylabel("Истинный класс")

for i in range(len(sklearn_confusion)):
    for j in range(len(sklearn_confusion)):
        axes[1].text(
            j,
            i,
            sklearn_confusion[i, j],
            ha="center",
            va="center",
        )

fig.colorbar(image1, ax=axes[0], label="Число объектов")
fig.colorbar(image2, ax=axes[1], label="Число объектов")

fig.legend(
    handles=[
        plt.Line2D([], [], linestyle="none", label=f"{code} — {name}")
        for code, name in enumerate(classes)
    ],
    title="Номер класса",
    loc="lower center",
    ncol=len(classes),
    bbox_to_anchor=(0.5, -0.12),
    handlelength=0,
    handletextpad=0,
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "2_confusion_matrices.png",
    ),
    dpi=150,
    bbox_inches="tight",
)

plt.close()




def ccv_gamma(L, train_size, tail):
    def log_comb(n, m):
        return (
            math.lgamma(n + 1)
            - math.lgamma(m + 1)
            - math.lgamma(n - m + 1)
        )

    log_denominator = log_comb(L - 1, train_size)
    gamma = []

    for j in range(1, L - train_size + 1):
        gamma.append(
            math.exp(
                log_comb(L - 1 - j, train_size - 1)
                - log_denominator
            )
        )

        if 1 - sum(gamma) < tail:
            break

    return np.array(gamma)


def error_indicators(i, neighbor_indices, y, length):
    e = np.ones(length)
    count = min(len(neighbor_indices), length)
    e[:count] = y[neighbor_indices[:count]] != y[i]
    return e


def ccv_risk(X, y, prototype_indices, gamma):
    K = len(gamma)
    total = 0.0

    for i in range(len(X)):

        current_indices = prototype_indices[
            prototype_indices != i
        ]

        distances = euclidean_distances(
            X[i],
            X[current_indices],
        )

        neighbors = current_indices[
            np.argsort(distances, kind="stable")[:K]
        ]

        e = error_indicators(i, neighbors, y, K)
        total += e @ gamma

    return total / len(X)


def sorted_neighbors(X, depth):
    depth = min(depth, len(X) - 1)
    result = np.empty((len(X), depth), dtype=np.int64)

    for i in range(len(X)):
        distances = euclidean_distances(X[i], X)
        distances[i] = np.inf

        nearest = np.argpartition(distances, depth - 1)[:depth]
        result[i] = nearest[
            np.argsort(distances[nearest], kind="stable")
        ]

    return result




def greedy_prototype_selection(X, y, gamma):
    L = len(X)
    K = len(gamma)

    order = sorted_neighbors(X, NEIGHBORS_DEPTH)
    in_omega = np.ones(L, dtype=bool)

    neighbors = [None] * L
    row_delta = [None] * L
    watchers = [set() for _ in range(L)]

    T = np.zeros(L)
    delta_ccv = np.zeros(L)

    def nearest_prototypes(i):
        candidates = order[i][in_omega[order[i]]]

        if len(candidates) >= K + 1:
            return candidates[:K + 1]

        prototype_indices = np.flatnonzero(in_omega)
        prototype_indices = prototype_indices[prototype_indices != i]

        distances = euclidean_distances(
            X[i],
            X[prototype_indices],
        )

        return prototype_indices[
            np.argsort(distances, kind="stable")[:K + 1]
        ]

    def refresh(i):
        if neighbors[i] is not None:
            old = neighbors[i]
            delta_ccv[old[:K]] -= row_delta[i][:len(old[:K])]

            for index in old:
                watchers[index].discard(i)

        current = nearest_prototypes(i)
        e = error_indicators(i, current, y, K + 1)

        T[i] = e[:K] @ gamma

        steps = (e[1:] - e[:-1]) * gamma
        delta = np.cumsum(steps[::-1])[::-1]

        neighbors[i] = current
        row_delta[i] = delta

        delta_ccv[current[:K]] += delta[:len(current[:K])]

        for index in current:
            watchers[index].add(i)

    for i in range(L):
        refresh(i)

    current_ccv = T.sum() / L
    best_ccv = current_ccv

    ccv_history = [current_ccv]
    size_history = [L]

    class_counts = np.bincount(y)

    print("\nНачальный набор:")
    print(f"|Omega| = {L}")
    print(f"CCV(Omega) = {current_ccv:.4f}")
    print(f"Глубина профиля K = {K}, Г(1) = {gamma[0]:.4f}")

    while True:
        candidate_delta = np.where(
            in_omega & (class_counts[y] > 1),
            delta_ccv,
            np.inf,
        )

        best_candidate = int(np.argmin(candidate_delta))

        if not np.isfinite(candidate_delta[best_candidate]):
            print("Удалять больше нечего.")
            break

        candidate_ccv = (
            current_ccv
            + candidate_delta[best_candidate] / L
        )

        if candidate_ccv > best_ccv + CCV_TOLERANCE + CCV_EPS:
            print(
                "CCV вырос больше допустимого "
                f"({candidate_ccv:.4f} > {best_ccv:.4f} + "
                f"{CCV_TOLERANCE}) — остановка."
            )
            break

        in_omega[best_candidate] = False
        class_counts[y[best_candidate]] -= 1

        for i in list(watchers[best_candidate]):
            refresh(i)

        current_ccv = T.sum() / L
        best_ccv = min(best_ccv, current_ccv)

        size_history.append(int(in_omega.sum()))
        ccv_history.append(current_ccv)

        if len(size_history) % 500 == 0:
            print(
                f"|Omega| = {size_history[-1]}, "
                f"CCV(Omega) = {current_ccv:.4f}"
            )

    prototypes = np.flatnonzero(in_omega)

    return (
        prototypes,
        size_history,
        ccv_history,
    )


print("\n" + "=" * 70)
print("ЖАДНЫЙ ОТБОР ЭТАЛОНОВ")
print("КРИТЕРИЙ CCV(Omega) -> min")
print("=" * 70)

ccv_weights = ccv_gamma(
    len(X_train),
    int(round(CCV_TRAIN_FRACTION * len(X_train))),
    CCV_TAIL,
)

selection_start = time.time()

(
    prototype_indices,
    prototype_sizes,
    ccv_history,
) = greedy_prototype_selection(
    X_train,
    y_train,
    ccv_weights,
)

selection_time = time.time() - selection_start


direct_ccv = ccv_risk(
    X_train,
    y_train,
    prototype_indices,
    ccv_weights,
)

if abs(direct_ccv - ccv_history[-1]) > 1e-9:
    raise AssertionError(
        f"CCV по определению {direct_ccv} "
        f"≠ инкрементальный {ccv_history[-1]}"
    )

X_prototypes = X_train[prototype_indices]
y_prototypes = y_train[prototype_indices]

print("\nРезультат отбора:")
print(
    f"Исходное количество объектов: "
    f"{len(X_train)}"
)

print(
    f"Количество эталонов: "
    f"{len(X_prototypes)}"
)

compression = (
    len(X_prototypes)
    / len(X_train)
    * 100
)

print(
    f"Осталось данных: "
    f"{compression:.2f}%"
)

print(
    f"CCV: {ccv_history[0]:.4f} -> {ccv_history[-1]:.4f} "
    f"(проверено по определению), время {selection_time:.1f} сек"
)



plt.figure(figsize=(10, 6))

plt.plot(
    prototype_sizes,
    ccv_history,
    "g-",
    linewidth=2,
    label="CCV(Omega) после каждого удаления",
)

plt.xlabel("Количество эталонов |Omega|")
plt.ylabel("CCV(Omega)")
plt.title(
    "Жадный отбор эталонов по критерию "
    "CCV(Omega) -> min"
)

plt.xscale("log")
plt.gca().invert_xaxis()
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "3_prototype_ccv.png",
    ),
    dpi=150,
    bbox_inches="tight",
)

plt.close()


X_centered = X_train - X_train.mean(axis=0)

_, singular_values, components = np.linalg.svd(
    X_centered,
    full_matrices=False,
)

explained_variance_ratio = (
    singular_values**2
    / np.sum(singular_values**2)
)

X_train_pca = X_centered @ components[:2].T

X_prototypes_pca = X_train_pca[
    prototype_indices
]

plt.figure(figsize=(12, 8))

train_scatter = plt.scatter(
    X_train_pca[:, 0],
    X_train_pca[:, 1],
    c=y_train,
    alpha=0.2,
    s=20,
)

plt.scatter(
    X_prototypes_pca[:, 0],
    X_prototypes_pca[:, 1],
    c=y_prototypes,
    edgecolors="black",
    linewidth=1,
    s=70,
    label="Эталоны",
)

plt.xlabel(
    "Первая главная компонента "
    f"({explained_variance_ratio[0]:.1%})"
)

plt.ylabel(
    "Вторая главная компонента "
    f"({explained_variance_ratio[1]:.1%})"
)

plt.title(
    f"Отобранные эталоны: "
    f"{len(prototype_indices)} из "
    f"{len(X_train)}"
)

class_handles, _ = train_scatter.legend_elements(alpha=1)

prototype_handle = plt.Line2D(
    [],
    [],
    marker="o",
    linestyle="none",
    markerfacecolor="white",
    markeredgecolor="black",
    markersize=9,
)

plt.legend(
    class_handles + [prototype_handle],
    list(classes) + ["Эталон (крупная точка с обводкой)"],
    title="Цвет — класс",
    loc="best",
)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "4_prototypes_pca.png",
    ),
    dpi=150,
    bbox_inches="tight",
)

plt.close()


print("\n" + "=" * 70)
print("KNN С ОТОБРАННЫМИ ЭТАЛОНАМИ")
print("=" * 70)

removed_mask = np.ones(len(X_train), dtype=bool)
removed_mask[prototype_indices] = False

prototype_k_values = [
    k for k in K_VALUES
    if k + 1 <= len(X_prototypes)
]

holdout_risks = []

for k in prototype_k_values:
    holdout_predictions = predict(
        X_train[removed_mask],
        X_prototypes,
        y_prototypes,
        k,
    )

    holdout_risks.append(
        np.mean(holdout_predictions != y_train[removed_mask])
    )

prototype_k = prototype_k_values[int(np.argmin(holdout_risks))]

print(
    f"k для эталонов: {prototype_k} "
    f"(ошибка на удалённых объектах "
    f"{min(holdout_risks):.4f})"
)

start_time = time.time()

prototype_predictions = predict(
    X_test,
    X_prototypes,
    y_prototypes,
    prototype_k,
)

prototype_time = time.time() - start_time

prototype_accuracy = accuracy_score(
    y_test,
    prototype_predictions,
)

prototype_balanced_accuracy = (
    balanced_accuracy_score(
        y_test,
        prototype_predictions,
    )
)


print("\n" + "=" * 80)
print("СРАВНЕНИЕ KNN С И БЕЗ ОТБОРА ЭТАЛОНОВ")
print("=" * 80)

print(
    f"{'Метрика':<30} | "
    f"{'Обычный KNN':<15} | "
    f"{'С эталонами':<15}"
)

print("-" * 68)

print(
    f"{'Accuracy':<30} | "
    f"{my_accuracy:<15.4f} | "
    f"{prototype_accuracy:<15.4f}"
)

print(
    f"{'Balanced accuracy':<30} | "
    f"{my_balanced_accuracy:<15.4f} | "
    f"{prototype_balanced_accuracy:<15.4f}"
)

print(
    f"{'Количество объектов':<30} | "
    f"{len(X_train):<15} | "
    f"{len(X_prototypes):<15}"
)

print(
    f"{'k':<30} | "
    f"{best_k:<15} | "
    f"{prototype_k:<15}"
)

print(
    f"{'Время предсказания':<30} | "
    f"{my_time:<15.4f} | "
    f"{prototype_time:<15.4f}"
)

if prototype_time > 0:
    print(
        f"\nУскорение предсказания: "
        f"{my_time / prototype_time:.2f}x"
    )


models = [
    "KNN",
    "KNN + эталоны",
    "sklearn KNN",
]

accuracies = [
    my_accuracy,
    prototype_accuracy,
    sklearn_accuracy,
]

plt.figure(figsize=(9, 6))

bars = plt.bar(
    models,
    accuracies,
)

plt.legend(
    bars,
    [
        f"{model}: {accuracy:.4f}"
        for model, accuracy in zip(models, accuracies)
    ],
    title="Accuracy на тесте",
    loc="lower right",
)

plt.ylabel("Accuracy")
plt.title("Сравнение качества алгоритмов")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "5_accuracy_comparison.png",
    ),
    dpi=150,
    bbox_inches="tight",
)

plt.close()


print("\n" + "=" * 70)
print("ГОТОВО")
print("=" * 70)

print(f"Оптимальное k: {best_k}")

print(
    f"Собственный KNN: "
    f"{my_accuracy:.4f}"
)

print(
    f"Sklearn KNN: "
    f"{sklearn_accuracy:.4f}"
)

print(
    f"KNN с эталонами: "
    f"{prototype_accuracy:.4f}"
)

print(
    f"Эталонов: "
    f"{len(X_prototypes)} / {len(X_train)}"
)

print(
    f"Графики сохранены в '{PLOTS_DIR}'"
)