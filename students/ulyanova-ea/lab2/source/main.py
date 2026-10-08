from itertools import combinations
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data():
    path = kagglehub.dataset_download("mastershomya/banknote-authetication")

    df = pd.read_csv(Path(path) / "data_banknote_authentication.txt").drop_duplicates().reset_index(drop=True)

    names = df.drop(columns="class").columns.to_list()
    X = df.drop(columns="class").to_numpy(dtype=float)
    y = np.where(df["class"].to_numpy() == 1, 1, -1)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train_raw, X_train, X_test, y_train, y_test, names


def parzen_predict(d, y, k):
    h = np.partition(d, k, axis=1)[:, k, None]

    W = np.exp(-2 * (d / np.where(h > 0, h, 1)) ** 2)
    W = np.where(h > 0, W, d == 0)

    v = W @ y
    return np.where(v > 0, 1, -1)


def loo_distances(X):
    n = len(X)
    d = np.empty((n, n))

    for i in range(n):
        scaler = StandardScaler().fit(np.delete(X, i, axis=0))
        X_i = scaler.transform(X)
        x = X_i[i:i + 1]
        distances = cdist(x, X_i)
        d[i] = distances[0]

    np.fill_diagonal(d, np.inf)
    return d


def choose_k(d, y):
    ks = np.arange(1, min(50, len(y) - 2) + 1)
    errors = []

    for k in ks:
        y_pred = parzen_predict(d, y, k)
        error = np.count_nonzero(y_pred != y)
        errors.append(error)

    errors = np.array(errors)
    k = int(ks[np.argmin(errors)])

    return k, ks, errors


def select_reference(d, y, min_size=3):
    ids = np.arange(len(y))
    rows = np.arange(len(y))
    history = []

    while True:
        d_i = d[:, ids].copy()

        first = np.argmin(d_i, axis=1)
        y_pred = y[ids[first]]

        wrong = (y_pred != y).astype(int)
        error = int(wrong.sum())

        history.append((len(ids), error))

        if len(ids) <= min_size:
            break

        d_i[rows, first] = np.inf

        second = np.argmin(d_i, axis=1)
        y_second = y[ids[second]]
        wrong_second = (y_second != y).astype(int)

        errors = np.empty(len(ids))

        for j in range(len(ids)):
            wrong_after = wrong.copy()

            mask = first == j
            wrong_after[mask] = wrong_second[mask]

            errors[j] = wrong_after.sum()

        for label in np.unique(y):
            positions = np.flatnonzero(y[ids] == label)

            if len(positions) == 1:
                errors[positions[0]] = np.inf

        best = int(np.argmin(errors))

        if errors[best] > error:
            break

        ids = np.delete(ids, best)

    return ids, np.array(history)


def plot_loo(ks, errors, n, path):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(ks, errors / n, marker=".", label="Окно Парзена")
    ax.set(xlabel="k", ylabel="Доля ошибок", title="Подбор k методом LOO")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_selection(history, n, path):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        history[:, 0],
        history[:, 1] / n,
        label="CCV: контроль из одного объекта",
    )

    ax.set(
        xlabel="Число оставшихся эталонов",
        ylabel="Доля ошибок LOO для 1NN",
        title="Жадное удаление неэталонных объектов",
    )

    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_reference(X, y, ids, names, path):
    pairs = list(combinations(range(X.shape[1]), 2))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for ax, (i, j) in zip(axes.ravel(), pairs):
        for label, color in [(-1, "tab:blue"), (1, "tab:orange")]:
            mask = y == label

            ax.scatter(
                X[mask, i],
                X[mask, j],
                s=18,
                alpha=0.4,
                color=color,
                label=f"Класс {label}",
            )

        ax.scatter(
            X[ids, i],
            X[ids, j],
            s=85,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            label="Эталоны",
        )

        ax.set(
            xlabel=f"{names[i]} (стандартизованный)",
            ylabel=f"{names[j]} (стандартизованный)",
            title=f"{names[i]} и {names[j]}",
        )

        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle(
        f"Отбор эталонов: {len(ids)} из {len(X)} объектов",
        fontsize=14,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    (X_train_raw, X_train, X_test, y_train, y_test, names) = load_data()

    img = Path(__file__).resolve().parent.parent / "img"
    img.mkdir(parents=True, exist_ok=True)

    n = len(y_train)

    d_loo = loo_distances(X_train_raw)
    k, ks, errors = choose_k(d_loo, y_train)

    print(f"Выбранное k: {k}")
    print(f"Доля ошибок LOO окна Парзена: {errors.min() / n:.4f}")

    plot_loo(ks, errors, n, img / "k_selection.png")

    d = cdist(X_train, X_train)
    np.fill_diagonal(d, np.inf)

    ids, history = select_reference(
        d,
        y_train,
        min_size=max(3, k + 1),
    )

    y_ref = y_train[ids]

    print(f"\nОбъектов до отбора: {n}")
    print(f"Эталонов после отбора: {len(ids)}")
    print(f"CCV 1NN до отбора: {history[0, 1] / n:.4f}")
    print(f"CCV 1NN после отбора: {history[-1, 1] / n:.4f}")

    plot_selection(history, n, img / "cvv.png")

    plot_reference(
        X_train,
        y_train,
        ids,
        names,
        img / "reference.png",
    )

    d_test = cdist(X_test, X_train)
    d_ref = d_test[:, ids]

    predictions = {
        "Парзен, все объекты": parzen_predict(d_test, y_train, k),
        "Парзен, эталоны": parzen_predict(d_ref, y_ref, k),
        "1NN, все объекты": y_train[np.argmin(d_test, axis=1)],
        "1NN, эталоны": y_ref[np.argmin(d_ref, axis=1)],
    }

    model = KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="euclidean")
    model.fit(X_train, y_train)
    predictions["sklearn KNN"] = model.predict(X_test)

    rows = []

    for name, y_pred in predictions.items():
        rows.append({
            "Алгоритм": name,
            "Ошибок": np.count_nonzero(y_pred != y_test),
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1": f1_score(
                y_test,
                y_pred,
                pos_label=1,
                zero_division=0,
            ),
        })

    results = pd.DataFrame(rows)

    print(f"\nОбъектов в тестовой выборке: {len(y_test)}")
    print(f"k для окна Парзена и sklearn KNN: {k}")
    print("Результаты на тестовой выборке:")
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
