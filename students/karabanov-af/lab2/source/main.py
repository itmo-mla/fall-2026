import os
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

import knn
import plots
import prototypes
from data import load_data, standardize, train_test_split
from metrics import accuracy, class_metrics, confusion_matrix

IMAGES = os.path.join(os.path.dirname(__file__), "..", "images")
KS = np.arange(1, 41)


def main():
    X, y, _, class_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"объектов: {len(y)}, признаков: {X.shape[1]}, классов: {len(np.unique(y))}")
    print(f"обучение: {X_train.shape}, по классам {np.bincount(y_train)}")
    print(f"тест: {X_test.shape}, по классам {np.bincount(y_test)}")

    Z_train = standardize(X_train)
    Z_test = standardize(X_test, X_train)

    standardization_effect(X_train, y_train, X_test, y_test, Z_train, Z_test)

    risks = {
        "голосование": knn.loo_risk(Z_train, y_train, KS, knn.predict),
        "окно Парзена": knn.loo_risk(Z_train, y_train, KS, knn.predict_parzen),
    }
    plots.plot_loo(KS, risks, "Эмпирический риск LOO в зависимости от k", os.path.join(IMAGES, "loo.png"))

    print("\nподбор k по LOO на обучающей выборке:")
    print(f"{'k':>3} {'голосование':>13} {'окно Парзена':>14}")
    for i, k in enumerate(KS):
        if k <= 12 or k % 5 == 0:
            print(f"{k:>3} {risks['голосование'][i]:>13.3f} {risks['окно Парзена'][i]:>14.3f}")

    D_test = knn.distances(Z_train, Z_test)
    print("\nлучшее k по LOO и качество на тесте:")
    for name, method in [("голосование", knn.predict), ("окно Парзена", knn.predict_parzen)]:
        k = KS[risks[name].argmin()]
        print(f"  {name:13s} k = {k:2d}, LOO = {risks[name].min():.3f}, "
              f"accuracy на тесте = {accuracy(y_test, method(D_test, y_train, k)):.3f}")

    best_k = KS[risks["окно Парзена"].argmin()]
    compare_with_reference(Z_train, y_train, Z_test, y_test, KS[risks["голосование"].argmin()])
    idx = prototype_experiment(Z_train, y_train, Z_test, y_test, class_names, best_k)
    quality_report(Z_train, y_train, Z_test, y_test, idx, class_names, best_k)
    prototype_comparison(X, y, best_k)
    k_stability(X, y)


def quality_report(Z_train, y_train, Z_test, y_test, idx, class_names, k):
    """Accuracy alone hides which class the errors belong to, so here is the full picture per class."""
    print(f"\nкачество по классам на тесте, окно Парзена, k = {k}:")
    matrices = {}
    for name, Z, labels in [("вся выборка", Z_train, y_train), ("только эталоны", Z_train[idx], y_train[idx])]:
        y_pred = knn.predict_parzen(knn.distances(Z, Z_test), labels, k)
        cm = confusion_matrix(y_test, y_pred, len(class_names))
        matrices[name] = cm
        precision, recall, f1 = class_metrics(cm)
        print(f"  {name}: accuracy = {accuracy(y_test, y_pred):.3f}")
        print(f"    {'класс':10s} {'precision':>10} {'recall':>8} {'f1':>7} {'объектов':>9}")
        for c, class_name in enumerate(class_names):
            print(f"    {class_name:10s} {precision[c]:>10.3f} {recall[c]:>8.3f} {f1[c]:>7.3f} {cm[c].sum():>9d}")
    plots.plot_confusions(matrices, class_names, f"Матрицы ошибок на тесте, k = {k}",
                          os.path.join(IMAGES, "confusion.png"))


def benchmark(call, n_repeats=20):
    """Average time of one call in milliseconds, a single run is too noisy to compare."""
    call()
    start = time.perf_counter()
    for _ in range(n_repeats):
        call()
    return (time.perf_counter() - start) / n_repeats * 1000


def standardization_effect(X_train, y_train, X_test, y_test, Z_train, Z_test):
    """The features are measured in different units, so without scaling the distance is defined by proline alone."""
    print("\naccuracy на тесте, голосование k соседей:")
    print(f"{'k':>3} {'сырые признаки':>16} {'стандартизованные':>19}")
    D_raw = knn.distances(X_train, X_test)
    D_scaled = knn.distances(Z_train, Z_test)
    for k in [1, 3, 5, 10, 20]:
        raw = accuracy(y_test, knn.predict(D_raw, y_train, k))
        scaled = accuracy(y_test, knn.predict(D_scaled, y_train, k))
        print(f"{k:>3} {raw:>16.3f} {scaled:>19.3f}")


def k_stability(X, y, n_splits=10):
    """The LOO curve is noisy, so its argmin on a single split is not to be trusted."""
    print(f"\nустойчивость выбора k, {n_splits} случайных разбиений:")
    for name, method in [("голосование", knn.predict), ("окно Парзена", knn.predict_parzen)]:
        risks, accuracies = [], []
        for seed in range(n_splits):
            X_train, X_test, y_train, y_test = train_test_split(X, y, seed=seed)
            Z_train, Z_test = standardize(X_train), standardize(X_test, X_train)
            D_test = knn.distances(Z_train, Z_test)
            risks.append(knn.loo_risk(Z_train, y_train, KS, method))
            accuracies.append([accuracy(y_test, method(D_test, y_train, k)) for k in KS])
        risks, accuracies = np.array(risks), np.array(accuracies)
        print(f"  {name:13s} лучшее k по разбиениям: {[int(KS[r.argmin()]) for r in risks]}")
        print(f"  {' ':13s} минимум усреднённой LOO при k = {KS[risks.mean(axis=0).argmin()]}, "
              f"максимум accuracy на тесте при k = {KS[accuracies.mean(axis=0).argmax()]}")


def prototype_experiment(Z_train, y_train, Z_test, y_test, class_names, k):
    idx, margins = prototypes.select(Z_train, y_train, k)
    noise = np.flatnonzero(margins <= 0)
    print(f"\nотбор эталонов при k = {k}:")
    print(f"  отсеяно как шум: {len(noise)}, эталонов: {len(idx)} из {len(y_train)} "
          f"({100 * len(idx) / len(y_train):.0f}%), по классам {np.bincount(y_train[idx])}")

    projection = PCA(n_components=2).fit_transform(Z_train)
    plots.plot_prototypes(projection, y_train, idx, noise, class_names,
                          f"Эталоны в проекции на две главные компоненты, k = {k}",
                          os.path.join(IMAGES, "prototypes.png"))
    plots.plot_margins(margins, f"Отступы обучающих объектов, k = {k}", os.path.join(IMAGES, "margins.png"))
    return idx


def prototype_comparison(X, y, k, n_splits=10):
    """Same experiment on several random splits: one test of 53 objects is too small to compare on."""
    print(f"\nKNN с отбором эталонов и без него, k = {k}, {n_splits} случайных разбиений:")
    stats = {name: {"accuracy": [], "size": [], "time": []} for name in ["вся выборка", "только эталоны"]}
    for seed in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(X, y, seed=seed)
        Z_train, Z_test = standardize(X_train), standardize(X_test, X_train)
        idx, _ = prototypes.select(Z_train, y_train, k)
        for name, Z, labels in [("вся выборка", Z_train, y_train), ("только эталоны", Z_train[idx], y_train[idx])]:
            start = time.perf_counter()
            y_pred = knn.predict_parzen(knn.distances(Z, Z_test), labels, k)
            stats[name]["time"].append(time.perf_counter() - start)
            stats[name]["accuracy"].append(accuracy(y_test, y_pred))
            stats[name]["size"].append(len(labels))

    for name, s in stats.items():
        print(f"  {name:15s} accuracy = {np.mean(s['accuracy']):.3f} ± {np.std(s['accuracy']):.3f}, "
              f"объектов в памяти {np.mean(s['size']):.0f}, "
              f"время предсказания {np.mean(s['time']) * 1000:.2f} мс")
    plots.plot_comparison({name: s["accuracy"] for name, s in stats.items()},
                          "Accuracy на тесте по 10 случайным разбиениям",
                          os.path.join(IMAGES, "prototypes_quality.png"))


def compare_with_reference(Z_train, y_train, Z_test, y_test, k):
    print(f"\nсравнение с эталоном sklearn, k = {k}:")
    D_test = knn.distances(Z_train, Z_test)
    own = {"своя реализация, голосование": knn.predict(D_test, y_train, k),
           "своя реализация, Парзен": knn.predict_parzen(D_test, y_train, k)}
    reference = {}
    for weights in ["uniform", "distance"]:
        model = KNeighborsClassifier(n_neighbors=k, weights=weights).fit(Z_train, y_train)
        reference[f"KNeighborsClassifier, {weights}"] = model.predict(Z_test)

    for name, y_pred in {**own, **reference}.items():
        print(f"  {name:32s} accuracy = {accuracy(y_test, y_pred):.3f}")
    votes, uniform = own["своя реализация, голосование"], reference["KNeighborsClassifier, uniform"]
    print(f"  расхождений с эталоном (голосование против uniform): {int((votes != uniform).sum())} из {len(y_test)}")

    own_time = benchmark(lambda: knn.predict(knn.distances(Z_train, Z_test), y_train, k))
    reference_time = benchmark(lambda: KNeighborsClassifier(n_neighbors=k).fit(Z_train, y_train).predict(Z_test))
    print(f"  время предсказания для {len(y_test)} объектов: своя {own_time:.2f} мс, sklearn {reference_time:.2f} мс")


if __name__ == "__main__":
    main()
