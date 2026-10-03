import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


def pairwise_distances(A, B):
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt(np.maximum((diff ** 2).sum(axis=2), 0.0))


class ParzenKNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(self.y_train)
        self.onehot_ = (self.y_train[:, None] == self.classes_[None, :]).astype(float)
        return self

    @staticmethod
    def window_width(dists, k):
        sorted_d = np.sort(dists, axis=1)
        k_eff = min(k, dists.shape[1] - 1)
        h = sorted_d[:, k_eff].copy()
        zero = h <= 0
        if zero.any():
            positive = np.where(sorted_d > 0, sorted_d, np.inf)
            first_nonzero = positive.min(axis=1)
            first_nonzero[~np.isfinite(first_nonzero)] = 1.0
            h[zero] = first_nonzero[zero]
        return h

    def scores_from_dists(self, dists, k):
        h = self.window_width(dists, k)
        weights = np.exp(-0.5 * (dists / h[:, None]) ** 2)
        return weights @ self.onehot_

    def predict_from_dists(self, dists, k=None):
        k = self.k if k is None else k
        scores = self.scores_from_dists(dists, k)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict(self, X, k=None):
        dists = pairwise_distances(np.asarray(X, dtype=float), self.X_train)
        return self.predict_from_dists(dists, k)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class NearestNeighborVote:
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(self.y_train)
        return self

    def predict(self, X):
        dists = pairwise_distances(np.asarray(X, dtype=float), self.X_train)
        k = min(self.k, len(self.y_train))
        nearest = np.argsort(dists, axis=1)[:, :k]
        votes = np.zeros((len(dists), len(self.classes_)))
        for ci, c in enumerate(self.classes_):
            votes[:, ci] = (self.y_train[nearest] == c).sum(axis=1)
        return self.classes_[np.argmax(votes, axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


def loo_risk_curve(X, y, k_values):
    D = pairwise_distances(X, X)
    np.fill_diagonal(D, np.inf)
    model = ParzenKNN().fit(X, y)
    risks = []
    for k in k_values:
        preds = model.predict_from_dists(D, k)
        risks.append(np.mean(preds != y))
    return np.array(risks)


def greedy_prototype_selection(X, y):
    n = len(y)
    D = pairwise_distances(X, X)
    np.fill_diagonal(D, np.inf)

    omega = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        centroid = X[idx].mean(axis=0)
        omega.append(int(idx[np.argmin(np.linalg.norm(X[idx] - centroid, axis=1))]))

    best_dist = np.full(n, np.inf)
    best_label = np.full(n, -1)
    for o in omega:
        closer = D[:, o] < best_dist
        best_dist = np.where(closer, D[:, o], best_dist)
        best_label = np.where(closer, y[o], best_label)
    errors = int(np.sum(best_label != y))
    history = [errors]

    while True:
        candidates = np.setdiff1d(np.arange(n), omega)
        if len(candidates) == 0:
            break
        cand_errors = np.empty(len(candidates), dtype=int)
        for j, c in enumerate(candidates):
            closer = D[:, c] < best_dist
            new_label = np.where(closer, y[c], best_label)
            cand_errors[j] = np.sum(new_label != y)
        j_best = int(np.argmin(cand_errors))
        if cand_errors[j_best] >= errors:
            break
        c = int(candidates[j_best])
        closer = D[:, c] < best_dist
        best_dist = np.where(closer, D[:, c], best_dist)
        best_label = np.where(closer, y[c], best_label)
        errors = int(cand_errors[j_best])
        omega.append(c)
        history.append(errors)

    return np.array(omega), np.array(history)


def load_data():
    data = load_wine()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


def metrics(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')


def visualize_loo_errors(k_values, risks, best_k, filename='loo_errors.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, risks, marker='o', linewidth=2, markersize=6)
    plt.axvline(best_k, color='red', linestyle='--', label=f'k* = {best_k}')
    plt.xlabel('Количество соседей (k)', fontsize=12)
    plt.ylabel('Эмпирический риск LOO', fontsize=12)
    plt.title('Эмпирический риск LOO для различных k', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def visualize_prototypes(X, y, omega, history, filename='prototypes_visualization.png'):
    pca = PCA(n_components=2).fit(X)
    Z = pca.transform(X)
    rest = np.setdiff1d(np.arange(len(y)), omega)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.scatter(Z[rest, 0], Z[rest, 1], c=y[rest], cmap='viridis', alpha=0.35, s=30)
    ax.scatter(Z[omega, 0], Z[omega, 1], c=y[omega], cmap='viridis', s=140,
               edgecolors='black', linewidths=2)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)', fontsize=12)
    ax.set_title(f'Эталоны ({len(omega)} из {len(y)}) выделены чёрной обводкой', fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(np.arange(1, len(history) + 1), history, marker='o', markersize=5, color='tab:green')
    ax.set_xlabel('Число эталонов', fontsize=12)
    ax.set_ylabel('Ошибок 1NN(Ω) на обучающей выборке', fontsize=12)
    ax.set_title('Сходимость критерия при добавлении эталонов', fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def main():
    print("=" * 60)
    print("Лабораторная работа №2: Метрическая классификация (KNN)")
    print("=" * 60)

    print("\n1. Загрузка датасета Wine...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"   Train: {X_train.shape[0]} объектов, {X_train.shape[1]} признаков")
    print(f"   Test: {X_test.shape[0]} объектов")
    print(f"   Баланс классов (train): {np.bincount(y_train).tolist()}")

    print("\n2. Подбор k методом LOO (окно Парзена переменной ширины)...")
    k_values = np.arange(1, 41)
    risks = loo_risk_curve(X_train, y_train, k_values)
    best_k = int(k_values[np.argmin(risks)])
    print(f"   Оптимальное k: {best_k} (ошибка LOO: {risks.min():.4f})")
    for k in (1, 3, 5, 10, 20, 40):
        print(f"   k={k:>2}: LOO-риск = {risks[k - 1]:.4f}")
    visualize_loo_errors(k_values, risks, best_k)

    print("\n3. Сравнение со sklearn...")
    own = ParzenKNN(k=best_k).fit(X_train, y_train)
    rows = [("Парзен (своя реализация)", own.predict(X_test))]
    for weights in ("uniform", "distance"):
        skl = KNeighborsClassifier(n_neighbors=best_k, weights=weights).fit(X_train, y_train)
        rows.append((f"sklearn KNN ({weights})", skl.predict(X_test)))
    for name, pred in rows:
        acc, f1 = metrics(y_test, pred)
        print(f"   {name}: accuracy {acc:.4f}, macro-F1 {f1:.4f}")

    print("\n   Сравнение по диапазону k (accuracy на тесте):")
    print(f"   {'k':>3} | {'Парзен':>8} | {'sklearn uniform':>15} | {'sklearn distance':>16}")
    for k in (1, 3, 5, 9, 15, 25, 40):
        a_own = accuracy_score(y_test, ParzenKNN(k=k).fit(X_train, y_train).predict(X_test))
        a_u = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).score(X_test, y_test)
        a_d = KNeighborsClassifier(n_neighbors=k, weights="distance").fit(X_train, y_train).score(X_test, y_test)
        print(f"   {k:>3} | {a_own:>8.4f} | {a_u:>15.4f} | {a_d:>16.4f}")

    print("\n4. Отбор эталонов (жадный, критерий: ошибки 1NN(Ω), эквивалент CCV при k=1)...")
    omega, history = greedy_prototype_selection(X_train, y_train)
    print(f"   Исходных объектов: {len(X_train)}")
    print(f"   Отобрано эталонов: {len(omega)}")
    print(f"   Сжатие выборки: {(1 - len(omega) / len(X_train)) * 100:.1f}%")
    print(f"   Ошибок 1NN(Ω): {history[0]} -> {history[-1]}")
    print(f"   Баланс классов в эталонах: {np.bincount(y_train[omega]).tolist()}")
    visualize_prototypes(X_train, y_train, omega, history)

    print("\n5. Сравнение KNN с отбором эталонов и без (1NN)...")
    full = NearestNeighborVote(k=1).fit(X_train, y_train)
    proto = NearestNeighborVote(k=1).fit(X_train[omega], y_train[omega])
    for name, model in (("1NN, полная выборка", full), (f"1NN, {len(omega)} эталонов", proto)):
        acc, f1 = metrics(y_test, model.predict(X_test))
        print(f"   {name}: accuracy {acc:.4f}, macro-F1 {f1:.4f}")

    k_proto = min(best_k, len(omega) - 1)
    parzen_proto = ParzenKNN(k=k_proto).fit(X_train[omega], y_train[omega])
    acc, f1 = metrics(y_test, parzen_proto.predict(X_test))
    print(f"   Парзен (k={k_proto}) на эталонах: accuracy {acc:.4f}, macro-F1 {f1:.4f}")

    print("\n" + "=" * 60)
    print("Все графики сохранены в текущей директории.")
    print("=" * 60)


if __name__ == "__main__":
    main()
