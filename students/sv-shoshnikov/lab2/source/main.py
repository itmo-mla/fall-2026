import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from collections import Counter


class KNNClassifier:
    def __init__(self, k=5, kernel='gaussian', variable_width=True):
        self.k = k
        self.kernel = kernel
        self.variable_width = variable_width
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def gaussian_kernel(self, distance, h):
        return np.exp(-(distance ** 2) / (2 * h ** 2))
    
    def predict_one(self, x):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_distances = distances[nearest_indices]
        nearest_labels = self.y_train[nearest_indices]
        
        if self.variable_width:
            h = nearest_distances[-1] if nearest_distances[-1] > 0 else 1.0
        else:
            h = 1.0
        
        weights = self.gaussian_kernel(nearest_distances, h)
        weights = weights / (weights.sum() + 1e-10)
        
        weighted_votes = {}
        for label, weight in zip(nearest_labels, weights):
            weighted_votes[label] = weighted_votes.get(label, 0) + weight
        
        return max(weighted_votes, key=weighted_votes.get)
    
    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


class PrototypeSelector:
    def __init__(self, base_classifier):
        self.base_classifier = base_classifier
        self.prototypes_X = None
        self.prototypes_y = None
    
    def fit(self, X, y):
        n_samples = len(X)
        prototypes_mask = np.ones(n_samples, dtype=bool)
        
        for i in range(n_samples):
            temp_mask = prototypes_mask.copy()
            temp_mask[i] = False
            
            X_temp = X[temp_mask]
            y_temp = y[temp_mask]
            
            self.base_classifier.fit(X_temp, y_temp)
            pred = self.base_classifier.predict_one(X[i])
            
            if pred != y[i]:
                prototypes_mask[i] = True
            else:
                prototypes_mask[i] = False
        
        self.prototypes_X = X[prototypes_mask]
        self.prototypes_y = y[prototypes_mask]
        
        return prototypes_mask
    
    def get_prototypes(self):
        return self.prototypes_X, self.prototypes_y


def load_data():
    data = load_wine()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, data.feature_names


def loo_cross_validation(X, y, k_values):
    n_samples = len(X)
    errors = {k: 0 for k in k_values}
    
    for i in range(n_samples):
        X_train_loo = np.delete(X, i, axis=0)
        y_train_loo = np.delete(y, i)
        x_test_loo = X[i:i+1]
        y_test_loo = y[i]
        
        for k in k_values:
            clf = KNNClassifier(k=k)
            clf.fit(X_train_loo, y_train_loo)
            pred = clf.predict(x_test_loo)[0]
            if pred != y_test_loo:
                errors[k] += 1
    
    loo_errors = {k: errors[k] / n_samples for k in k_values}
    return loo_errors


def visualize_loo_errors(loo_errors):
    k_values = sorted(loo_errors.keys())
    error_values = [loo_errors[k] for k in k_values]
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, error_values, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Количество соседей (k)', fontsize=12)
    plt.ylabel('Ошибка LOO', fontsize=12)
    plt.title('Эмпирический риск для различных k', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loo_errors.png', dpi=300)
    plt.close()
    
    best_k = min(loo_errors, key=loo_errors.get)
    print(f"\nОптимальное k: {best_k} (ошибка LOO: {loo_errors[best_k]:.4f})")
    return best_k


def visualize_prototypes(X_train, y_train, prototypes_mask):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', alpha=0.6, s=50)
    plt.xlabel('Признак 1', fontsize=12)
    plt.ylabel('Признак 2', fontsize=12)
    plt.title(f'Все объекты ({len(X_train)})', fontsize=14)
    plt.colorbar(label='Класс')
    
    plt.subplot(1, 2, 2)
    X_proto = X_train[prototypes_mask]
    y_proto = y_train[prototypes_mask]
    plt.scatter(X_proto[:, 0], X_proto[:, 1], c=y_proto, cmap='viridis', alpha=0.8, s=100, edgecolors='black', linewidths=2)
    plt.xlabel('Признак 1', fontsize=12)
    plt.ylabel('Признак 2', fontsize=12)
    plt.title(f'Эталоны ({len(X_proto)})', fontsize=14)
    plt.colorbar(label='Класс')
    
    plt.tight_layout()
    plt.savefig('prototypes_visualization.png', dpi=300)
    plt.close()


def main():
    print("="*60)
    print("Лабораторная работа №2: Метрическая классификация (KNN)")
    print("="*60)
    
    print("\n1. Загрузка датасета Wine...")
    X_train, X_test, y_train, y_test, feature_names = load_data()
    print(f"   Train: {X_train.shape[0]} объектов, {X_train.shape[1]} признаков")
    print(f"   Test: {X_test.shape[0]} объектов")
    print(f"   Классы: {len(np.unique(y_train))}")
    
    print("\n2. Подбор оптимального k методом LOO...")
    k_values = list(range(1, 21))
    loo_errors = loo_cross_validation(X_train, y_train, k_values)
    best_k = visualize_loo_errors(loo_errors)
    
    print("\n3. Обучение KNN с окном Парзена (переменная ширина)...")
    knn_parzen = KNNClassifier(k=best_k, variable_width=True)
    knn_parzen.fit(X_train, y_train)
    train_acc_parzen = knn_parzen.score(X_train, y_train)
    test_acc_parzen = knn_parzen.score(X_test, y_test)
    print(f"   Train accuracy: {train_acc_parzen:.4f}")
    print(f"   Test accuracy: {test_acc_parzen:.4f}")
    
    print("\n4. Сравнение с эталонной реализацией (sklearn)...")
    knn_sklearn = KNeighborsClassifier(n_neighbors=best_k)
    knn_sklearn.fit(X_train, y_train)
    train_acc_sklearn = knn_sklearn.score(X_train, y_train)
    test_acc_sklearn = knn_sklearn.score(X_test, y_test)
    print(f"   Train accuracy: {train_acc_sklearn:.4f}")
    print(f"   Test accuracy: {test_acc_sklearn:.4f}")
    
    print("\n5. Отбор эталонов...")
    selector = PrototypeSelector(KNNClassifier(k=best_k))
    prototypes_mask = selector.fit(X_train, y_train)
    X_proto, y_proto = selector.get_prototypes()
    print(f"   Исходных объектов: {len(X_train)}")
    print(f"   Отобрано эталонов: {len(X_proto)}")
    print(f"   Сжатие: {len(X_proto)/len(X_train)*100:.1f}%")
    visualize_prototypes(X_train, y_train, prototypes_mask)
    
    print("\n6. KNN с отбором эталонов...")
    knn_proto = KNNClassifier(k=min(best_k, len(X_proto)))
    knn_proto.fit(X_proto, y_proto)
    train_acc_proto = knn_proto.score(X_train, y_train)
    test_acc_proto = knn_proto.score(X_test, y_test)
    print(f"   Train accuracy: {train_acc_proto:.4f}")
    print(f"   Test accuracy: {test_acc_proto:.4f}")
    
    print("\n" + "="*60)
    print("ИТОГИ:")
    print("="*60)
    print(f"KNN с окном Парзена:      Test = {test_acc_parzen:.4f}")
    print(f"Эталон (sklearn):         Test = {test_acc_sklearn:.4f}")
    print(f"KNN с отбором эталонов:   Test = {test_acc_proto:.4f}")
    print(f"\nОптимальное k: {best_k}")
    print(f"Сжатие выборки: {len(X_proto)}/{len(X_train)} ({len(X_proto)/len(X_train)*100:.1f}%)")
    print("\nВсе графики сохранены в текущей директории.")
    print("="*60)


if __name__ == "__main__":
    main()
