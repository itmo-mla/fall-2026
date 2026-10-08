import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from .models import ParzenKNN, prototype_selection_greedy_removal

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
reports_dir = os.path.join(base_dir, "reports")
os.makedirs(reports_dir, exist_ok=True)

def plot_decision_boundary(X_train, y_train, predict_func, title, filename):
    h = 0.02
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    if isinstance(predict_func, KNeighborsClassifier):
        predict_func.fit(X_train, y_train)
        Z = predict_func.predict(grid_points)
    else:
        Z = predict_func(X_train, y_train, grid_points)
        
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', s=50)
    plt.title(title)
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.savefig(os.path.join(reports_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    np.random.seed(42)
    
    X, y = make_moons(n_samples=300, noise=0.25, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nПодбор параметра k (LOO)")
    k_values = range(1, 31)
    loo_errors = []
    
    for k in k_values:
        model = ParzenKNN(k=k)
        loo = model.calculate_loo(X_train, y_train)
        loo_errors.append(loo)
        print(f"k = {k:2d} | LOO error = {loo:.4f}")
        
    optimal_k = k_values[np.argmin(loo_errors)]
    print(f"\nОптимальное k: {optimal_k} с ошибкой LOO: {min(loo_errors):.4f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, loo_errors, marker='o', linestyle='-', color='b', label='LOO Error')
    plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
    plt.title("Зависимость эмпирического риска (LOO) от числа соседей k")
    plt.xlabel("Число соседей (k)")
    plt.ylabel("Доля ошибок (LOO)")
    plt.xticks(k_values)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(reports_dir, "loo_error_vs_k.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    sklearn_knn = KNeighborsClassifier(n_neighbors=optimal_k, weights='distance')
    
    def parzen_predict(X_tr, y_tr, X_te):
        m = ParzenKNN(k=optimal_k)
        m.fit(X_tr, y_tr)
        return m.predict(X_te)
        
    plot_decision_boundary(X_train, y_train, parzen_predict, f"Parzen KNN (k={optimal_k})", "parzen_knn_boundary.png")
    plot_decision_boundary(X_train, y_train, sklearn_knn, f"sklearn KNN (k={optimal_k})", "sklearn_knn_boundary.png")
    
    print("\nАлгоритм отбора эталонов")
    k_control = int(0.2 * len(X_train)) 
    prototypes_X, prototypes_y = prototype_selection_greedy_removal(X_train, y_train, k_control=k_control, tolerance=1e-6)
    print(f"Итоговый размер множества эталонов: {len(prototypes_X)}/{len(X_train)}")
    
    is_prototype = np.zeros(len(X_train), dtype=bool)
    for i in range(len(X_train)):
        if np.any(np.linalg.norm(prototypes_X - X_train[i], axis=1) < 1e-8):
            is_prototype[i] = True
            
    plt.figure(figsize=(10, 8))
    plt.scatter(X_train[~is_prototype, 0], X_train[~is_prototype, 1], c='lightgray', label='Удаленные объекты (шум)', s=40, alpha=0.6)
    plt.scatter(prototypes_X[:, 0], prototypes_X[:, 1], c=prototypes_y, cmap='coolwarm', 
                edgecolors='black', s=120, linewidth=1.5, label='Отобранные эталоны')
    plt.title("Результат работы алгоритма отбора эталонов")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(reports_dir, "prototype_selection.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    model_full = ParzenKNN(k=optimal_k)
    model_full.fit(X_train, y_train)
    preds_full = model_full.predict(X_test)
    acc_full = accuracy_score(y_test, preds_full)
    f1_full = f1_score(y_test, preds_full, average='macro')
    
    model_proto = ParzenKNN(k=optimal_k)
    model_proto.fit(prototypes_X, prototypes_y)
    preds_proto = model_proto.predict(X_test)
    acc_proto = accuracy_score(y_test, preds_proto)
    f1_proto = f1_score(y_test, preds_proto, average='macro')
    
    print(f"[Без отбора] Размер базы: {len(X_train)} | Accuracy: {acc_full:.4f} | F1-score: {f1_full:.4f}")
    print(f"[С отбором]  Размер базы: {len(prototypes_X)} | Accuracy: {acc_proto:.4f} | F1-score: {f1_proto:.4f}")
    
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    Z_full = model_full.predict(grid_points).reshape(xx.shape)
    Z_proto = model_proto.predict(grid_points).reshape(xx.shape)
    
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    

    ax1.contourf(xx, yy, Z_full, alpha=0.6, cmap='coolwarm')
    ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', s=20, alpha=0.4, label='Train')
    ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', edgecolors='black', s=80, marker='^', label='Test')
    ax1.set_title(f"БЕЗ отбора эталонов\n(База: {len(X_train)} | Acc={acc_full:.2f})")
    ax1.legend(loc='upper right')
    

    ax2.contourf(xx, yy, Z_proto, alpha=0.6, cmap='coolwarm')
    ax2.scatter(prototypes_X[:, 0], prototypes_X[:, 1], c=prototypes_y, cmap='coolwarm', edgecolors='k', s=80, alpha=1.0, label='Prototypes')
    ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', edgecolors='black', s=80, marker='^', label='Test')
    ax2.set_title(f"С отбором эталонов\n(База: {len(prototypes_X)} | Acc={acc_proto:.2f})")
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "comparison_prototypes_vs_full.png"), dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()