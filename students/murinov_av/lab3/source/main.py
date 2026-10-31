import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .svm import CustomSVM


def plot_and_save_decision_boundary(model, X, y, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=40, label='Объекты выборки')
    
    if hasattr(model, 'support_vectors_') and len(model.support_vectors_) > 0:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], 
                   s=200, facecolors='none', edgecolors='k', linewidths=1.5, label='Опорные векторы')
        
    ax.set_title(title)
    ax.set_xlabel('Признак 1 (X1)')
    ax.set_ylabel('Признак 2 (X2)')
    ax.legend(loc='best')
    
    os.makedirs('reports', exist_ok=True)
    save_path = os.path.join('reports', filename)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

if __name__ == "__main__":
    X, y = make_moons(n_samples=300, noise=0.15, random_state=42)
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Линейные модели
    svm_linear = CustomSVM(C=1.0, kernel='linear')
    svm_linear.fit(X_train, y_train)

    svc_linear = SVC(kernel='linear', C=1.0)
    svc_linear.fit(X_train, y_train)

    # Нелинейные модели (RBF)
    svm_rbf = CustomSVM(C=10.0, kernel='rbf', gamma=2.0)
    svm_rbf.fit(X_train, y_train)

    svc_rbf = SVC(kernel='rbf', C=10.0, gamma=2.0)
    svc_rbf.fit(X_train, y_train)

    plot_and_save_decision_boundary(svm_linear, X_train, y_train, "Кастомный SVM (Линейное ядро)", "svm_linear_custom.png")
    plot_and_save_decision_boundary(svc_linear, X_train, y_train, "Эталонный SVC (Линейное ядро)", "svm_linear_sklearn.png")
    plot_and_save_decision_boundary(svm_rbf, X_train, y_train, "Кастомный SVM (RBF ядро)", "svm_rbf_custom.png")
    plot_and_save_decision_boundary(svc_rbf, X_train, y_train, "Эталонный SVC (RBF ядро)", "svm_rbf_sklearn.png")

    print("\nЛинейное ядро")
    print(f"Кастомный SVM | Accuracy: {np.mean(svm_linear.predict(X_test) == y_test):.4f} | Опорных векторов: {len(svm_linear.support_vectors_)}")
    print(f"Эталонный SVC | Accuracy: {np.mean(svc_linear.predict(X_test) == y_test):.4f} | Опорных векторов: {len(svc_linear.support_vectors_)}\n")

    print("RBF ядро")
    print(f"Кастомный SVM | Accuracy: {np.mean(svm_rbf.predict(X_test) == y_test):.4f} | Опорных векторов: {len(svm_rbf.support_vectors_)}")
    print(f"Эталонный SVC | Accuracy: {np.mean(svc_rbf.predict(X_test) == y_test):.4f} | Опорных векторов: {len(svc_rbf.support_vectors_)}")