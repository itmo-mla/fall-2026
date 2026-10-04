"""Шаг 5: визуализация разделяющей границы в пространстве двух главных компонент (PCA)."""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from classifier import DualSVM

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


def plot_boundary(ax, X, y, predict_fn, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = predict_fn(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=[-1.5, 0, 1.5], colors=['#ffd7d7', '#d7e8ff'], alpha=0.8)
    ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=1.5)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='#1f77b4', s=15, label='отказ (+1)', edgecolors='k', linewidths=0.3)
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c='#d62728', s=15, label='норма (-1)', edgecolors='k', linewidths=0.3)
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)


def visualize(X_train, y_train, C=1.0, gamma_rbf=0.5, save_path=None):
    """Обучает обе реализации заново в 2D (PCA) и рисует границы 2x2."""
    pca = PCA(n_components=2, random_state=42)
    X_train_2d = pca.fit_transform(X_train)
    print(f"Объяснённая дисперсия: {pca.explained_variance_ratio_.sum():.3f}")

    own_lin = DualSVM(C=C, kernel='linear').fit(X_train_2d, y_train)
    sk_lin = SVC(C=C, kernel='linear').fit(X_train_2d, y_train)
    own_rbf = DualSVM(C=C, kernel='rbf', gamma=gamma_rbf).fit(X_train_2d, y_train)
    sk_rbf = SVC(C=C, kernel='rbf', gamma=gamma_rbf).fit(X_train_2d, y_train)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plot_boundary(axes[0, 0], X_train_2d, y_train, own_lin.predict, 'Своя реализация — линейное ядро')
    plot_boundary(axes[0, 1], X_train_2d, y_train, sk_lin.predict, 'sklearn — линейное ядро')
    plot_boundary(axes[1, 0], X_train_2d, y_train, own_rbf.predict, 'Своя реализация — RBF-ядро')
    plot_boundary(axes[1, 1], X_train_2d, y_train, sk_rbf.predict, 'sklearn — RBF-ядро')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120)

    plt.show()
