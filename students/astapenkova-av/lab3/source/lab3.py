from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC


def linear_kernel(A, B):
    return A @ B.T


def rbf_kernel(A, B, gamma):
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


class DualSVM:
    def __init__(self, C=1.0, kernel='linear', gamma=None, tol=1e-6):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol

    def _k(self, A, B):
        if self.kernel == 'linear':
            return linear_kernel(A, B)
        return rbf_kernel(A, B, self.gamma)

    def fit(self, X, y):
        n = X.shape[0]
        if self.kernel == 'rbf' and self.gamma is None:
            self.gamma = 1.0 / X.shape[1]
        K = self._k(X, X)
        Q = (y[:, None] * y[None, :]) * K

        res = minimize(
            lambda l: 0.5 * l @ Q @ l - l.sum(),
            x0=np.zeros(n),
            jac=lambda l: Q @ l - np.ones(n),
            bounds=[(0.0, self.C)] * n,
            constraints=[{'type': 'eq', 'fun': lambda l: l @ y, 'jac': lambda l: y}],
            method='SLSQP',
            options={'maxiter': 500, 'ftol': 1e-9},
        )

        lmbda = res.x
        sv = lmbda > self.tol
        margin = sv & (lmbda < self.C - self.tol)
        idx = margin if margin.any() else sv

        self.lmbda = lmbda[sv]
        self.sv_X = X[sv]
        self.sv_y = y[sv]
        self.b = np.mean(y[idx] - (lmbda * y) @ K[:, idx])
        self.n_sv = int(sv.sum())
        if self.kernel == 'linear':
            self.w = (self.lmbda * self.sv_y) @ self.sv_X
        return self

    def decision_function(self, X):
        return self._k(X, self.sv_X) @ (self.lmbda * self.sv_y) + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1.0, -1.0)


def load_data():
    d = load_breast_cancer()
    X = d.data.astype(float)
    y = np.where(d.target == 0, 1.0, -1.0)
    return X, y


def train_test_split(X, y, test_size=0.3, seed=42):
    rng = np.random.default_rng(seed)
    test_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        test_idx.extend(idx[:int(round(len(idx) * test_size))])
    mask = np.zeros(len(y), dtype=bool)
    mask[np.array(test_idx)] = True
    return X[~mask], X[mask], y[~mask], y[mask]


class Standardizer:
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean) / self.std


class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        self.mean = X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X - self.mean, full_matrices=False)
        self.components = Vt[:self.n_components]
        return self

    def transform(self, X):
        return (X - self.mean) @ self.components.T


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def f1(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == -1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == -1))
    return float(2 * tp / (2 * tp + fp + fn))


def main():
    C = 1.0
    img_dir = Path(__file__).resolve().parent.parent / 'images'
    img_dir.mkdir(exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    scaler = Standardizer().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    gamma = 1.0 / X_train_s.shape[1]

    own_lin = DualSVM(C=C, kernel='linear').fit(X_train_s, y_train)
    own_rbf = DualSVM(C=C, kernel='rbf', gamma=gamma).fit(X_train_s, y_train)
    ref_lin = SVC(C=C, kernel='linear').fit(X_train_s, y_train)
    ref_rbf = SVC(C=C, kernel='rbf', gamma=gamma).fit(X_train_s, y_train)

    models = {
        'own linear': own_lin,
        'sklearn linear': ref_lin,
        'own rbf': own_rbf,
        'sklearn rbf': ref_rbf,
    }

    rows = []
    for name, m in models.items():
        pred = m.predict(X_test_s)
        n_sv = m.n_sv if isinstance(m, DualSVM) else int(m.n_support_.sum())
        rows.append({
            'model': name,
            'accuracy': accuracy(y_test, pred),
            'f1': f1(y_test, pred),
            'n_support': n_sv,
        })
    print(pd.DataFrame(rows).to_string(index=False))

    w_ref = ref_lin.coef_.ravel()
    b_ref = ref_lin.intercept_[0]
    print('||w_own - w_sklearn|| =', np.linalg.norm(own_lin.w - w_ref))
    print('|b_own - b_sklearn|   =', abs(own_lin.b - b_ref))
    print('cos(w_own, w_sklearn) =', own_lin.w @ w_ref / (np.linalg.norm(own_lin.w) * np.linalg.norm(w_ref)))
    print('prediction agreement linear:', np.mean(own_lin.predict(X_test_s) == ref_lin.predict(X_test_s)))
    print('prediction agreement rbf:   ', np.mean(own_rbf.predict(X_test_s) == ref_rbf.predict(X_test_s)))

    pca = PCA(n_components=2).fit(X_train_s)
    Z_train = pca.transform(X_train_s)
    Z_test = pca.transform(X_test_s)
    gamma_2d = 0.5

    panels = [
        ('Собственный SVM, линейное ядро', DualSVM(C=C, kernel='linear').fit(Z_train, y_train)),
        ('sklearn SVC, линейное ядро', SVC(C=C, kernel='linear').fit(Z_train, y_train)),
        ('Собственный SVM, RBF-ядро', DualSVM(C=C, kernel='rbf', gamma=gamma_2d).fit(Z_train, y_train)),
        ('sklearn SVC, RBF-ядро', SVC(C=C, kernel='rbf', gamma=gamma_2d).fit(Z_train, y_train)),
    ]

    pad = 1.0
    xx, yy = np.meshgrid(
        np.linspace(Z_train[:, 0].min() - pad, Z_train[:, 0].max() + pad, 300),
        np.linspace(Z_train[:, 1].min() - pad, Z_train[:, 1].max() + pad, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    legend_items = [
        Line2D([], [], marker='o', color='w', markerfacecolor='tab:red', markersize=7, label='Злокачественная (M, +1)'),
        Line2D([], [], marker='o', color='w', markerfacecolor='tab:blue', markersize=7, label='Доброкачественная (B, -1)'),
        Line2D([], [], marker='o', color='w', markerfacecolor='none', markeredgecolor='k', markersize=10, label='Опорные векторы'),
        Line2D([], [], color='k', linestyle='-', label='Граница решения f(x)=0'),
        Line2D([], [], color='k', linestyle='--', label='Границы отступа f(x)=±1'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (title, m) in zip(axes.ravel(), panels):
        f = m.decision_function(grid).reshape(xx.shape)
        ax.contourf(xx, yy, f, levels=[f.min(), 0, f.max()], colors=['#cfe3ff', '#ffd6d6'], alpha=0.6)
        ax.contour(xx, yy, f, levels=[-1, 0, 1], colors='k', linestyles=['--', '-', '--'], linewidths=1.2)
        ax.scatter(Z_train[y_train == 1, 0], Z_train[y_train == 1, 1], c='tab:red', s=18)
        ax.scatter(Z_train[y_train == -1, 0], Z_train[y_train == -1, 1], c='tab:blue', s=18)
        sv_pts = m.sv_X if isinstance(m, DualSVM) else m.support_vectors_
        ax.scatter(sv_pts[:, 0], sv_pts[:, 1], s=90, facecolors='none', edgecolors='k', linewidths=1.0)
        acc = accuracy(y_test, m.predict(Z_test))
        ax.set_title(f'{title}\nточность на тесте = {acc:.3f}')
        ax.set_xlabel('Первая главная компонента (PC1)')
        ax.set_ylabel('Вторая главная компонента (PC2)')
        ax.legend(handles=legend_items, loc='lower left', fontsize=8)

    fig.suptitle('Сравнение решений SVM в проекции на две главные компоненты', fontsize=14)
    plt.tight_layout()
    plt.savefig(img_dir / 'decision_boundaries.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()