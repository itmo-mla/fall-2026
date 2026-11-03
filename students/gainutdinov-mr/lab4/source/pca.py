"""Собственная реализация алгоритма: PCA."""
# U, D, Vh = np.linalg.svd(FT_x_F, full_matrices=True)
# в лекции V @ D @ U.T
# Вместо U @ D V.T
# Получается в нашем варианте

import numpy as np


class PCA:
    """
    n_components - число компонент
        > None - оставить все.

    D - сингулярные значения F.
    L - большая лямбда:  = G.T @ G = D**2
    """

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        """Выполняем SVD и сохраняем направления преобразования."""
        F = X
        self.V, self.D, U_T = np.linalg.svd(F, full_matrices=False)
        self.U = U_T.T

        m = len(self.D) if self.n_components is None else self.n_components
        self.U_effective = self.U[:, :m]
        self.V_effective = self.V[:, :m]
        self.L = np.diag(self.D[:m] ** 2)  # L = G.T @ G
        return self

    def transform(self, X):
        """Переход в новое пространство признаков: G = F @ U_m."""
        return X @ self.U_effective

    def inverse_transform(self, Z):
        """Восстановление исходных признаков из новых."""
        return Z @ self.U_effective.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def calculate_em(self):
        """Ошибки E_m для m от 1 до len(D), включая последний ноль."""
        eigs_full_sum = np.sum(self.D**2)
        Ems = []
        for m in range(1, len(self.D) + 1):
            Em = np.sum(self.D[m:] ** 2) / eigs_full_sum
            Ems.append(Em)
        return np.array(Ems)

    def get_model_without_targets(self):
        m = self.U_effective.shape[1]
        D_effective = self.D[:m]
        beta_without_targets = np.diag(1 / D_effective) @ self.V_effective.T
        G_train = self.V_effective @ np.diag(D_effective)
        return G_train @ beta_without_targets
