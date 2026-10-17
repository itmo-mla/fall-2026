"""PCA через сингулярное разложение (лекция, слайды 19-22).

F = V D U^T  (слайд 7), столбцы U - главные направления, G = F U - новые признаки,
lambda_j = d_j^2 - собственные значения F^T F (слайд 21, теорема).
"""
import numpy as np


class PCASVD:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, F):
        F = np.asarray(F, dtype=float)
        self.l_, self.n_ = F.shape
        self.mean_ = F.mean(axis=0)
        Fc = F - self.mean_
        # numpy: Fc = V diag(d) Ut, строки Ut - столбцы U из лекции
        V, d, Ut = np.linalg.svd(Fc, full_matrices=False)
        # детерминированный знак: наибольший по модулю элемент компоненты > 0
        idx = np.argmax(np.abs(Ut), axis=1)
        sign = np.sign(Ut[np.arange(Ut.shape[0]), idx])
        sign[sign == 0] = 1.0
        Ut, V = Ut * sign[:, None], V * sign[None, :]
        self.V_, self.d_, self.U_ = V, d, Ut.T
        self.eigvals_ = d ** 2                      # lambda_1 >= ... >= lambda_n
        self.m_ = self.n_ if self.n_components is None else int(self.n_components)
        return self

    # --- свойства в терминах sklearn (для сравнения) ---
    @property
    def components_(self):                          # m x n, строки = u_j^T
        return self.U_[:, :self.m_].T

    @property
    def singular_values_(self):
        return self.d_[:self.m_]

    @property
    def explained_variance_(self):                  # lambda_j / (l - 1)
        return self.eigvals_[:self.m_] / (self.l_ - 1)

    @property
    def explained_variance_ratio_(self):            # lambda_j / sum(lambda)
        return self.eigvals_[:self.m_] / self.eigvals_.sum()

    def transform(self, F):
        """G = F U_m (слайд 21, теорема)."""
        return (np.asarray(F, float) - self.mean_) @ self.U_[:, :self.m_]

    def inverse_transform(self, G):
        """F_hat = G U_m^T (слайд 20)."""
        return np.asarray(G, float) @ self.U_[:, :self.m_].T + self.mean_

    def reconstruction_error(self, F):
        """||G U^T - F||^2 (слайд 20); по теореме = sum_{j>m} lambda_j на обучающей F."""
        F = np.asarray(F, float)
        return float(np.sum((self.inverse_transform(self.transform(F)) - F) ** 2))
