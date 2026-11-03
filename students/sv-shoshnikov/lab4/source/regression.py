import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


class SVDRegression:
    """Линейная регрессия через SVD: w = sum_j f_j (u_j^T y / s_j) v_j.

    method: "ols" (f_j = 1), "ridge" (f_j = s_j^2 / (s_j^2 + tau)), "pcr" (f_j = 1 при j < k, иначе 0).
    Стандартизация признаков и центрирование ответа выполняются внутри fit по обучающим данным.
    """

    def __init__(self, method="ols", param=None):
        self.method = method
        self.param = param

    def _filter(self, s):
        if self.method == "ols":
            return np.ones_like(s)
        if self.method == "ridge":
            return s ** 2 / (s ** 2 + self.param)
        if self.method == "pcr":
            return (np.arange(len(s)) < self.param).astype(float)
        raise ValueError(f"Неизвестный метод: {self.method}")

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.mean_, self.std_ = X.mean(axis=0), X.std(axis=0)
        self.y_mean_ = y.mean()
        Z = (X - self.mean_) / self.std_
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        self.singular_values_ = s
        keep = s > s.max() * max(Z.shape) * np.finfo(float).eps
        gain = np.where(keep, self._filter(s) / np.where(keep, s, 1.0), 0.0)
        self.w_ = Vt.T @ (gain * (U.T @ (y - self.y_mean_)))
        return self

    def predict(self, X):
        return ((np.asarray(X, dtype=float) - self.mean_) / self.std_) @ self.w_ + self.y_mean_

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def degrees_of_freedom(self):
        """След матрицы сглаживания: sum_j f_j (для ridge это sum_j lambda_j / (lambda_j + tau))."""
        return float(np.sum(self._filter(self.singular_values_)))


def cv_scores(X, y, method, params, folds=5, seed=42):
    """Средний R^2 по k-fold CV для каждого значения параметра (стандартизация пересчитывается в блоках)."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    splits = list(KFold(folds, shuffle=True, random_state=seed).split(X))
    return np.array([
        np.mean([r2_score(y[b], SVDRegression(method, p).fit(X[a], y[a]).predict(X[b])) for a, b in splits])
        for p in params
    ])


def gcv_scores(X, y, taus):
    """Обобщённая кросс-валидация для Ridge: GCV(tau) = ||y - y_hat||^2 / (n - 1 - tr H)^2, tr H = sum_j f_j."""
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    yc = y - y.mean()
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    coef = U.T @ yc
    out = []
    for tau in taus:
        f = s ** 2 / (s ** 2 + tau)
        resid = yc - U @ (f * coef)
        out.append(resid @ resid / (len(y) - 1.0 - f.sum()) ** 2)
    return np.array(out)
