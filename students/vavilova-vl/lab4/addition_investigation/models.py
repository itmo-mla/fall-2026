"""МНК, регрессия на главных компонентах и гребневая регрессия через одно SVD.

Обозначения лекции 4: F = V D Uᵀ (F центрирована), D = diag(√λ_j).
Все три метода — это α = Σ_j w_j · u_j (v_jᵀ y) / √λ_j с разными весами w_j:
  МНК:        w_j = 1;
  PCR:        w_j = 1 для j ≤ m, иначе 0;
  гребневая:  w_j = λ_j / (λ_j + τ).
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from gas_data import Preprocessor


@dataclass
class SVDFit:
    """Одно разложение обучающей выборки, из которого решаются все задачи."""
    x_mean: np.ndarray
    y_mean: float
    U: np.ndarray            # n × r, главные направления
    lam: np.ndarray          # λ_j
    Vty: np.ndarray          # v_jᵀ (y − ȳ)

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray) -> "SVDFit":
        x_mean, y_mean = X.mean(axis=0), y.mean()
        V, d, Ut = np.linalg.svd(X - x_mean, full_matrices=False)
        return cls(x_mean, y_mean, Ut.T, d ** 2, V.T @ (y - y_mean))

    def coefficients(self, factors: np.ndarray) -> tuple[np.ndarray, float]:
        """α = U diag(factors) Vᵀ y; factors_j = w_j / √λ_j."""
        alpha = self.U @ (factors * self.Vty)
        return alpha, self.y_mean - self.x_mean @ alpha

    def ols(self):
        return self.coefficients(1 / np.sqrt(self.lam))

    def pcr(self, m: int):
        factors = np.zeros_like(self.lam)
        factors[:m] = 1 / np.sqrt(self.lam[:m])
        return self.coefficients(factors)

    def ridge(self, tau: float):
        # √λ / (λ + τ) — без деления на √λ, поэтому определено и при λ_j = 0.
        return self.coefficients(np.sqrt(self.lam) / (self.lam + tau))


def effective_dimension_ridge(lam: np.ndarray, tau: float) -> float:
    """n_eff(τ) = tr F(FᵀF + τI)⁻¹Fᵀ = Σ λ_j / (λ_j + τ)."""
    return float(np.sum(lam / (lam + tau)))


def grouped_folds(groups: np.ndarray, n_splits: int, n_repeats: int,
                  seed: int = 0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """t×q-fold по сериям испытаний: все режимы одной скважины попадают в один блок.

    Иначе соседний режим той же скважины оказался бы в обучении, и контроль
    проверял бы интерполяцию между штуцерами, а не прогноз для новой скважины.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    for _ in range(n_repeats):
        for block in np.array_split(rng.permutation(unique), n_splits):
            test = np.isin(groups, block)
            yield np.flatnonzero(~test), np.flatnonzero(test)


def random_folds(n_objects: int, n_splits: int, n_repeats: int,
                 seed: int = 0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Обычный t×q-fold по отдельным замерам — для сравнения."""
    return grouped_folds(np.arange(n_objects), n_splits, n_repeats, seed)


Solver = Callable[[SVDFit], tuple[np.ndarray, float]]


def cv_rmse(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds,
            solvers: list[Solver]) -> np.ndarray:
    """RMSE на контроле для каждого решателя.

    X — признаки с пропусками в исходных единицах. Заполнение пропусков,
    стандартизация и SVD строятся только по обучающей части блока; одно SVD
    обслуживает сразу все решатели — так дешевле перебирать m и τ.
    """
    sq_errors = np.zeros(len(solvers))
    count = 0
    for train, test in folds:
        prep = Preprocessor().fit(X[train], groups[train])
        X_train, X_test = prep.transform(X[train], groups[train]), prep.transform(X[test], groups[test])
        svd = SVDFit.fit(X_train, y[train])
        for k, solve in enumerate(solvers):
            alpha, b = solve(svd)
            sq_errors[k] += np.sum((y[test] - X_test @ alpha - b) ** 2)
        count += len(test)
    return np.sqrt(sq_errors / count)


def bootstrap_coefficients(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                           solvers: list[Solver], n_boot: int = 500,
                           seed: int = 0) -> np.ndarray:
    """Коэффициенты на бутстреп-выборках серий испытаний: n_boot × len(solvers) × n.

    Пропуски заполнены и признаки стандартизованы по всей выборке, чтобы
    коэффициенты разных бутстреп-выборок были в одном масштабе. Качество
    прогноза здесь не оценивается, поэтому утечки в оценку нет.
    """
    rng = np.random.default_rng(seed)
    Xs = Preprocessor().fit_transform(X, groups)
    unique = np.unique(groups)
    result = np.empty((n_boot, len(solvers), X.shape[1]))
    for b in range(n_boot):
        idx = np.concatenate([np.flatnonzero(groups == g) for g in rng.choice(unique, len(unique))])
        svd = SVDFit.fit(Xs[idx], y[idx])
        for k, solve in enumerate(solvers):
            result[b, k] = solve(svd)[0]
    return result
