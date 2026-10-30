"""Проверки реализации: основная теорема PCA и сравнение с sklearn.decomposition.PCA."""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA as SklearnPCA

from pca import PCA


@dataclass
class Comparison:
    n_components: int
    components: float          # max |V_ours − V_sk| после выравнивания знаков
    explained_variance: float
    explained_variance_ratio: float
    singular_values: float
    transformed: float         # max |G_ours − G_sk|
    reconstructed: float       # max |X̂_ours − X̂_sk|
    sign_flips: int            # сколько компонент различались знаком

    @property
    def max_error(self) -> float:
        return max(self.components, self.explained_variance, self.explained_variance_ratio,
                   self.singular_values, self.transformed, self.reconstructed)


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def compare(X: np.ndarray, n_components: int) -> Comparison:
    ours = PCA(n_components).fit(X)
    ref = SklearnPCA(n_components=n_components, svd_solver="full").fit(X)

    # Главные направления совпадают с точностью до знака — выравниваем по скалярному произведению.
    signs = np.sign(np.sum(ours.components_ * ref.components_, axis=1))
    ref_components = ref.components_ * signs[:, None]
    G_ours, G_ref = ours.transform(X), ref.transform(X) * signs

    return Comparison(
        n_components=n_components,
        components=_max_abs(ours.components_, ref_components),
        explained_variance=_max_abs(ours.explained_variance_, ref.explained_variance_),
        explained_variance_ratio=_max_abs(ours.explained_variance_ratio_, ref.explained_variance_ratio_),
        singular_values=_max_abs(ours.singular_values_, ref.singular_values_),
        transformed=_max_abs(G_ours, G_ref),
        reconstructed=_max_abs(ours.inverse_transform(G_ours), ref.inverse_transform(ref.transform(X))),
        sign_flips=int(np.sum(signs < 0)),
    )


def reconstruction_errors(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Относительная ошибка восстановления ‖X − X̂‖² / ‖X − x̄‖² для m = 1, …, n."""
    n = X.shape[1]
    total = np.sum((X - X.mean(axis=0)) ** 2)
    ours, ref = np.empty(n), np.empty(n)
    for m in range(1, n + 1):
        p = PCA(m).fit(X)
        ours[m - 1] = np.sum((X - p.inverse_transform(p.transform(X))) ** 2) / total
        s = SklearnPCA(n_components=m, svd_solver="full").fit(X)
        ref[m - 1] = np.sum((X - s.inverse_transform(s.transform(X))) ** 2) / total
    return ours, ref


@dataclass
class TheoremCheck:
    n_components: int
    orthonormal_u: float       # max |UᵀU − I_m|
    orthogonal_g: float        # max |GᵀG − Λ| / λ_1
    eigen_u: float             # max |FᵀF U − U Λ| / λ_1
    residual: float            # |‖GUᵀ − F‖² − Σ_{j>m} λ_j| / ‖F‖²

    @property
    def max_error(self) -> float:
        return max(self.orthonormal_u, self.orthogonal_g, self.eigen_u, self.residual)


def check_theorem(X: np.ndarray, n_components: int) -> TheoremCheck:
    """Свойства 1–4 основной теоремы PCA (лекция 4, слайд 21) для центрированной F."""
    pca = PCA(n_components).fit(X)
    F = X - pca.mean_
    U = pca.components_.T
    G = pca.transform(X)
    lam = pca.eigenvalues_
    scale = pca.all_eigenvalues_[0]
    total = np.sum(F ** 2)
    return TheoremCheck(
        n_components=n_components,
        orthonormal_u=_max_abs(U.T @ U, np.eye(n_components)),
        orthogonal_g=_max_abs(G.T @ G, np.diag(lam)) / scale,
        eigen_u=_max_abs(F.T @ F @ U, U * lam) / scale,
        residual=abs(np.sum((G @ U.T - F) ** 2) - pca.all_eigenvalues_[n_components:].sum()) / total,
    )
