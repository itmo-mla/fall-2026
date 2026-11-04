"""Оценка эффективной размерности выборки по спектру FᵀF (лекция 4, слайд 23)."""

import numpy as np


def residual_variance(eigenvalues: np.ndarray) -> np.ndarray:
    """E_m = (λ_{m+1} + … + λ_n) / (λ_1 + … + λ_n) для m = 0, …, n.

    По основной теореме PCA E_m = ‖G Uᵀ − F‖² / ‖F‖² — относительная
    ошибка восстановления по первым m компонентам.
    """
    tail = np.concatenate([np.cumsum(eigenvalues[::-1])[::-1], [0.0]])
    return tail / eigenvalues.sum()


def effective_dimension(eigenvalues: np.ndarray, eps: float) -> int:
    """Наименьшее m, при котором E_m ≤ ε."""
    E = residual_variance(eigenvalues)
    return int(np.argmax(E <= eps + 1e-12))


def steep_slope_dimension(eigenvalues: np.ndarray) -> int:
    """Критерий «крутого склона»: m, при котором E_{m−1} / E_m максимально.

    m = n не рассматривается: там E_n = 0 и отношение бесконечно при любом спектре.
    """
    E = residual_variance(eigenvalues)
    ratios = E[:-2] / E[1:-1]  # E_{m−1} / E_m для m = 1, …, n − 1
    return int(np.argmax(ratios)) + 1


def condition_number(eigenvalues: np.ndarray) -> float:
    """μ(FᵀF) = λ_max / λ_min — во столько раз МНК усиливает погрешности (слайд 9)."""
    return float(eigenvalues.max() / eigenvalues.min())
