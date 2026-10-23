import numpy as np


def n_components_for_variance(explained_variance_ratio: np.ndarray,
                              threshold: float = 0.95) -> int:
    cumulative = np.cumsum(explained_variance_ratio)
    idx = int(np.searchsorted(cumulative, threshold)) + 1
    return min(idx, len(explained_variance_ratio))


def n_components_kaiser(explained_variance: np.ndarray) -> int:
    return int(np.sum(explained_variance > 1.0))


def effective_rank(singular_values: np.ndarray) -> float:
    p = np.asarray(singular_values, dtype=float) ** 2
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log(p)))
    return float(np.exp(entropy))
