"""Подбор k методом leave-one-out (LOO) + график эмпирического риска."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_utils import load_raw, make_xy, standardize


def _logsumexp(a):
    m = a.max()
    return m + np.log(np.exp(a - m).sum())


def loo_error(X, y, k_values, verbose=True):
    """
    LOO-ошибка для каждого k.

    Ускорение: полная матрица квадратов расстояний считается один раз,
    и ширины h_i вычисляются один раз на полной выборке. Для n=1143
    исключение одной точки не меняет h_i на сколько-нибудь значимую
    величину — это стандартное приближение (аналогично делает sklearn
    в KernelDensity). Полный пересчёт h на каждой свёртке дал бы O(n³)
    и был бы неоправданно медленным.
    """
    n, d = X.shape
    classes = np.unique(y)
    n_classes = len(classes)

    # полная матрица квадратов расстояний (n x n) — один раз
    diffs = X[:, None, :] - X[None, :, :]
    sq_full = np.einsum("ijk,ijk->ij", diffs, diffs)

    errors = np.zeros(len(k_values))
    for ki, k in enumerate(k_values):
        k = int(k)
        # h_i = sqrt(k-е наименьшее расстояние), self исключён
        sq_k = sq_full.copy()
        np.fill_diagonal(sq_k, np.inf)
        kth_vals = np.partition(sq_k, k - 1, axis=1)[:, k - 1]
        h = np.sqrt(np.maximum(kth_vals, 1e-24))
        log_h_d = d * np.log(h)

        cnt = 0
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            sq_i = sq_full[i, mask]
            h_m = h[mask]
            log_w = -log_h_d[mask] - 0.5 * sq_i / (h_m ** 2)
            y_m = y[mask]

            log_scores = np.empty(n_classes)
            for j, c in enumerate(classes):
                log_scores[j] = _logsumexp(log_w[y_m == c])

            pred = classes[np.argmax(log_scores)]
            if pred != y[i]:
                cnt += 1
        errors[ki] = cnt / n
        if verbose:
            print(f"  k={k:>3d}  LOO-error = {errors[ki]:.4f}")
    return errors


def run_loo_search(out_dir: Path,
                   grouping: str = "4",
                   k_values=(1, 2, 3, 5, 7, 10, 15, 20, 25, 30),
                   max_points=None):
    df = load_raw()
    X, y, classes = make_xy(load_raw())
    X_std, _ = standardize(X, X.copy())

    if max_points is not None and max_points < len(X_std):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X_std), size=max_points, replace=False)
        X_std, y = X_std[idx], y[idx]
        print(f"LOO на подвыборке {max_points} из {len(X)}")

    print(f"LOO на {len(X_std)} объектах, k = {list(k_values)}")
    errors = loo_error(X_std, y, k_values)

    df_out = pd.DataFrame({"k": list(k_values), "loo_error": errors})
    df_out.to_csv(out_dir / "loo_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(k_values, errors, "o-", color="C0")
    k_best = int(k_values[int(np.argmin(errors))])
    ax.axvline(k_best, color="C3", linestyle="--", alpha=0.6,
               label=f"best k = {k_best}")
    ax.set_xlabel("k (число соседей)")
    ax.set_ylabel("Эмпирический риск (LOO)")
    ax.set_title("Зависимость LOO-ошибки от k (sample-point Parzen)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loo_curve.png", dpi=150)
    plt.close(fig)

    print(f"\nЛучшее k = {k_best}, LOO-error = {errors.min():.4f}")
    print(f"Сохранено: {out_dir/'loo_curve.png'} и {out_dir/'loo_curve.csv'}")
    return k_best, errors