"""Подбор k методом leave-one-out (LOO) + график эмпирического риска."""

"""
Замечание :
    В LOO исключённый объект остаётся в расчёте ширин окон других объектов: h заранее вычислен по всей выборке, затем только маскируется. 
    Для sample-point окна удаление объекта может менять h у соседей и их веса, особенно при малом k. 
    Пересчитывать/корректно обновлять ширины без контрольного объекта; 
    текущее приближение не является заявленным LOO.
"""

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

    n, d = X.shape
    classes = np.unique(y)
    n_classes = len(classes)

    # полная матрица квадратов расстояний (n x n) — один раз
    diffs = X[:, None, :] - X[None, :, :]
    sq_full = np.einsum("ijk,ijk->ij", diffs, diffs)

    errors = np.zeros(len(k_values))
    for ki, k in enumerate(k_values):
        k = int(k)
        if k > n - 2:
            raise ValueError(
                f"k={k} слишком большой для LOO при n={n}" #словили ошибку
            )

        sq_k = sq_full.copy()
        np.fill_diagonal(sq_k, np.inf)

        # k-е и (k+1)-е наименьшие расстояния от каждой точки
        part = np.partition(sq_k, [k - 1, k], axis=1)
        kth_sq    = part[:, k - 1]
        kplus1_sq = part[:, k]

        # c_m — сколько точек j != m лежат не дальше kth_m
        count_le_kth = np.sum(sq_k <= kth_sq[:, None], axis=1)

        h_kth    = np.sqrt(np.maximum(kth_sq,    1e-24))
        h_kplus1 = np.sqrt(np.maximum(kplus1_sq, 1e-24))
        log_h_kth_d    = d * np.log(h_kth)
        log_h_kplus1_d = d * np.log(h_kplus1)

        # changed_mat[i, m] = True, если при исключении i ширина h_m сдвигается на (k+1)-ю статистику.
        changed_mat = ((sq_full <= kth_sq[None, :])
                       & (count_le_kth[None, :] == k))
        np.fill_diagonal(changed_mat, False)

        cnt = 0
        for i in range(n):
            changed   = changed_mat[i]
            h_m       = np.where(changed, h_kplus1, h_kth)
            log_h_m_d = np.where(changed, log_h_kplus1_d, log_h_kth_d)

            mask = np.ones(n, dtype=bool)
            mask[i] = False

            sq_i       = sq_full[i, mask]
            h_mi       = h_m[mask]
            log_h_mi_d = log_h_m_d[mask]
            log_w = -log_h_mi_d - 0.5 * sq_i / (h_mi ** 2)
            y_m   = y[mask]

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


def loo_error_naive(X, y, k_values):
    """Наивный O(n^3) LOO для сверки корректности."""
    n, d = X.shape
    classes = np.unique(y)
    n_classes = len(classes)
    errors = np.zeros(len(k_values))

    for ki, k in enumerate(k_values):
        k = int(k)
        cnt = 0
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            Xm, ym = X[mask], y[mask]

            diffs = Xm[:, None, :] - Xm[None, :, :]
            sq = np.einsum("ijk,ijk->ij", diffs, diffs)
            np.fill_diagonal(sq, np.inf)
            kth = np.partition(sq, k - 1, axis=1)[:, k - 1]
            h = np.sqrt(np.maximum(kth, 1e-24))

            di = Xm - X[i]
            sq_i = np.einsum("ij,ij->i", di, di)
            log_w = -d * np.log(h) - 0.5 * sq_i / (h ** 2)

            log_scores = np.empty(n_classes)
            for j, c in enumerate(classes):
                log_scores[j] = _logsumexp(log_w[ym == c])
            if classes[np.argmax(log_scores)] != y[i]:
                cnt += 1
        errors[ki] = cnt / n
    return errors


def run_loo_search(out_dir: Path,
                   k_values=(1, 2, 3, 5, 7, 10, 15, 20, 25, 30),
                   max_points=None):
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