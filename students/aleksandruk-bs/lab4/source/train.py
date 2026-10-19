"""Эксперименты лабораторной №4: PCA через SVD, эффективная размерность, регрессия."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parent
IMAGE_DIR = ROOT / "image"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from dataset import correlation, load_housing, variance_inflation
from pca import (
    approximation_error,
    effective_dimension,
    fit_pca,
    gcv_scores,
    ols_coefficients,
    predict_ols,
    predict_ridge,
    ridge_coefficients,
    scree_dimension,
)

plt.rcParams.update({"figure.figsize": (8, 5), "axes.grid": True, "font.size": 11})
EPS = 0.05


def save_fig(fig, name: str) -> None:
    IMAGE_DIR.mkdir(exist_ok=True)
    fig.savefig(IMAGE_DIR / f"{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def align_signs(ours: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Столбцы собственных векторов домножаются на ±1, чтобы совпасть с эталоном."""
    signs = np.sign(np.sum(ours * reference, axis=0))
    signs[signs == 0.0] = 1.0
    return ours * signs


def self_check(model, y: np.ndarray) -> None:
    eye = model.U.T @ model.U
    if np.max(np.abs(eye - np.eye(eye.shape[0]))) > 1e-8:
        raise RuntimeError("U не ортонормирована")
    if np.max(np.abs(model.G - model.V * model.sigma)) > 1e-8:
        raise RuntimeError("G не совпало с V D")
    restored = model.V * model.sigma @ model.U.T
    if np.max(np.abs(restored - model.F)) > 1e-8:
        raise RuntimeError("полное SVD не восстановило F")
    errors = approximation_error(model.lam)
    total = float(np.sum(model.F * model.F))
    for m in range(model.F.shape[1] + 1):
        gap = model.reconstruct(m) - model.F
        direct = float(np.sum(gap * gap)) / total
        if abs(direct - errors[m]) > 1e-8:
            raise RuntimeError(f"E_{m} не совпало с нормой Фробениуса")
    beta = ols_coefficients(model, y, m=model.F.shape[1])
    y_mean = float(np.mean(y))
    y_hat = predict_ols(model, beta, model.F * model.scale + model.mean, y_mean)
    lstsq, *_ = np.linalg.lstsq(model.F, y - y_mean, rcond=None)
    y_lstsq = model.F @ lstsq + y_mean
    if np.max(np.abs(y_hat - y_lstsq)) > 1e-6:
        raise RuntimeError("МНК в главных компонентах не совпал с обычным МНК")


def plot_dataset(X: np.ndarray, names: tuple[str, ...], corr: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    std = X.std(axis=0)
    axes[0].bar(range(len(names)), std, color="#1f78b4", label="СКО до нормировки")
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=40, ha="right")
    axes[0].set_ylabel("СКО, логарифмическая шкала")
    axes[0].set_title("Масштаб признаков")
    axes[0].legend(loc="upper left")

    image = axes[1].imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_yticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=40, ha="right")
    axes[1].set_yticklabels(names)
    axes[1].set_title("Корреляции признаков")
    axes[1].grid(False)
    bar = fig.colorbar(image, ax=axes[1], fraction=0.046)
    bar.set_label("корреляция")
    fig.tight_layout()
    save_fig(fig, "01_dataset")


def plot_spectrum(lam: np.ndarray, scree_m: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    index = np.arange(1, len(lam) + 1)
    ax.plot(index, lam, "o-", color="#1f78b4", label="λ_j")
    ax.axvline(scree_m, color="#ff7f00", linestyle="--", label=f"крутой склон, m = {scree_m}")
    ax.set_xlabel("номер компоненты")
    ax.set_ylabel("собственное значение")
    ax.set_xticks(index)
    ax.set_title("Спектр F^T F")
    ax.legend(loc="best")
    save_fig(fig, "02_spectrum")


def plot_error(errors: np.ndarray, m_eps: int, m_scree: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ms = np.arange(len(errors))
    ax.plot(ms, errors, "o-", color="#1f78b4", label="E_m")
    ax.axhline(EPS, color="0.3", linestyle=":", label=f"порог ε = {EPS}")
    if m_eps == m_scree:
        ax.axvline(m_eps, color="#ff7f00", linestyle="--", label=f"оба критерия, m = {m_eps}")
    else:
        ax.axvline(m_eps, color="#33a02c", linestyle="--", label=f"E_m ≤ ε, m = {m_eps}")
        ax.axvline(m_scree, color="#ff7f00", linestyle="--", label=f"крутой склон, m = {m_scree}")
    ax.set_xlabel("число компонент m")
    ax.set_ylabel("относительная ошибка восстановления")
    ax.set_xticks(ms)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Ошибка приближения матрицы признаков")
    ax.legend(loc="upper right")
    save_fig(fig, "03_error")


def plot_sklearn(model, sk: PCA) -> dict:
    U_ref = sk.components_.T
    U_aligned = align_signs(model.U, U_ref)
    G_aligned = model.F @ U_aligned
    G_ref = sk.transform(model.F)
    ratio = model.lam / model.lam.sum()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    index = np.arange(1, len(ratio) + 1)
    axes[0].plot(index, ratio, "o-", color="#1f78b4", label="своя доля λ")
    axes[0].plot(index, sk.explained_variance_ratio_, "s--", color="#ff7f00", label="sklearn")
    axes[0].set_xlabel("компонента")
    axes[0].set_ylabel("доля собственного значения")
    axes[0].set_xticks(index)
    axes[0].set_title("Объяснённая доля")
    axes[0].legend(loc="best")

    axes[1].scatter(G_ref[:, 0], G_aligned[:, 0], s=8, c="#1f78b4", label="первая компонента", alpha=0.5)
    limit = float(np.max(np.abs(G_ref[:, 0]))) * 1.05
    axes[1].plot([-limit, limit], [-limit, limit], color="black", linewidth=1, label="совпадение")
    axes[1].set_xlabel("sklearn, первая компонента")
    axes[1].set_ylabel("своя G после согласования знака")
    axes[1].set_title("Проекции на первую компоненту")
    axes[1].legend(loc="best")
    fig.tight_layout()
    save_fig(fig, "04_sklearn")
    return {
        "max_abs_U": float(np.max(np.abs(U_aligned - U_ref))),
        "max_abs_G": float(np.max(np.abs(G_aligned - G_ref))),
        "max_abs_ratio": float(np.max(np.abs(ratio - sk.explained_variance_ratio_))),
        "max_abs_variance": float(np.max(np.abs(model.lam / (len(model.F) - 1) - sk.explained_variance_))),
    }


def plot_regression(ms: np.ndarray, test_rmse: np.ndarray, full_rmse: float, m_eps: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(ms, test_rmse, "o-", color="#1f78b4", label="МНК в m компонентах")
    ax.axhline(full_rmse, color="black", linestyle="--", label="МНК по всем признакам")
    ax.axvline(m_eps, color="#33a02c", linestyle="--", label=f"эффективная размерность m = {m_eps}")
    ax.set_xlabel("число компонент m")
    ax.set_ylabel("RMSE на тесте")
    ax.set_xticks(ms)
    ax.set_title("Регрессия стоимости жилья")
    ax.legend(loc="best")
    save_fig(fig, "05_regression")


def plot_ridge(alphas: np.ndarray, gcv: np.ndarray, test_rmse: np.ndarray, alpha_hat: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    axes[0].plot(alphas, gcv, color="#1f78b4", label="GCV")
    axes[0].axvline(alpha_hat, color="#ff7f00", linestyle="--", label=f"минимум, α = {alpha_hat:.3g}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("α")
    axes[0].set_ylabel("GCV")
    axes[0].set_title("Подбор гребня по обучению")
    axes[0].legend(loc="best")
    axes[1].plot(alphas, test_rmse, color="#1f78b4", label="RMSE на тесте")
    axes[1].axvline(alpha_hat, color="#ff7f00", linestyle="--", label="α, выбранное по GCV")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("α")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Гребневая регрессия на тесте")
    axes[1].legend(loc="best")
    fig.tight_layout()
    save_fig(fig, "06_ridge")


def regression_curve(model, data) -> list[dict]:
    y_mean = float(np.mean(data.y_train))
    rows = []
    for m in range(1, model.F.shape[1] + 1):
        beta = ols_coefficients(model, data.y_train, m)
        train_hat = predict_ols(model, beta, data.X_train, y_mean)
        test_hat = predict_ols(model, beta, data.X_test, y_mean)
        rows.append(
            {
                "m": m,
                "train_rmse": rmse(data.y_train, train_hat),
                "test_rmse": rmse(data.y_test, test_hat),
                "train_r2": float(r2_score(data.y_train, train_hat)),
                "test_r2": float(r2_score(data.y_test, test_hat)),
            }
        )
    return rows


def main() -> None:
    data = load_housing()
    corr = correlation(data.X_train)
    vif = variance_inflation(corr)
    model = fit_pca(data.X_train, scale=True)
    self_check(model, data.y_train)

    errors = approximation_error(model.lam)
    m_eps = effective_dimension(errors, EPS)
    m_scree = scree_dimension(model.lam)
    plot_dataset(data.X_train, data.feature_names, corr)
    plot_spectrum(model.lam, m_scree)
    plot_error(errors, m_eps, m_scree)

    sk = PCA(n_components=None, svd_solver="full").fit(model.F)
    sklearn_gap = plot_sklearn(model, sk)

    curve = regression_curve(model, data)
    full = curve[-1]
    plot_regression(
        np.array([row["m"] for row in curve]),
        np.array([row["test_rmse"] for row in curve]),
        full["test_rmse"],
        m_eps,
    )

    # Эталонный МНК на тех же нормированных признаках.
    lin = LinearRegression().fit(model.transform(data.X_train), data.y_train)
    lin_test = lin.predict(model.transform(data.X_test))
    beta_full = ols_coefficients(model, data.y_train, model.F.shape[1])
    # β в координатах компонент переводится в исходные веса: w = U β.
    weights = model.U @ beta_full

    alphas = np.logspace(-3, 3, 61)
    gcv = gcv_scores(model, data.y_train, alphas)
    alpha_hat = float(alphas[int(np.argmin(gcv))])
    ridge_test = []
    for alpha in alphas:
        beta = ridge_coefficients(model, data.y_train, float(alpha))
        hat = predict_ridge(model, beta, data.X_test, float(np.mean(data.y_train)))
        ridge_test.append(rmse(data.y_test, hat))
    plot_ridge(alphas, gcv, np.array(ridge_test), alpha_hat)

    beta_ridge = ridge_coefficients(model, data.y_train, alpha_hat)
    ridge_hat = predict_ridge(model, beta_ridge, data.X_test, float(np.mean(data.y_train)))
    sk_ridge = Ridge(alpha=alpha_hat, fit_intercept=True).fit(model.F, data.y_train)
    sk_ridge_hat = sk_ridge.predict(model.transform(data.X_test))

    chosen = curve[m_eps - 1]
    summary = {
        "n_train": int(len(data.y_train)),
        "n_test": int(len(data.y_test)),
        "n_features": int(model.F.shape[1]),
        "feature_std": data.X_train.std(axis=0).tolist(),
        "correlation": corr.tolist(),
        "vif": vif.tolist(),
        "condition": float(np.linalg.cond(corr)),
        "lambda": model.lam.tolist(),
        "sigma": model.sigma.tolist(),
        "errors": errors.tolist(),
        "m_eps": m_eps,
        "m_scree": m_scree,
        "eps": EPS,
        "sklearn": sklearn_gap,
        "regression": curve,
        "ols_weight_gap": float(np.max(np.abs(weights - lin.coef_))),
        "ols_test_rmse_gap": rmse(data.y_test, lin_test) - full["test_rmse"],
        "pca_m": chosen,
        "ridge_alpha": alpha_hat,
        "ridge_test_rmse": rmse(data.y_test, ridge_hat),
        "ridge_test_r2": float(r2_score(data.y_test, ridge_hat)),
        "ridge_coef_gap": float(np.max(np.abs(beta_ridge - sk_ridge.coef_))),
        "ridge_predict_gap": float(np.max(np.abs(ridge_hat - sk_ridge_hat))),
        "full_test_rmse": full["test_rmse"],
        "full_test_r2": full["test_r2"],
    }
    IMAGE_DIR.mkdir(exist_ok=True)
    (IMAGE_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "correlation"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
