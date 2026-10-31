from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def _save(path: str | Path, show: bool = False) -> None:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    if not show:
        plt.close()


def plot_explained_variance(
    ratios: np.ndarray, output_path: str | Path, show: bool = False,
) -> None:

    x = np.arange(1, len(ratios) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, ratios * 100, alpha=0.8, label="Individual variance (%)")
    ax.plot(x, np.cumsum(ratios) * 100, marker="o", label="Cumulative variance (%)")
    ax.axhline(95, linestyle="--", label="95% threshold")
    ax.set(xlabel="Number of components", ylabel="Explained variance (%)", title="PCA explained variance")
    ax.set_xticks(x)
    ax.set_ylim(0, 105)
    ax.legend()
    _save(output_path, show=show)


def plot_reconstruction_errors(
    df: pd.DataFrame, output_path: str | Path, epsilon: float = 0.05,
    show: bool = False,
) -> None:

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["n_components"], df["theoretical_E_m"], "o-", label="Theoretical train $E_m$")
    ax.plot(df["n_components"], df["train_E_m"], "x--", label="Measured train $E_m$")
    ax.plot(df["n_components"], df["val_E_m"], "^-", label="Validation relative error")
    ax.plot(df["n_components"], df["test_E_m"], "s-", label="Test relative error")
    ax.axhline(epsilon, linestyle=":", color="black", label=rf"$\epsilon={epsilon}$")
    ax.set(xlabel="Number of components", ylabel="Relative squared reconstruction error",
           title="PCA reconstruction quality")
    ax.set_xticks(df["n_components"])
    ax.legend()
    _save(output_path, show=show)


def plot_pca_2d(
    X_2d: np.ndarray, y: np.ndarray, output_path: str | Path, show: bool = False,
) -> None:

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="viridis", s=30, alpha=0.85)
    fig.colorbar(scatter, ax=ax, label="Diabetes progression (target)")
    ax.set(xlabel="Principal component 1", ylabel="Principal component 2",
           title="Train data projected onto first two components")
    _save(output_path, show=show)


def plot_validation_regression(
    df: pd.DataFrame, output_path: str | Path, best_k: int, show: bool = False,
) -> None:

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["n_components"], df["val_rmse_custom"], "o-", label="Custom PCA + Ridge SVD")
    ax.plot(df["n_components"], df["val_rmse_sklearn"], "x--", label="sklearn PCA + Ridge")
    ax.axvline(best_k, color="black", linestyle=":", label=f"Best k = {best_k}")
    ax.set(xlabel="Number of components", ylabel="Validation RMSE",
           title="Ridge regression after PCA")
    ax.set_xticks(df["n_components"])
    ax.legend()
    _save(output_path, show=show)


def plot_predictions(
    y: np.ndarray, predictions: dict[str, np.ndarray], output_path: str | Path,
    show: bool = False,
) -> None:

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, y_pred in predictions.items():
        ax.scatter(y, y_pred, alpha=0.65, s=35, label=label)
    lo = min(float(np.min(y)), *(float(np.min(v)) for v in predictions.values()))
    hi = max(float(np.max(y)), *(float(np.max(v)) for v in predictions.values()))
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", label="Perfect prediction")
    ax.set(xlabel="True target", ylabel="Predicted target", title="Ridge regression on test data")
    ax.legend()
    _save(output_path, show=show)
