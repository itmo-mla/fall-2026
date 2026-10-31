import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error = y_true - y_pred
    mse = float(np.mean(error ** 2))
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))

    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(1.0 - np.sum(error ** 2) / denominator) if denominator > 0 else float("nan"),
    }

def reconstruction_mse(X: np.ndarray, X_reconstructed: np.ndarray) -> float:
    return float(np.mean((np.asarray(X) - np.asarray(X_reconstructed)) ** 2))

def relative_reconstruction_error(
        X: np.ndarray,
        X_reconstructed: np.ndarray,
        center: np.ndarray
) -> float:
    numerator = np.linalg.norm(X - X_reconstructed, ord="fro") ** 2
    denominator = np.linalg.norm(X - center, ord="fro") ** 2
    return float(numerator / denominator) if denominator > 0 else float("nan")

def compare_pca(custom_pca, reference_pca, X: np.ndarray) -> dict[str, float]:
    W = custom_pca.components_
    W_ref = reference_pca.components_

    signs = np.where(np.sum(W * W_ref, axis=1) >= 0, 1.0, -1.0)
    scores = custom_pca.transform(X)
    scores_ref = reference_pca.transform(X)

    reconstructed = custom_pca.inverse_transform(scores)
    reconstructed_ref = reference_pca.inverse_transform(scores_ref)

    return {
        "mean_max_abs_diff": float(np.max(np.abs(custom_pca.mean_ - reference_pca.mean_))),
        "singular_max_abs_diff": float(np.max(np.abs(custom_pca.singular_values_ - reference_pca.singular_values_))),
        "explained_variance_max_abs_diff": float(np.max(np.abs(custom_pca.explained_variance_ - reference_pca.explained_variance_))),
        "explained_ratio_max_abs_diff": float(np.max(np.abs(custom_pca.explained_variance_ratio_ - reference_pca.explained_variance_ratio_))),
        "components_max_abs_diff_sign_aligned": float(np.max(np.abs(W * signs[:, None] - W_ref))),
        "scores_max_abs_diff_sign_aligned": float(np.max(np.abs(scores * signs - scores_ref))),
        "projector_max_abs_diff": float(np.max(np.abs(W.T @ W - W_ref.T @ W_ref))),
        "reconstruction_max_abs_diff": float(np.max(np.abs(reconstructed - reconstructed_ref))),
    }