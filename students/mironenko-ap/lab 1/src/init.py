"""Инициализация весов и мультистарт."""
import numpy as np
from .core import log

def init_random(d, rng=None):
    rng = rng or np.random.default_rng(0)
    return rng.normal(0.0, 1.0 / np.sqrt(d), d), 0.0


def init_correlation(X, y):
    """w_j = corr(x_j, y), нормированный к 1/sqrt(d)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    denom = np.linalg.norm(Xc, axis=0) * np.linalg.norm(yc) + 1e-12
    w = np.nan_to_num((Xc.T @ yc) / denom)
    n = np.linalg.norm(w)
    if n > 1e-12:
        w = w / n / np.sqrt(X.shape[1])
    return w, 0.0


def multistart(fit_fn, X_tr, y_tr, X_val, y_val,
               n_starts=5, seed=0, lam=0.0, use_l2=True):
    from .core import loss_only
    lam_eff = lam if use_l2 else 0.0
    rng = np.random.default_rng(seed)
    d = X_tr.shape[1]
    best = None

    for k in range(n_starts):
        log(f"[multistart] запуск {k + 1}/{n_starts}")
        w0, b0 = init_random(d, rng)
        w, b, hist = fit_fn(
            w0, b0, X_tr, y_tr,
            X_val=X_val, y_val=y_val, verbose=False,
        )
        val_loss = loss_only(X_val, y_val, w, b, lam_eff)
        log(f"  → val_loss={val_loss:.4f}")
        if best is None or val_loss < best[0]:
            best = (val_loss, w, b, hist)

    _, w, b, hist = best
    log(f"[multistart] лучший val_loss={best[0]:.4f}")
    return w, b, hist