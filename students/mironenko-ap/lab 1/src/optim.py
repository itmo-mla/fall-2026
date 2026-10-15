"""SGD с инерцией, скорейший спуск (exact step для квадратичной потери),
порядок предъявления, EMA, lr-decay."""
import numpy as np
from .core import loss_grad, loss_grad_sample, loss_only, margin, log

def random_order(N, rng):
    """Случайная перестановка индексов 0..N-1."""
    return rng.permutation(N)

# Порядок предъявления
def margin_order(X, y, w, b, reverse=False):
    """Порядок по модулю отступа |M_i|.

    reverse=False — easy-first: сначала |M| большие.
    reverse=True  — hard-first: сначала |M| маленькие.
    """
    mag = np.abs(margin(X, y, w, b))
    return np.argsort(mag)[::-1] if not reverse else np.argsort(mag)


def _ordering(kind, X, y, w, b, rng):
    if kind == "random":
        return random_order(len(y), rng)
    if kind == "margin":            # easy-first
        return margin_order(X, y, w, b, reverse=False)
    if kind == "margin_hard":       # hard-first
        return margin_order(X, y, w, b, reverse=True)
    raise ValueError(f"ordering={kind}")


# SGD с инерцией
def sgd_momentum(X, y, w0, b0, lr=0.01, mu=0.9, lam=1e-3, use_l2=True,
                 epochs=50, ordering="random", rng=None,
                 X_val=None, y_val=None, print_every=10, verbose=True,
                 lr_decay=0.0, ema_alphas=(0.9, 0.99, 0.999),
                 early_stop_delta=5e-3, early_stop_patience=1):
    rng = rng or np.random.default_rng(0)
    w = w0.copy().astype(np.float64)
    b = float(b0)
    vw = np.zeros_like(w)
    vb = 0.0
    history = []
    N = len(y)
    alphas = list(ema_alphas)
    ema_state = {a: None for a in alphas}

    if verbose:
        log(f"\n[momentum] старт | epochs={epochs} | lr={lr} | mu={mu} "
            f"| lam={lam} | use_l2={use_l2} (lam_eff={lam}) "
            f"| ordering={ordering} | N={N} "
            f"| lr_decay={lr_decay} | ema_alphas={alphas}")
    prev_val = None
    n_stale = 0
    for ep in range(1, epochs + 1):
        lr_ep = lr / (1.0 + lr_decay * (ep - 1))
        idx = _ordering(ordering, X, y, w, b, rng)

        for i in idx:
            # градиент — с lam_eff (0.0, если L2 выключена)
            _, gw, gb = loss_grad_sample(X[i], y[i], w, b, lam)
            vw = mu * vw - lr_ep * gw
            vb = mu * vb - lr_ep * gb
            w += vw
            b += vb

            # EMA — всегда по loss без L2, независимо от use_l2
            li = loss_grad_sample(X[i], y[i], w, b, 0.0)[0]
            for a in alphas:
                prev = ema_state[a]
                ema_state[a] = li if prev is None else a * prev + (1 - a) * li

        # полная цель (с L2, если включена) и data loss (всегда без L2)
        loss, _, _ = loss_grad(X, y, w, b, lam)
        data_loss, _, _ = loss_grad(X, y, w, b, 0.0)

        rec = {"loss": float(loss),
               "data_loss": float(data_loss),
               "lr": float(lr_ep)}
        for a in alphas:
            rec[f"ema_{a}"] = float(ema_state[a])
        if X_val is not None:
            vloss = loss_only(X_val, y_val, w, b, lam)
            vpred = np.where(X_val @ w + b >= 0, 1, -1)
            rec["val_loss"] = float(vloss)
            rec["val_acc"] = float(np.mean(vpred == y_val))
        history.append(rec)

        if early_stop_delta is not None and X_val is not None:
            if prev_val is not None and abs(prev_val - rec["val_loss"]) < early_stop_delta:
                n_stale += 1
            else:
                n_stale = 0
            prev_val = rec["val_loss"]
            if n_stale >= early_stop_patience:
                if verbose:
                    log(f"[momentum] early stop на ep {ep}: "
                        f"|Δval| < {early_stop_delta:.2e} "
                        f"в течение {early_stop_patience} эпох")
                break

        if verbose and (ep == 1 or ep % print_every == 0 or ep == epochs):
            line = (f"[momentum] ep {ep:4d}/{epochs} | lr={lr_ep:.5f}"
                    f" | train={loss:.4f} | data={data_loss:.4f}")
            for a in alphas:
                line += f" | ema_{a}={rec[f'ema_{a}']:.4f}"
            if X_val is not None:
                line += (f" | val={rec['val_loss']:.4f}"
                         f" | acc={rec['val_acc']:.4f}")
            log(line)

    return w, b, history


# Скорейший градиентный спуск (полнобатчевый, точный шаг для квадратичной)
def steepest_descent(X, y, w0, b0, lam=1e-3, use_l2=True,
                     epochs=100, X_val=None, y_val=None, print_every=10, verbose=True,
                     ema_alphas=(0.9, 0.99, 0.999),
                     alpha_min=1e-12,
                     early_stop_delta=1e-3, early_stop_patience=1):
    w = w0.copy().astype(np.float64)
    b = float(b0)
    history = []
    alphas = list(ema_alphas)
    ema_state = {a: None for a in alphas}
    N = len(y)
    prev_val = None
    n_stale = 0

    # Предвычислим блоки гессиана
    XtX = (X.T @ X) / N
    Xt1 = X.mean(axis=0)                    # (1/N) X^T 1
    H_ww = XtX + lam * np.eye(X.shape[1])
    H_bb = 1.0

    if verbose:
        log(f"\n[steepest] старт | epochs={epochs} | lam={lam} "
            f"| use_l2={use_l2} (lam_eff={lam}) "
            f"| exact step for quadratic")

    for ep in range(1, epochs + 1):
        loss, gw, gb = loss_grad(X, y, w, b, lam)
        gnorm2 = float(gw @ gw + gb * gb)
        if gnorm2 < 1e-14:
            if verbose:
                log(f"[steepest] ep {ep}: градиент ≈ 0, стоп")
            break

        # g^T H g
        Hg_w = H_ww @ gw + Xt1 * gb
        Hg_b = float(Xt1 @ gw + H_bb * gb)
        gHg = float(gw @ Hg_w + gb * Hg_b)
        alpha = gnorm2 / (gHg + 1e-18)
        if alpha < alpha_min:
            alpha = alpha_min

        w = w - alpha * gw
        b = b - alpha * gb

        # полная цель (с L2, если включена) и data loss (всегда без L2)
        loss_new, _, _ = loss_grad(X, y, w, b, lam)
        data_loss_new, _, _ = loss_grad(X, y, w, b, 0.0)

        # EMA — по data loss, как в sgd_momentum
        for a in alphas:
            prev = ema_state[a]
            ema_state[a] = data_loss_new if prev is None \
                else a * prev + (1 - a) * data_loss_new

        rec = {"loss": float(loss_new),
               "data_loss": float(data_loss_new),
               "lr": float(alpha)}
        for a in alphas:
            rec[f"ema_{a}"] = float(ema_state[a])
        if X_val is not None:
            vloss = loss_only(X_val, y_val, w, b, lam)
            vpred = np.where(X_val @ w + b >= 0, 1, -1)
            rec["val_loss"] = float(vloss)
            rec["val_acc"] = float(np.mean(vpred == y_val))
        history.append(rec)

        if verbose and (ep == 1 or ep % print_every == 0 or ep == epochs):
            line = (f"[steepest] ep {ep:4d}/{epochs} | alpha={alpha:.5f}"
                    f" | train={loss_new:.4f} | data={data_loss_new:.4f}")
            for a in alphas:
                line += f" | ema_{a}={rec[f'ema_{a}']:.4f}"
            if X_val is not None:
                line += (f" | val={rec['val_loss']:.4f}"
                         f" | acc={rec['val_acc']:.4f}")
            log(line)

        if early_stop_delta is not None and X_val is not None:
            if prev_val is not None and abs(prev_val - rec["val_loss"]) < early_stop_delta:
                n_stale += 1
            else:
                n_stale = 0
            prev_val = rec["val_loss"]
            if n_stale >= early_stop_patience:
                if verbose:
                    log(f"[steepest] early stop на ep {ep}: "
                        f"|Δval| < {early_stop_delta:.2e} "
                        f"в течение {early_stop_patience} эпох")
                break

    return w, b, history
