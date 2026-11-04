"""Отступ, квадратичные потери, градиент, EMA, логгер (numpy + pandas)."""
import numpy as np
from pathlib import Path


_LOG_BUFFER: list = []


def log(line=""):
    print(line, flush=True)
    _LOG_BUFFER.append(str(line))


def clear_log():
    _LOG_BUFFER.clear()


def dump_log(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_LOG_BUFFER), encoding="utf-8")
    print(f"\nОтчёт сохранён: {path}", flush=True)



# Отступ, потери, градиент
def margin(X, y, w, b):
    """M_i = y_i * (w^T x_i + b), y_i in {-1, +1}."""
    return y * (X @ w + b)


def margin_stats(X, y, w, b):
    m = margin(X, y, w, b)
    return dict(
        mean=float(m.mean()),
        std=float(m.std()),
        min=float(m.min()),
        max=float(m.max()),
        frac_negative=float(np.mean(m < 0)),
        frac_below_one=float(np.mean(m < 1)),
    )


def loss_grad(X, y, w, b, lam=0.0):
    z = X @ w + b
    r = z - y
    N = len(y)
    loss = 0.5 * float(r @ r) / N + 0.5 * lam * float(w @ w)
    gw = (X.T @ r) / N + lam * w
    gb = float(r.sum()) / N
    return loss, gw, gb


def loss_grad_sample(x, y_i, w, b, lam=0.0):
    x = np.asarray(x).ravel()
    z = float(x @ w + b)
    r = z - y_i
    loss = 0.5 * r * r + 0.5 * lam * float(w @ w)
    gw = r * x + lam * w
    gb = r
    return loss, gw, gb


def loss_only(X, y, w, b, lam=0.0):
    return loss_grad(X, y, w, b, lam)[0]


def check_gradient(X, y, w, b, lam=0.0, n=200, seed=0, eps=1e-6):
    """Проверка градиента центральной конечной разностью."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=min(n, len(y)), replace=False)
    Xs, ys = X[idx], y[idx]

    _, gw, gb = loss_grad(Xs, ys, w, b, lam)
    g_anal = np.r_[gw, gb]

    p = np.r_[w, b].astype(np.float64)
    g_num = np.zeros_like(p)
    for i in range(len(p)):
        p1, p2 = p.copy(), p.copy()
        p1[i] += eps
        p2[i] -= eps
        g_num[i] = (loss_only(Xs, ys, p1[:-1], p1[-1], lam)
                    - loss_only(Xs, ys, p2[:-1], p2[-1], lam)) / (2 * eps)

    denom = np.linalg.norm(g_anal) + np.linalg.norm(g_num) + 1e-12
    err = float(np.linalg.norm(g_anal - g_num) / denom)
    log(f"[grad-check] относительная ошибка={err:.2e}")
    return err


class EMA:
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        self.value = x if self.value is None \
            else self.alpha * self.value + (1 - self.alpha) * x
        return self.value

    def reset(self):
        self.value = None
