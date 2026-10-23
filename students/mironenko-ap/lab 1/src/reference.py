"""Эталонная модель sklearn: SGDClassifier."""
import numpy as np
from sklearn.linear_model import SGDClassifier

from .core import log, loss_only

def train_logreg(X, y, C=1.0):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
    model.fit(X, y)
    w = model.coef_.ravel().astype(np.float64)
    b = float(np.asarray(model.intercept_).ravel()[0])
    log(f"[logreg] C={C}, ||w||={np.linalg.norm(w):.4f}")
    return w, b

def _build_sgd(w0, b0, loss, penalty, alpha, lr, mu, lr_decay,
               average, seed):
    """Собирает SGDClassifier с начальными параметрами."""
    if lr_decay > 0:
        learning_rate = "invscaling"
        power_t = float(lr_decay)
    else:
        learning_rate = "constant"
        power_t = 0.5

    clf = SGDClassifier(
        loss=loss,
        penalty=penalty,
        alpha=alpha,
        fit_intercept=True,
        max_iter=1,          # одна эпоха за один partial_fit
        tol=None,            # без early stopping
        shuffle=True,
        learning_rate=learning_rate,
        eta0=lr,
        power_t=power_t,
        early_stopping=False,
        average=average,
        random_state=seed,
        warm_start=True,
        verbose=0,
    )
    # Ручная установка начальных параметров.
    # Форма для бинарной задачи: coef_ (1, D), intercept_ (1,).
    clf.coef_ = np.asarray(w0, dtype=np.float64).reshape(1, -1)
    clf.intercept_ = np.array([float(b0)])
    clf.classes_ = np.array([-1.0, 1.0])   # порядок: -1 → 0, +1 → 1
    return clf


def sgd_classifier_train(
    X, y, w0, b0,
    # потеря и регуляризация
    loss="squared_error",       # "squared_error" | "log_loss" | ...
    lam=1e-3, use_l2=True,
    # оптимизация
    lr=0.01, mu=0.9, epochs=50,
    # совместимость сигнатуры (ordering игнорируется)
    ordering="random", rng=None,
    # валидация и логирование
    X_val=None, y_val=None,
    print_every=10, verbose=True,
    # затухание lr: приближённое соответствие invscaling
    lr_decay=0.0,
    ema_alphas=(0.9, 0.99, 0.999),
    # специфика sklearn
    average=False,
):
    rng = rng or np.random.default_rng(0)
    alphas = list(ema_alphas)
    ema_state = {a: None for a in alphas}

    # L2: penalty=None → без регуляризации, alpha=0
    penalty = "l2" if use_l2 else None
    alpha = float(lam) if use_l2 else 0.0
    lam_eff = float(lam) if use_l2 else 0.0

    if verbose:
        log(f"\n[sklearn-SGD] старт | loss={loss} | penalty={penalty} "
            f"| alpha={alpha} | eta0={lr} | momentum={mu} "
            f"| epochs={epochs} | lr_decay={lr_decay}")

    clf = _build_sgd(
        w0, b0, loss, penalty, alpha, lr, mu, lr_decay, average,
        int(rng.integers(0, 2**31 - 1)),
    )

    history = []

    for ep in range(1, epochs + 1):
        # Первый partial_fit обязан получить classes, дальше — не нужно.
        if ep == 1:
            clf.partial_fit(X, y, classes=np.array([-1.0, 1.0]))
        else:
            clf.partial_fit(X, y)

        w = clf.coef_.ravel().astype(np.float64)
        b = float(clf.intercept_.ravel()[0])

        # Полная цель (с L2, если включена) и data loss (всегда без L2).
        full_loss = loss_only(X, y, w, b, lam_eff)
        data_loss = loss_only(X, y, w, b, 0.0)

        # EMA — по data loss, как в sgd_momentum / steepest_descent.
        for a in alphas:
            prev = ema_state[a]
            ema_state[a] = data_loss if prev is None \
                else a * prev + (1 - a) * data_loss

        rec = {"loss": float(full_loss),
               "data_loss": float(data_loss),
               "lr": float(lr)}
        for a in alphas:
            rec[f"ema_{a}"] = float(ema_state[a])
        if X_val is not None:
            vloss = loss_only(X_val, y_val, w, b, lam_eff)
            vpred = np.where(X_val @ w + b >= 0, 1, -1)
            rec["val_loss"] = float(vloss)
            rec["val_acc"] = float(np.mean(vpred == y_val))
        history.append(rec)

        if verbose and (ep == 1 or ep % print_every == 0 or ep == epochs):
            line = (f"[sklearn-SGD] ep {ep:4d}/{epochs}"
                    f" | train={full_loss:.4f} | data={data_loss:.4f}")
            for a in alphas:
                line += f" | ema_{a}={rec[f'ema_{a}']:.4f}"
            if X_val is not None:
                line += (f" | val={rec['val_loss']:.4f}"
                         f" | acc={rec['val_acc']:.4f}")
            log(line)

    w = clf.coef_.ravel().astype(np.float64)
    b = float(clf.intercept_.ravel()[0])
    return w, b, history