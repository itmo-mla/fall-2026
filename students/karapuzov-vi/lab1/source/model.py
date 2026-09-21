import numpy as np
from data import prepare_data, PLOTS_DIR
import matplotlib.pyplot as plt

def add_bias(X):
    X = np.asarray(X, dtype=float)
    return np.c_[np.ones(X.shape[0]), X]

def margins(X, y, w: np.ndarray):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    return y * (X @ w)


def init_random(n_features, seed: int = 42):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.01, size=n_features)


def init_correlation(X, y) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.zeros(X.shape[1])
    for j in range(1, X.shape[1]):
        w[j] = np.corrcoef(X[:, j], y)[0, 1]
    return np.nan_to_num(w)


def error_rate(m: np.ndarray) -> float:
    return float((m < 0).mean())


def predict(X, w) -> np.ndarray:
    scores = np.asarray(X, dtype=float) @ np.asarray(w, dtype=float)
    pred = np.sign(scores)
    pred[pred == 0] = -1
    return pred.astype(int)


def classify_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    acc = float((y_true == y_pred).mean())
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == -1)))
    fn = int(np.sum((y_pred == -1) & (y_true == 1)))
    tn = int(np.sum((y_pred == -1) & (y_true == -1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "error": 1.0 - acc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def train_multistart(X, y, n_starts: int = 5, **sgd_kwargs):
    best_w, best_q, best_hist = None, np.inf, None
    for s in range(n_starts):
        w0 = init_random(X.shape[1], seed=s)
        w, hist = sgd_momentum(X, y, w0, seed=1000 + s, **sgd_kwargs)
        q = empirical_risk(X, y, w)
        if q < best_q:
            best_w, best_q, best_hist = w, q, hist
    return best_w, best_hist, best_q


def plot_margins(m: np.ndarray, title: str, filename: str) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(m, bins=50, color="skyblue", edgecolor="black", alpha=0.8)
    plt.axvline(0, color="red", linestyle="dashed", linewidth=2, label="Граница (margin = 0)")
    plt.title(title, fontsize=14)
    plt.xlabel("Значение отступа M")
    plt.ylabel("Количество объектов")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.savefig(PLOTS_DIR / filename, dpi=120)
    plt.close()

def quadratic_loss(M):
    M = np.asarray(M, dtype=float)
    return (1 - M) ** 2


def loss_gradient(x, y, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    y = float(y)
    M = y * (x @ w)
    return -2 * (1 - M) * y * x


def loss_gradient_batch(X, y, w):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    grad = np.zeros_like(w, dtype=float)
    for i in range(len(X)):
        grad += loss_gradient(X[i], y[i], w)
    return grad / len(X)


def empirical_risk(X, y, w):
    return float(quadratic_loss(margins(X, y, w)).mean())


def numerical_gradient(X, y, w, eps: float = 1e-5):
    w = np.asarray(w, dtype=float)
    grad = np.zeros_like(w)
    for j in range(len(w)):
        step = np.zeros_like(w)
        step[j] = eps
        q_plus = empirical_risk(X, y, w + step)
        q_minus = empirical_risk(X, y, w - step)
        grad[j] = (q_plus - q_minus) / (2 * eps)
    return grad


def sgd_momentum(X, y, w0, h=0.001, tau=0.9, lambda_q=None, n_steps=8000, seed=0, l2=0.0, steepest=False,sampling="uniform"):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w0, dtype=float).copy()
    w_prev = w.copy()
    rng = np.random.default_rng(seed)

    if lambda_q is None:
        lambda_q = 1.0 / len(X)

    Q = float(quadratic_loss(y[0] * (X[0] @ w)))
    history = [Q]

    for step in range(n_steps):
        if sampling == "margin":
            M_all = y * (X @ w)
            scores = 1.0 / (np.abs(M_all) + 1e-6)
            p = scores / scores.sum()
            i = int(rng.choice(len(X), p=p))
        else:
            i = int(rng.integers(0, len(X)))
        x_i = X[i]
        y_i = y[i]

        M = y_i * (x_i @ w)
        L = float(quadratic_loss(M))
        g = loss_gradient(x_i, y_i, w)
        g = g.copy()
        g[1:] = g[1:] + l2 * w[1:]
        delta = w - w_prev
        w_prev = w.copy()

        tau_step = 0.0 if steepest else tau
        h_step = (
            steepest_step_size(x_i, y_i, w, g, l2=l2, tau=tau_step, delta=delta)
            if steepest
            else h
        )
        w_next = w - h_step * g + tau_step * delta
        if np.all(np.isfinite(w_next)):
            w = w_next

        Q = lambda_q * L + (1.0 - lambda_q) * Q
        if (step + 1) % 50 == 0:
            history.append(Q)

    return w, history


def steepest_step_size(x, y, w, g, l2=0.0, tau=0.0, delta=None, h_grid=None, h_max=0.05):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    y = float(y)
    if delta is None:
        delta = np.zeros_like(w)
    if h_grid is None:
        h_grid = np.logspace(-6, np.log10(h_max), 20)

    best_h = h_grid[0]
    best_val = np.inf
    for h_try in h_grid:
        w_try = w - h_try * g + tau * delta
        if not np.all(np.isfinite(w_try)):
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            val = (1 - y * (x @ w_try)) ** 2 + 0.5 * l2 * np.sum(w_try[1:] ** 2)
        if np.isfinite(val) and val < best_val:
            best_val = val
            best_h = h_try
    return best_h

def plot_q_history(history, title: str, filename: str) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(history)) * 50, history, label=r"рекуррентная оценка $\tilde Q$")
    plt.title(title)
    plt.xlabel("Итерация SGD")
    plt.ylabel("Оценка эмпирического риска")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=120)
    plt.close()

if __name__ == "__main__":
    data = prepare_data()
    X_train = add_bias(data["X_train"])
    y_train = data["y_train"]

    w_random = init_random(X_train.shape[1])
    w_corr = init_correlation(X_train, y_train)

    m_random = margins(X_train, y_train, w_random)
    m_corr = margins(X_train, y_train, w_corr)

    print("случайные веса:     ошибка =", round(error_rate(m_random) * 100, 2), "%  среднее M =", round(m_random.mean(), 3))
    print("веса по корреляции: ошибка =", round(error_rate(m_corr) * 100, 2), "%  среднее M =", round(m_corr.mean(), 3))
    #print("веса по корреляции:", dict(zip(["bias"] + data["feature_names"], w_corr.round(3))))

    plot_margins(m_random, "Отступы при случайных весах", "margins_random.png")
    plot_margins(m_corr, "Отступы при инициализации корреляцией", "margins_correlation.png")
    print("графики:", PLOTS_DIR / "margins_random.png", "и", PLOTS_DIR / "margins_correlation.png")

    w = w_corr
    q = empirical_risk(X_train, y_train, w)
    grad_analytic = loss_gradient_batch(X_train, y_train, w)
    grad_numeric = numerical_gradient(X_train, y_train, w)
    max_diff = np.max(np.abs(grad_analytic - grad_numeric))

    print("\n=== проверка градиента (веса из корреляции) ===")
    print("средний штраф Q =", round(q, 4))
    print("норма градиента  =", round(np.linalg.norm(grad_analytic), 4))
    print("max |аналитический - численный| =", max_diff)
    print("проверка", "OK" if max_diff < 1e-4 else "FAILED")

    error_before = error_rate(margins(X_train, y_train, w_corr))
    w_sgd, hist = sgd_momentum(X_train, y_train, w_corr)
    error_after = error_rate(margins(X_train, y_train, w_sgd))
    print("\n=== SGD с инерцией (старт из корреляции) ===")
    print("ошибка train до  =", round(error_before * 100, 2), "%")
    print("ошибка train после =", round(error_after * 100, 2), "%")
    print("Q в конце EMA     =", round(hist[-1], 4))
    plot_q_history(hist, "Рекуррентная оценка Q во время SGD", "sgd_q_history.png")
    print("график Q:", PLOTS_DIR / "sgd_q_history.png")

    print("\n=== L2: один и тот же SGD, разный λ ===")
    print("ошибка до обучения:", round(error_before * 100, 2), "%")
    for l2 in (0.0, 0.1, 1.0, 10.0):
        w_l2, _ = sgd_momentum(X_train, y_train, w_corr, l2=l2)
        err = error_rate(margins(X_train, y_train, w_l2))
        print(
            f"λ={l2:<4}  ошибка train={round(err * 100, 2):5.2f}%  "
            f"||w||={np.linalg.norm(w_l2):.4f}  ||w без bias||={np.linalg.norm(w_l2[1:]):.4f}"
        )

    print("\n=== константный h vs скорейший спуск ===")
    w_const, _ = sgd_momentum(X_train, y_train, w_corr, l2=0.1)
    w_steep, hist_steep = sgd_momentum(X_train, y_train, w_corr, steepest=True, l2=0.1)
    print(
        "const h:     ошибка =",
        round(error_rate(margins(X_train, y_train, w_const)) * 100, 2),
        "%  ||w|| =",
        round(np.linalg.norm(w_const), 4),
    )
    print(
        "steepest:    ошибка =",
        round(error_rate(margins(X_train, y_train, w_steep)) * 100, 2),
        "%  ||w|| =",
        round(np.linalg.norm(w_steep), 4),
    )
    plot_q_history(hist_steep, "Рекуррентная оценка Q (скорейший спуск)", "sgd_steepest_q.png")
    print("график Q steepest:", PLOTS_DIR / "sgd_steepest_q.png")

    print("\n=== предъявление: uniform vs |M| ===")
    w_uni, _ = sgd_momentum(X_train, y_train, w_corr, l2=0.1, sampling="uniform")
    w_mrg, _ = sgd_momentum(X_train, y_train, w_corr, l2=0.1, sampling="margin")
    m_uni = margins(X_train, y_train, w_uni)
    m_mrg = margins(X_train, y_train, w_mrg)
    print(
        "uniform:  ошибка =", round(error_rate(m_uni) * 100, 2),
        "%  среднее |M| =", round(np.abs(m_uni).mean(), 3),
        "  среднее M =", round(m_uni.mean(), 3),
    )
    print(
        "margin:   ошибка =", round(error_rate(m_mrg) * 100, 2),
        "%  среднее |M| =", round(np.abs(m_mrg).mean(), 3),
        "  среднее M =", round(m_mrg.mean(), 3),
    )
    plot_margins(m_uni, "Отступы после SGD (случайное предъявление)", "margins_after_uniform.png")
    plot_margins(m_mrg, "Отступы после SGD (предъявление по |M|)", "margins_after_margin.png")
    print("графики:", PLOTS_DIR / "margins_after_uniform.png", "и", PLOTS_DIR / "margins_after_margin.png")
