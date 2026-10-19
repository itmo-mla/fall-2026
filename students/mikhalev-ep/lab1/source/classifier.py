import numpy as np
import pandas as pd


# -------------------------------------------------------------
# Базовые математические операции
# -------------------------------------------------------------
def margin(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Вычисление отступа: M = y * <w, x>"""
    return y * (X @ w)


def loss(m: np.ndarray) -> np.ndarray:
    """Квадратичная функция потерь: (1 - M)^2"""
    return (1.0 - m) ** 2


def loss_gradient(x: np.ndarray, y: float, w: np.ndarray) -> np.ndarray:
    """Градиент квадратичных потерь: grad = -2 * y * x * (1 - m)"""
    m = y * np.dot(x, w)
    return -2.0 * y * x * (1.0 - m)


def mean_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Средняя квадратичная ошибка по выборке"""
    return float(np.mean(loss(margin(X, y, w))))


def predict_scores(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return X @ w


def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.where(predict_scores(X, w) >= 0, 1, -1)


# -------------------------------------------------------------
# Инициализация весов (Шаг 9.i и 9.ii)
# -------------------------------------------------------------
def correlation_init(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Инициализация через корреляцию признаков (Шаг 9.i): w_j = <y, f_j> / <f_j, f_j>"""
    numerator = X.T @ y
    denominator = np.sum(X**2, axis=0)
    denominator = np.where(denominator == 0, 1.0, denominator)
    return numerator / denominator


def random_init(n_features: int, rng: np.random.Generator) -> np.ndarray:
    """Случайная инициализация: w_j ~ U(-1/(2n), 1/(2n))"""
    bound = 1.0 / (2.0 * n_features)
    return rng.uniform(-bound, bound, size=n_features)


def initialize_q(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, rng: np.random.Generator, fraction: float = 0.10
) -> float:
    """Инициализация рекуррентной оценки Q на 10% подвыборки (Шаг 4)"""
    subset_size = max(1, int(fraction * len(X)))
    idx = rng.choice(len(X), size=subset_size, replace=False)
    return mean_loss(X[idx], y[idx], w)


# -------------------------------------------------------------
# Выбор объекта (Шаг 8)
# -------------------------------------------------------------
def choose_object(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, rng: np.random.Generator, sampling: str = "random", eps: float = 1e-8
) -> int:
    """Предъявление объектов: равномерное или по модулю отступа"""
    if sampling == "random":
        return int(rng.integers(len(X)))

    if sampling == "margin":
        m = margin(X, y, w)
        weights = 1.0 / (np.abs(m) + eps)
        probabilities = weights / weights.sum()
        return int(rng.choice(len(X), p=probabilities))

    raise ValueError("sampling должен быть 'random' или 'margin'")


# -------------------------------------------------------------
# Обучение SGD + Momentum + L2 (Шаги 3, 4, 5, 6, 8, 9)
# -------------------------------------------------------------
def fit_sgd(
    X: np.ndarray,
    y: np.ndarray,
    w_init: np.ndarray,
    n_iter: int = 5000,
    learning_rate: float = 0.001,
    gamma: float = 0.9,
    lambda_q: float = 0.01,
    tau: float = 0.001,
    sampling: str = "random",
    random_state: int = 42,
):
    rng = np.random.default_rng(random_state)
    w = np.asarray(w_init, dtype=float).copy()
    v = np.zeros_like(w)

    Q = initialize_q(X, y, w, rng)
    q_history = [Q]

    eval_every = max(1, n_iter // 200)
    train_loss_history = [(0, mean_loss(X, y, w))]

    for iteration in range(1, n_iter + 1):
        i = choose_object(X, y, w, rng, sampling)

        x_i = X[i]
        y_i = y[i]

        m_i = y_i * np.dot(x_i, w)
        loss_i = loss(m_i)

        # Шаг 3: Градиент квадратичной функции потерь
        grad = loss_gradient(x_i, y_i, w)

        # Шаг 5: Momentum (инерция)
        v = gamma * v + (1.0 - gamma) * grad

        # Шаг 6: L2-регуляризация (сжатие весов)
        w = (1.0 - learning_rate * tau) * w - learning_rate * v

        # Шаг 4: Рекуррентная оценка функционала качества
        Q = lambda_q * loss_i + (1.0 - lambda_q) * Q
        q_history.append(float(Q))

        if iteration % eval_every == 0 or iteration == n_iter:
            train_loss_history.append((iteration, mean_loss(X, y, w)))

    return {
        "w": w,
        "q_history": np.asarray(q_history),
        "train_loss_history": np.asarray(train_loss_history, dtype=float),
    }


# -------------------------------------------------------------
# Шаг 7: Скорейший градиентный спуск (аналитический шаг h*)
# -------------------------------------------------------------
def fit_steepest(
    X: np.ndarray,
    y: np.ndarray,
    w_init: np.ndarray,
    n_iter: int = 5000,
    lambda_q: float = 0.01,
    random_state: int = 42,
    eps: float = 1e-12,
):
    """Точный поиск шага для одного объекта, а не всей train-выборки.
    """
    rng = np.random.default_rng(random_state)
    w = np.asarray(w_init, dtype=float).copy()

    Q = initialize_q(X, y, w, rng)
    q_history = [Q]

    eval_every = max(1, n_iter // 200)
    train_loss_history = [(0, mean_loss(X, y, w))]

    for iteration in range(1, n_iter + 1):
        i = int(rng.integers(len(X)))

        x_i = X[i]
        y_i = y[i]

        m_i = y_i * np.dot(x_i, w)
        loss_i = loss(m_i)

        grad = loss_gradient(x_i, y_i, w)

        # Аналитический шаг для квадратичных потерь
        h_star = 1.0 / (2.0 * np.dot(x_i, x_i) + eps)
        w = w - h_star * grad

        Q = lambda_q * loss_i + (1.0 - lambda_q) * Q
        q_history.append(float(Q))

        if iteration % eval_every == 0 or iteration == n_iter:
            train_loss_history.append((iteration, mean_loss(X, y, w)))

    return {
        "w": w,
        "q_history": np.asarray(q_history),
        "train_loss_history": np.asarray(train_loss_history, dtype=float),
    }


# -------------------------------------------------------------
# Шаг 9.ii: Мультистарт
# -------------------------------------------------------------
def fit_multistart(
    X: np.ndarray,
    y: np.ndarray,
    n_starts: int = 10,
    random_state: int = 42,
    n_iter: int = 5000,
    learning_rate: float = 0.001,
    gamma: float = 0.9,
    lambda_q: float = 0.01,
    tau: float = 0.001,
):
    if n_starts < 1:
        raise ValueError("n_starts должен быть положительным")
    rng = np.random.default_rng(random_state)
    best_result = None
    best_loss = np.inf
    rows = []

    for start in range(1, n_starts + 1):
        w_init = random_init(X.shape[1], rng)
        seed = int(rng.integers(0, 2**32 - 1))

        result = fit_sgd(
            X, y,
            w_init=w_init,
            n_iter=n_iter,
            learning_rate=learning_rate,
            gamma=gamma,
            lambda_q=lambda_q,
            tau=tau,
            sampling="random",
            random_state=seed,
        )
        current_loss = mean_loss(X, y, result["w"])
        rows.append({"start": start, "train_loss": current_loss})

        if current_loss < best_loss:
            best_loss = current_loss
            best_result = result

    return best_result, pd.DataFrame(rows)
