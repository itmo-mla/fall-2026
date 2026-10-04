"""Линейный классификатор с квадратичной функцией потерь.

Модель: a(x) = sign(<w, x>), метки y ∈ {-1, +1}.
Отступ: M(x, y) = y <w, x>.
Потери: L(M) = (1 - M)^2.
Регуляризация: (λ/2) ||w_без_bias||^2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def margins(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Отступ каждого объекта: M_i = y_i <w, x_i>."""
    return y * (X @ w)


def quadratic_loss(M: np.ndarray) -> np.ndarray:
    return (1.0 - M) ** 2


def loss_gradient(x: np.ndarray, y: float, w: np.ndarray) -> np.ndarray:
    """Градиент L(M) = (1 - y <w, x>)^2 по вектору весов.

    dL/dw = -2 (1 - M) y x
    """
    m = float(y * np.dot(w, x))
    return -2.0 * (1.0 - m) * y * x


def l2_gradient(w: np.ndarray, l2: float) -> np.ndarray:
    """Градиент (λ/2)||w||^2 без свободного члена w[0]."""
    g = l2 * w.copy()
    g[0] = 0.0
    return g


def weight_norm(w: np.ndarray) -> float:
    """Евклидова норма весов без свободного члена."""
    return float(np.linalg.norm(w[1:]))


def epochs_until_risk(history: list[float], threshold: float = 0.5) -> int:
    """Первая эпоха, где эмпирический риск опустился не выше порога."""
    for epoch, value in enumerate(history):
        if value <= threshold:
            return epoch
    return len(history) - 1


def empirical_risk(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 0.0) -> float:
    M = margins(X, y, w)
    reg = 0.5 * l2 * float(np.dot(w[1:], w[1:]))
    return float(np.mean(quadratic_loss(M)) + reg)


def full_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 0.0) -> np.ndarray:
    """Полный градиент среднего квадратичного риска с L2."""
    M = margins(X, y, w)
    residual = 1.0 - M
    data_grad = -(2.0 / len(y)) * (X.T @ (y * residual))
    return data_grad + l2_gradient(w, l2)


def steepest_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, direction: np.ndarray, l2: float) -> float:
    """Оптимальный шаг η при обновлении w ← w - η g для квадратичного риска.

    Q квадратичен, поэтому минимум вдоль направления находится точно:
    η = ||g||^2 / (g^T H g), где H = (2/n) X^T X + λ I без свободного члена.
    """
    g = direction
    xg = X @ g
    den = (2.0 / len(y)) * float(np.dot(xg, xg)) + l2 * float(np.dot(g[1:], g[1:]))
    if den <= 1e-18:
        return 0.0
    return float(np.dot(g, g) / den)


def init_correlation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Старт по формуле лекции: w_j = <y, f_j> / <f_j, f_j>.

    Для столбца единиц это среднее меток. Оценка оптимальна при квадратичной
    потере и некоррелированных признаках.
    """
    w = np.zeros(X.shape[1], dtype=np.float64)
    for j in range(X.shape[1]):
        feature = X[:, j]
        denom = float(np.dot(feature, feature))
        if denom < 1e-12:
            continue
        w[j] = float(np.dot(y.astype(np.float64), feature) / denom)
    return w


def init_random(n_weights: int, rng: np.random.Generator, n_features: int | None = None) -> np.ndarray:
    """Равномерно на (-1/(2n), 1/(2n)), n — число исходных признаков."""
    n = n_features if n_features is not None else max(n_weights - 1, 1)
    half = 1.0 / (2.0 * n)
    return rng.uniform(-half, half, size=n_weights)


@dataclass
class TrainResult:
    w: np.ndarray
    q_history: list[float] = field(default_factory=list)
    risk_history: list[float] = field(default_factory=list)
    n_updates: int = 0


class LinearSGDClassifier:
    """SGD с инерцией, L2 и квадратичной функцией потерь."""

    def __init__(
        self,
        l2: float = 1e-3,
        lr: float = 0.05,
        momentum: float = 0.9,
        n_epochs: int = 50,
        smooth: float = 0.1,
        sampling: str = "random",
        step: str = "constant",
        random_state: int = 42,
    ) -> None:
        if sampling not in {"random", "margin"}:
            raise ValueError("sampling must be 'random' or 'margin'")
        if step not in {"constant", "steepest"}:
            raise ValueError("step must be 'constant' or 'steepest'")
        self.l2 = l2
        self.lr = lr
        self.momentum = momentum
        self.n_epochs = n_epochs
        self.smooth = smooth
        self.sampling = sampling
        self.step = step
        self.random_state = random_state
        self.w_: np.ndarray | None = None
        self.q_history_: list[float] = []
        self.risk_history_: list[float] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("Модель не обучена")
        return np.sign(X @ self.w_)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("Модель не обучена")
        return X @ self.w_

    def _sample_index(self, M: np.ndarray, rng: np.random.Generator) -> int:
        n = len(M)
        if self.sampling == "random":
            return int(rng.integers(0, n))
        # Чаще пограничные (|M| малы). Явные выбросы M << 0 почти не берём.
        weights = 1.0 / (np.abs(M) + 0.15)
        weights = np.where(M < -2.0, weights * 0.05, weights)
        weights = weights / weights.sum()
        return int(rng.choice(n, p=weights))

    def fit(self, X: np.ndarray, y: np.ndarray, w0: np.ndarray | None = None) -> TrainResult:
        rng = np.random.default_rng(self.random_state)
        w = (
            init_random(X.shape[1], rng, n_features=max(X.shape[1] - 1, 1))
            if w0 is None
            else w0.astype(np.float64).copy()
        )
        velocity = np.zeros_like(w)
        M = margins(X, y, w)
        q_hat = float(np.mean(quadratic_loss(M)))
        q_history = [q_hat]
        risk_history = [empirical_risk(X, y, w, self.l2)]
        n_updates = 0
        n = len(y)

        if self.step == "steepest":
            for _ in range(self.n_epochs):
                g = full_gradient(X, y, w, self.l2)
                eta = steepest_step(X, y, w, g, self.l2)
                w = w - eta * g
                n_updates += 1
                M = margins(X, y, w)
                q_hat = float(np.mean(quadratic_loss(M)))
                q_history.append(q_hat)
                risk_history.append(empirical_risk(X, y, w, self.l2))
        else:
            for _ in range(self.n_epochs):
                M = margins(X, y, w)
                for _ in range(n):
                    i = self._sample_index(M, rng)
                    x_i, y_i = X[i], float(y[i])
                    g = loss_gradient(x_i, y_i, w) / n + l2_gradient(w, self.l2)
                    velocity = self.momentum * velocity + self.lr * g
                    w = w - velocity
                    n_updates += 1
                    m_i = float(y_i * np.dot(w, x_i))
                    q_hat = (1.0 - self.smooth) * q_hat + self.smooth * (1.0 - m_i) ** 2
                    M[i] = m_i
                q_history.append(q_hat)
                risk_history.append(empirical_risk(X, y, w, self.l2))

        self.w_ = w
        self.q_history_ = q_history
        self.risk_history_ = risk_history
        return TrainResult(w=w, q_history=q_history, risk_history=risk_history, n_updates=n_updates)


def multistart(
    X: np.ndarray,
    y: np.ndarray,
    n_starts: int = 8,
    random_state: int = 42,
    **fit_kwargs,
) -> tuple[LinearSGDClassifier, list[float]]:
    """Несколько случайных стартов, выбирается модель с наименьшим риском на train."""
    rng = np.random.default_rng(random_state)
    best_model: LinearSGDClassifier | None = None
    best_risk = np.inf
    risks: list[float] = []

    for k in range(n_starts):
        model = LinearSGDClassifier(random_state=random_state + k, **fit_kwargs)
        w0 = init_random(X.shape[1], rng, n_features=max(X.shape[1] - 1, 1))
        model.fit(X, y, w0=w0)
        risk = empirical_risk(X, y, model.w_, model.l2)
        risks.append(risk)
        if risk < best_risk:
            best_risk = risk
            best_model = model

    assert best_model is not None
    return best_model, risks
