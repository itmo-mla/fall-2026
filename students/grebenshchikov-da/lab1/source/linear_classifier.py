from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np


def add_bias(X: np.ndarray) -> np.ndarray:
    """Добавляет единицу для свободного коэффициента."""
    return np.column_stack([X, np.ones(X.shape[0])])


def margins(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Положительный отступ означает правильный ответ модели."""
    return y * (add_bias(X) @ weights)


def quadratic_loss(margin: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    return (1.0 - margin) ** 2


def regularization_mask(size: int) -> np.ndarray:
    """Свободный коэффициент не регуляризируется."""
    mask = np.ones(size)
    mask[-1] = 0.0
    return mask


def empirical_risk(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    l2: float,
) -> float:
    """Возвращает средний loss по выборке вместе со штрафом L2."""
    data_loss = np.mean(quadratic_loss(margins(X, y, weights)))
    penalty = 0.5 * l2 * np.sum((weights * regularization_mask(len(weights))) ** 2)
    return float(data_loss + penalty)


def object_gradient(
    x_with_bias: np.ndarray,
    y: int,
    weights: np.ndarray,
    l2: float,
) -> np.ndarray:
    """Вычисляет градиент по одному объекту для шага SGD."""
    margin = y * np.dot(x_with_bias, weights)
    loss_gradient = -2.0 * (1.0 - margin) * y * x_with_bias
    return loss_gradient + l2 * weights * regularization_mask(len(weights))


def stochastic_steepest_step(
    x_with_bias: np.ndarray,
    gradient: np.ndarray,
    l2: float,
) -> float:
    """Находит лучший шаг для текущего объекта и квадратичного loss"""
    mask = regularization_mask(len(gradient))
    denominator = (
        2.0 * np.dot(x_with_bias, gradient) ** 2
        + l2 * np.sum((mask * gradient) ** 2)
    )
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(gradient, gradient) / denominator)


def correlation_initialization(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Начальный вес пропорционален связи признака с ответом."""
    denominator = np.sum(X**2, axis=0)
    denominator[denominator == 0] = 1.0
    feature_weights = (X.T @ y) / denominator
    bias = float(np.mean(y))
    return np.append(feature_weights, bias)


def random_initialization(feature_count: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(loc=0.0, scale=0.05, size=feature_count + 1)


@dataclass
class LinearClassifier:
    l2: float = 0.01
    weights: Optional[np.ndarray] = None
    initial_weights: Optional[np.ndarray] = None
    multistart_risks: List[float] = field(default_factory=list)
    best_start: Optional[int] = None
    history: Dict[str, List[float]] = field(
        default_factory=lambda: {"risk": [], "quality": []}
    )

    def fit_sgd(
        self,
        X: np.ndarray,
        y: np.ndarray,
        initial_weights: np.ndarray,
        epochs: int = 80,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        quality_rate: float = 0.02,
        presentation: str = "random",
        step_mode: str = "constant",
        random_state: int = 42,
    ) -> "LinearClassifier":
        """Обучает модель методом SGD с momentum и рекуррентной оценкой Q."""
        if step_mode not in {"constant", "steepest"}:
            raise ValueError("step_mode должен быть constant или steepest")
        if step_mode == "steepest" and momentum != 0.0:
            raise ValueError("Скорейший шаг используется без momentum")

        rng = np.random.default_rng(random_state)
        X_bias = add_bias(X)
        self.weights = initial_weights.astype(float).copy()
        self.initial_weights = self.weights.copy()
        self.history = {"risk": [], "quality": []}
        velocity = np.zeros_like(self.weights)
        quality = float(np.mean(quadratic_loss(margins(X, y, self.weights))))
        self.history["risk"].append(empirical_risk(X, y, self.weights, self.l2))
        self.history["quality"].append(quality)

        for _ in range(epochs):
            # При порядке margin сначала идут объекты около границы классов.
            if presentation == "margin":
                order = np.argsort(np.abs(y * (X_bias @ self.weights)))
            else:
                order = rng.permutation(len(X))

            for index in order:
                gradient = object_gradient(
                    X_bias[index],
                    int(y[index]),
                    self.weights,
                    self.l2,
                )
                if step_mode == "steepest":
                    # Шаг минимизирует loss текущего объекта вдоль антиградиента
                    step = stochastic_steepest_step(
                        X_bias[index],
                        gradient,
                        self.l2,
                    )
                    self.weights -= step * gradient
                else:
                    # Momentum сохраняет часть предыдущего направления движения.
                    velocity = momentum * velocity + gradient
                    self.weights -= learning_rate * velocity

                current_margin = y[index] * np.dot(X_bias[index], self.weights)
                current_loss = float(quadratic_loss(current_margin))
                # Q обновляется по одному объекту без пересчёта всей выборки.
                quality = (1.0 - quality_rate) * quality + quality_rate * current_loss

            self.history["risk"].append(empirical_risk(X, y, self.weights, self.l2))
            self.history["quality"].append(quality)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Модель ещё не обучена")
        return add_bias(X) @ self.weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.where(self.decision_function(X) >= 0, 1, -1)


def fit_multistart(
    X: np.ndarray,
    y: np.ndarray,
    starts: int,
    presentation: str,
    l2: float,
    random_state: int,
    **fit_arguments: object,
) -> LinearClassifier:
    """Запускает SGD с разными весами и оставляет минимальный train-риск."""
    rng = np.random.default_rng(random_state)
    best_model: Optional[LinearClassifier] = None
    best_risk = np.inf
    best_start: Optional[int] = None
    risks: List[float] = []

    for start in range(starts):
        initial = random_initialization(X.shape[1], rng)
        model = LinearClassifier(l2=l2).fit_sgd(
            X,
            y,
            initial_weights=initial,
            presentation=presentation,
            random_state=random_state + start,
            **fit_arguments,
        )
        risk = empirical_risk(X, y, model.weights, l2)
        risks.append(risk)
        if risk < best_risk:
            best_risk = risk
            best_model = model
            best_start = start + 1

    if best_model is None:
        raise RuntimeError("Не удалось обучить модель")
    best_model.multistart_risks = risks
    best_model.best_start = best_start
    return best_model
