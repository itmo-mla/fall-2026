"""Базовые ядра и правила их синтеза.

Все ядра принимают две выборки X и Z и возвращают матрицу K(X, Z).
Арифметические операции над ядрами выполняются поэлементно.
"""

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np


class Kernel:
    """Общий интерфейс и операции, сохраняющие PSD-свойство ядер.

    Вызов принимает выборки (n, d), (m, d) и возвращает матрицу (n, m).
    Числа в выражениях трактуются как неотрицательные константные ядра.

    Примеры:
        2 * LinearKernel() + RBFKernel() — взвешенная сумма;
        LinearKernel() * RBFKernel() — произведение;
        (LinearKernel() + 1) ** 3 — полиномиальное ядро.
    """

    def __call__(self, X, Z) -> np.ndarray:
        """Вычислить матрицу попарных значений ядра для объектов X и Z."""
        raise NotImplementedError

    def __add__(self, other):
        """Сложить два ядра или прибавить неотрицательную константу."""
        right = _as_kernel(other)
        if right is NotImplemented:
            return NotImplemented
        return SumKernel(self, right)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        """Перемножить два ядра или умножить на неотрицательный коэффициент."""
        right = _as_kernel(other)
        if right is NotImplemented:
            return NotImplemented
        return ProductKernel(self, right)

    def __rmul__(self, other):
        return self * other

    def __pow__(self, degree):
        """Возвести ядро в целую неотрицательную степень."""
        return PowerKernel(self, degree)

    def transform(self, function: Callable) -> "TransformedKernel":
        """K(f(X), f(Z)); function преобразует целую матрицу объектов.

        Число объектов и их порядок сохраняются, размерность признаков
        может меняться. Одна и та же функция применяется к обеим выборкам.
        """
        return TransformedKernel(self, function)

    def exp(self) -> "ExponentialKernel":
        """Поэлементная exp(K), сохраняющая положительную полуопределённость."""
        return ExponentialKernel(self)

    def weighted(self, function: Callable) -> "WeightedKernel":
        """g(x) K(x, z) g(z); function возвращает один вес на объект."""
        return WeightedKernel(self, function)


def _as_kernel(value):
    """Преобразовать число в ядро; прочие типы передать механизму операторов Python."""
    if isinstance(value, Kernel):
        return value
    if isinstance(value, Real):
        return ConstantKernel(value)
    return NotImplemented


def _require_kernel(value):
    if not isinstance(value, Kernel):
        raise TypeError("expected a Kernel instance")


def _validate_samples(samples) -> np.ndarray:
    """Привести выборку к вещественной матрице без NaN и бесконечностей."""
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        raise ValueError("samples must be a two-dimensional matrix")
    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must be finite")
    return samples


@dataclass(frozen=True)
class ConstantKernel(Kernel):
    """K(x, z) = value; допустимы только неотрицательные константы."""

    value: float = 1.0

    def __post_init__(self):
        if (
            not isinstance(self.value, Real)
            or not np.isfinite(self.value)
            or self.value < 0
        ):
            raise ValueError("constant must be finite and nonnegative")

    def __call__(self, X, Z):
        X, Z = _validate_samples(X), _validate_samples(Z)
        return np.full((len(X), len(Z)), self.value, dtype=float)


@dataclass(frozen=True)
class LinearKernel(Kernel):
    """K(x, z) = <x, z>."""

    def __call__(self, X, Z):
        X, Z = _validate_samples(X), _validate_samples(Z)
        return X @ Z.T


@dataclass(frozen=True)
class RBFKernel(Kernel):
    """K(x, z) = exp(-gamma * ||x - z||²), gamma > 0."""

    gamma: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.gamma) or self.gamma <= 0:
            raise ValueError("gamma must be finite and positive")

    def __call__(self, X, Z):
        X, Z = _validate_samples(X), _validate_samples(Z)
        X_squared_norms = np.sum(X**2, axis=1)[:, None]
        Z_squared_norms = np.sum(Z**2, axis=1)[None, :]
        squared_distances = X_squared_norms + Z_squared_norms - 2 * X @ Z.T
        # Погрешность вычислений может дать расстояние чуть меньше нуля.
        squared_distances = np.maximum(squared_distances, 0)
        return np.exp(-self.gamma * squared_distances)


@dataclass(frozen=True)
class PolynomialKernel(Kernel):
    """K(x, z) = (<x, z> + coef0)^degree, coef0 >= 0, degree >= 1."""

    degree: int = 2
    coef0: float = 1.0

    def __post_init__(self):
        if not isinstance(self.degree, int) or self.degree < 1:
            raise ValueError("degree must be a positive integer")
        if not np.isfinite(self.coef0) or self.coef0 < 0:
            raise ValueError("coef0 must be finite and nonnegative")

    def __call__(self, X, Z):
        X, Z = _validate_samples(X), _validate_samples(Z)
        dot_products = X @ Z.T
        return (dot_products + self.coef0) ** self.degree


@dataclass(frozen=True)
class SumKernel(Kernel):
    """K(x, z) = K_left(x, z) + K_right(x, z)."""

    left: Kernel
    right: Kernel

    def __post_init__(self):
        _require_kernel(self.left)
        _require_kernel(self.right)

    def __call__(self, X, Z):
        return self.left(X, Z) + self.right(X, Z)


@dataclass(frozen=True)
class ProductKernel(Kernel):
    """K(x, z) = K_left(x, z) * K_right(x, z)."""

    left: Kernel
    right: Kernel

    def __post_init__(self):
        _require_kernel(self.left)
        _require_kernel(self.right)

    def __call__(self, X, Z):
        # Именно поэлементное произведение, а не матричное @.
        return self.left(X, Z) * self.right(X, Z)


@dataclass(frozen=True)
class PowerKernel(Kernel):
    """K_new(x, z) = K(x, z)^degree для целого degree >= 0."""

    kernel: Kernel
    degree: int

    def __post_init__(self):
        _require_kernel(self.kernel)
        if not isinstance(self.degree, Integral) or self.degree < 0:
            raise ValueError("degree must be a nonnegative integer")

    def __call__(self, X, Z):
        return self.kernel(X, Z) ** self.degree


@dataclass(frozen=True)
class ExponentialKernel(Kernel):
    """K_new(x, z) = exp(K(x, z))."""

    kernel: Kernel

    def __post_init__(self):
        _require_kernel(self.kernel)

    def __call__(self, X, Z):
        return np.exp(self.kernel(X, Z))


@dataclass(frozen=True)
class TransformedKernel(Kernel):
    """K_new(x, z) = K(f(x), f(z)) для фиксированного преобразования f."""

    kernel: Kernel
    function: Callable

    def __post_init__(self):
        _require_kernel(self.kernel)
        if not callable(self.function):
            raise TypeError("function must be callable")

    def _transform(self, X):
        X = _validate_samples(X)
        transformed = _validate_samples(self.function(X))
        if len(transformed) != len(X):
            raise ValueError("transform must preserve the number of samples")
        return transformed

    def __call__(self, X, Z):
        transformed_X = self._transform(X)
        transformed_Z = self._transform(Z)
        return self.kernel(transformed_X, transformed_Z)


@dataclass(frozen=True)
class WeightedKernel(Kernel):
    """K_new(x, z) = g(x) K(x, z) g(z); знак весов g не ограничен."""

    kernel: Kernel
    function: Callable

    def __post_init__(self):
        _require_kernel(self.kernel)
        if not callable(self.function):
            raise TypeError("function must be callable")

    def _weights(self, X):
        weights = np.asarray(self.function(X), dtype=float)
        if weights.shape != (len(X),) or not np.all(np.isfinite(weights)):
            raise ValueError("weight function must return one finite weight per sample")
        return weights

    def __call__(self, X, Z):
        X, Z = _validate_samples(X), _validate_samples(Z)
        X_weights = self._weights(X)[:, None]
        Z_weights = self._weights(Z)[None, :]
        kernel_matrix = self.kernel(X, Z)
        return X_weights * kernel_matrix * Z_weights
