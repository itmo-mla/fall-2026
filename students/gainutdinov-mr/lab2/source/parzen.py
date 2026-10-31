"""Интерфейс классификатора с окном Парзена."""

from collections.abc import Callable
from typing import Literal, Self

import numpy as np


class ParzenClassifier:
    """Классификация с фиксированной или переменной шириной окна.

    В режиме variable ширина равна расстоянию до (k + 1)-го соседа.
    Гауссовые веса получают все опорные объекты. При k=1 это по-прежнему
    классификатор Парзена. Алгоритмические методы пока являются пустышками.
    """

    # TODO: Обновить описание класса: методы реализованы, фразу про пустышки удалить.

    def __init__(
        self,
        *,
        window: Literal["variable", "fixed"] = "variable",
        k: int = 1,
        bandwidth: float | None = None,
        kernel: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.window = window
        self.k = k
        self.bandwidth = bandwidth
        self.kernel = kernel if kernel is not None else self.gaussian_kernel

        self.X_: np.ndarray | None = None
        self.y_: np.ndarray | None = None
        self.distances_matrix: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Запомнить опорную выборку, классы и вернуть self.

        X уже стандартизована снаружи. Проверить параметры окна:
        для variable требуется len(X) >= k + 1, для fixed требуется bandwidth > 0.
        """
        if self.window not in ("fixed", "variable"):
            raise ValueError("Invalid self.window for ParzenClassifier")
        # TODO: Для variable проверять, что k является положительным целым числом.
        if self.window == "variable" and len(X) < self.k + 1:
            raise ValueError("len(X) must be >= k+1")
        # TODO: Перед сравнением с нулём проверить bandwidth is not None.
        # Для fixed без заданной ширины выдавать ValueError.
        if self.window == "fixed" and self.bandwidth <= 0:
            raise ValueError("bandwidth is <=0, but must be > 0")

        self.X_ = X
        self.y_ = y
        self.classes_ = np.sort(np.unique(y))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Вернуть предсказанные классы для всех строк X через predict_one."""
        # TODO: В predict и predict_one проверять, что fit уже вызван.
        # Если модель не обучена, выдавать RuntimeError.
        result_classes = np.zeros(len(X), dtype=self.classes_.dtype)
        for i in range(len(X)):
            x = X[i]
            prediction = self.predict_one(x)
            result_classes[i] = prediction
        return result_classes

    def predict_one(self, sample: np.ndarray) -> str:
        """Предсказать класс одного объекта по суммам весов каждого класса.

        При ширине 0 голосуют только объекты на нулевом расстоянии.
        При равных суммах выбрать первый класс в отсортированном списке.
        """
        distances = self._distances(sample)  # Будем считать что Sample нет в выборке X

        h = self._window_width(distances)
        if h == 0:
            weights = (distances == 0).astype(float)
        else:
            weights = self.kernel(distances / h)

        votes_per_class = [
            weights[self.y_ == current_class].sum() for current_class in self.classes_
        ]
        return self.classes_[np.argmax(votes_per_class)]

    @staticmethod
    def gaussian_kernel(values: np.ndarray) -> np.ndarray:
        """Вернуть гауссовы веса exp(-2 * values**2), как в ноутбуке."""
        return np.exp(-2 * values**2)

    @staticmethod
    def rectangular_kernel(values: np.ndarray) -> np.ndarray:
        """Вернуть веса прямоугольного ядра [abs(values) <= 1]"""
        return np.abs(values) <= 1

    @staticmethod
    def triangular_kernel(values: np.ndarray) -> np.ndarray:
        """Вернуть веса треугольного ядра (1-abs(r))[abs(values) <= 1]"""
        return (1 - np.abs(values)) * (np.abs(values) <= 1)

    @staticmethod
    def yepanechnikov_kernel(values: np.ndarray) -> np.ndarray:
        """Вернуть веса ядра Епанечникова (1-r**2)[abs(values) <= 1]"""
        return (1 - values**2) * (np.abs(values) <= 1)

    def _distances(self, sample: np.ndarray) -> np.ndarray:
        """Вычислить евклидовы расстояния от sample до всех строк X_."""
        # TODO: Может быть переделать так, чтобы сразу вычислялась матрица расстояний, типо для образца первого - расстояния именно от первого к остальным
        # Для второго - от второго к остальным
        # Матрица n x n получается. С нулями на диагонале, которые можно заполнить np.inf если так надо

        # При том, sample если есть в выборке X, то его бы надо бы как-то исключить.
        return np.linalg.norm(self.X_ - sample, axis=1)

    @staticmethod
    def get_self_distances_matrix(X):
        # X = [
        #   [0, 0]
        #   [3, 4]
        #   [5, 6]
        # ]
        #
        #
        # X[:, None, :]  # (N, 1, D)
        # = [
        #   [[0, 0]]
        #   [[3, 4]]
        #   [[5, 6]]
        # ]
        # X[None, :, :]  # (1, N, D)
        # = [
        #   [[0, 0], [3, 4], [5, 6]]
        # ]
        # -> (N, N, D)
        # = [
        #   [[0, 0], [0, 0], [0, 0]]
        #   [[3, 4], [3, 4], [3, 4]]
        #   [[5, 6], [5, 6], [5, 6]]
        # ]
        # ------
        # = [
        #   [[0, 0], [3, 4], [5, 6]]
        #   [[0, 0], [3, 4], [5, 6]]
        #   [[0, 0], [3, 4], [5, 6]]
        # ]
        # =========
        # = [
        #   [[0, 0] - [0, 0], [0, 0] - [3, 4], [0, 0] - [5, 6]]
        #   [[3, 4] - [0, 0], [3, 4] - [3, 4], [3, 4] - [5, 6]]
        #   [[5, 6] - [0, 0], [5, 6] - [3, 4], [5, 6] - [5, 6]]
        # ]
        # =====
        # = [
        #   [[0, 0],    [-3, -4 ],      [-5, -6 ]]
        #   [[3, 4],    [0, 0   ],      [-2, -2]]
        #   [[5, 6],    [2, 2   ],      [0, 0]]
        # ]
        diff = X[:, None, :] - X[None, :, :]
        distances = np.linalg.norm(
            diff, axis=2
        )  # То есть надо взять норму, именно от значений в ячейках.
        # distances = [
        #   [0,         5,          sqrt(61)]
        #   [5,         0,          sqrt(8) ]
        #   [sqrt(61),  sqrt(8),    0       ]
        # ]

        return distances

    def _window_width(self, distances: np.ndarray) -> float:
        """Получить заданную bandwidth или расстояние до (k + 1)-го соседа."""
        if self.window not in ("fixed", "variable"):
            raise ValueError("Invalid self.window for ParzenClassifier")

        if self.window == "fixed":
            return self.bandwidth

        return np.partition(distances, self.k)[self.k]
