"""Профиль компактности и CCV для отбора эталонов по критерию 1NN."""

import math

import numpy as np

# TODO: Реализовать FastPrototypeSelector в отдельном модуле fast_prototypes.py.
# Сохранить интерфейс PrototypeSelector, формат history и правила выбора при равенстве CCV.
# Кэшировать порядок соседей и пересчитывать профиль только для затронутых объектов.
# При добавлении вставлять кандидата в список соседей, при удалении брать запасного соседа.
# Проверить совпадение индексов, CCV и истории с простой реализацией этого класса.


class PrototypeSelector:
    """Отбирает эталоны из переданной обучающей подвыборки по всем признакам.

    Тестовые данные в интерфейс отбора не входят. Индексы эталонов относятся
    к строкам X_train, а не ко всему исходному CSV. l = len(X_train) - control_size.
    Текущее множество prototype_indices локально для каждого запуска отбора.
    Для каждого пробного множества вычисляется CCV через метод ccv.

    Оба метода возвращают (indices, history). history содержит список словарей:
        step: номер шага, начиная с 0;
        indices: копия текущих индексов эталонов;
        changed_index: выбранный добавленный/удалённый индекс, на шаге 0 равен None;
        n_prototypes: количество эталонов;
        ccv: значение CCV после шага.

    Шаг 0 содержит начальное множество, остальные шаги содержат принятые изменения.
    Причина остановки сохраняется в stop_reason_. Расстояния и веса CCV
    вычисляются при первом обращении и используются повторно
    для разных множеств эталонов.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        control_size: int = 2,
    ) -> None:
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train)
        if X_train.ndim != 2 or len(X_train) < 2 or not np.isfinite(X_train).all():
            raise ValueError(
                "X_train должен содержать хотя бы два объекта с конечными признаками"
            )
        if y_train.ndim != 1 or len(y_train) != len(X_train):
            raise ValueError(
                "y_train должен содержать одну метку для каждой строки X_train"
            )
        if (
            isinstance(control_size, (bool, np.bool_))
            or not isinstance(control_size, (int, np.integer))
            or not 1 <= control_size < len(X_train)
        ):
            raise ValueError("Должно выполняться 1 <= control_size < len(X_train)")

        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.control_size = int(control_size)
        self.distance_cache_: np.ndarray | None = None
        self.ccv_weights_: np.ndarray | None = None
        self.stop_reason_: str | None = None

    def compact_profile(
        self,
        M: int,
        indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Вернуть профиль компактности для первых M соседей из indices.

        None означает весь train. Результат имеет форму (M,) и содержит
        доли ошибок для каждого ранга соседа по всем объектам X_train.
        Собственный индекс исключается, совпадающие координаты допустимы.
        При равных расстояниях выбрать меньший индекс строки X_train.
        """
        if indices is None:
            indices = np.arange(len(self.X_train))
        else:
            indices = np.asarray(indices)

        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("indices должен быть одномерным массивом целых индексов")
        if np.any(indices < 0) or np.any(indices >= len(self.X_train)):
            raise ValueError("Индексы должны принадлежать диапазону [0, len(X_train))")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Индексы эталонов не должны повторяться")
        if (
            isinstance(M, (bool, np.bool_))
            or not isinstance(M, (int, np.integer))
            or not 1 <= M < len(indices)
        ):
            raise ValueError("Каждому объекту нужны M соседей: 1 <= M < len(indices)")

        if self.distance_cache_ is None:
            self.distance_cache_ = self._build_distance_cache()
        neighbours = self._cached_neighbours(indices, M)
        return np.mean(
            self.y_train[:, None] != self.y_train[neighbours], axis=0
        )  # Считаем профиль компакности по формуле

    def ccv(self, indices: np.ndarray | None = None) -> float:
        """Рассчитать CCV для 1NN по указанным эталонам; None означает весь train.

        Исключить собственный индекс проверяемого объекта из его соседей.
        При равных расстояниях выбрать меньший индекс строки X_train.
        Для каждого объекта требуется control_size доступных соседей.
        Веса зависят от размера полного train и control_size.
        """
        if self.ccv_weights_ is None:
            self.ccv_weights_ = self._calculate_ccv_weights_iterative()
        profile = self.compact_profile(len(self.ccv_weights_), indices)
        return float(profile @ self.ccv_weights_)

    def greedy_addition(self) -> tuple[np.ndarray, list[dict[str, object]]]:
        """Вернуть эталоны и историю жадного добавления.

        Начать с первого объекта каждого класса. Для каждого кандидата
        рассчитать CCV множества с этим объектом и выбрать лучший вариант.
        Принять добавление только при строгом уменьшении CCV.
        Если начальных эталонов мало, дополнить до control_size + 1.
        При равных оценках выбрать меньший индекс.
        """
        prototype_indices = self._initial_prototypes()
        all_indices = np.arange(len(self.X_train))
        self.stop_reason_ = None
        current_ccv = self.ccv(prototype_indices)
        history = [self._history_entry(prototype_indices, current_ccv, 0)]

        while True:
            candidates = np.setdiff1d(all_indices, prototype_indices)
            if len(candidates) == 0:
                self.stop_reason_ = "Выбраны все обучающие объекты"
                break

            # Ищем объект, добавление которого даёт наименьший CCV.
            best_index = None
            best_ccv = np.inf
            for index in candidates:
                candidate_indices = np.append(prototype_indices, index)
                candidate_ccv = self.ccv(candidate_indices)
                if candidate_ccv < best_ccv:
                    best_index = int(index)
                    best_ccv = candidate_ccv

            if best_ccv >= current_ccv:
                self.stop_reason_ = "Нет добавления, уменьшающего CCV"
                break

            prototype_indices = np.append(prototype_indices, best_index)
            current_ccv = best_ccv
            history.append(
                self._history_entry(
                    prototype_indices, current_ccv, len(history), best_index
                )
            )

        return prototype_indices, history

    def greedy_removal(
        self,
        *,
        tolerance: float = 0.0,
    ) -> tuple[np.ndarray, list[dict[str, object]]]:
        """Вернуть эталоны и историю жадного удаления, начиная со всего train.

        Для каждого допустимого удаления рассчитать CCV оставшегося множества.
        Выбрать лучший вариант и принять его, если новый CCV не превышает
        предыдущий CCV + tolerance. Допуск относится к одному шагу.
        Сохранить каждый класс и минимум control_size + 1 объектов.
        При равных оценках выбрать меньший индекс.
        """
        if (
            isinstance(tolerance, (bool, np.bool_))
            or not isinstance(tolerance, (int, float, np.integer, np.floating))
            or not np.isfinite(tolerance)
            or tolerance < 0
        ):
            raise ValueError("tolerance должен быть конечным неотрицательным числом")

        prototype_indices = np.arange(len(self.X_train))
        minimum_size = self.control_size + 1
        self.stop_reason_ = None
        current_ccv = self.ccv(prototype_indices)
        history = [self._history_entry(prototype_indices, current_ccv, 0)]

        while True:
            if len(prototype_indices) == minimum_size:
                self.stop_reason_ = "Достигнуто минимальное число эталонов"
                break

            # Ищем объект, удаление которого даёт наименьший CCV.
            best_index = None
            best_ccv = np.inf
            prototype_labels = self.y_train[prototype_indices]
            for index in prototype_indices:
                # Последний эталон своего класса удалять нельзя.
                if np.count_nonzero(prototype_labels == self.y_train[index]) == 1:
                    continue

                candidate_indices = prototype_indices[prototype_indices != index]
                candidate_ccv = self.ccv(candidate_indices)
                if candidate_ccv < best_ccv:
                    best_index = int(index)
                    best_ccv = candidate_ccv

            if best_index is None:
                self.stop_reason_ = "Каждый класс представлен единственным эталоном"
                break
            if best_ccv > current_ccv + tolerance:
                self.stop_reason_ = "Лучшее удаление превышает допуск CCV"
                break

            prototype_indices = prototype_indices[prototype_indices != best_index]
            current_ccv = best_ccv
            history.append(
                self._history_entry(
                    prototype_indices, current_ccv, len(history), best_index
                )
            )

        return prototype_indices, history

    def _initial_prototypes(self) -> np.ndarray:
        """Выбрать по одному объекту каждого класса и дополнить до минимума."""
        _, prototype_indices = np.unique(self.y_train, return_index=True)
        extra_count = self.control_size + 1 - len(prototype_indices)
        if extra_count > 0:
            remaining_indices = np.setdiff1d(
                np.arange(len(self.X_train)), prototype_indices
            )
            prototype_indices = np.concatenate(
                (prototype_indices, remaining_indices[:extra_count])
            )
        return prototype_indices

    @staticmethod
    def _history_entry(
        indices: np.ndarray,
        ccv: float,
        step: int,
        changed_index: int | None = None,
    ) -> dict[str, object]:
        """Сохранить независимую копию множества и значение CCV для одного шага."""
        return {
            "step": step,
            "indices": indices.copy(),
            "changed_index": changed_index,
            "n_prototypes": len(indices),
            "ccv": float(ccv),
        }

    def _cached_neighbours(self, indices: np.ndarray, count: int) -> np.ndarray:
        """Получить первые count соседей из текущего множества эталонов."""
        # Расстояния от каждого объекта до существующих эталонов: (N, |Omega|).
        distances = self.distance_cache_[:, indices]
        # При равных расстояниях учитываем исходный индекс эталона.
        order = np.lexsort(
            (np.broadcast_to(indices, distances.shape), distances), axis=1
        )[:, :count]
        return indices[order]  # (N, count)

    def _build_distance_cache(self) -> np.ndarray:
        """Построить общий кэш расстояний для текущей выборки; диагональ равна inf."""
        X = np.asarray(self.X_train, dtype=float)
        if X.ndim != 2 or len(X) < 2 or not np.isfinite(X).all():
            raise ValueError(
                "X должен содержать хотя бы два объекта с конечными признаками"
            )
        distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        return distances

    def _calculate_ccv_weights_iterative(self) -> np.ndarray:
        """Получить комбинаторные веса CCV для заданного control_size."""
        # self.control_size = control_size = k = L-l
        k = self.control_size
        L = len(self.X_train)
        l = L - k
        weights = np.empty(k, dtype=float)
        weights[0] = l / (L - 1)
        # R(m+1) / R(m) = (L-l-m) / (L-1-m).
        for m in range(1, k):
            weights[m] = weights[m - 1] * (k - m) / (L - 1 - m)
        return weights

    def _calculate_ccv_weights(self) -> np.ndarray:
        """Получить комбинаторные веса CCV для заданного control_size."""
        # self.control_size = control_size = k = L-l
        k = self.control_size
        L = len(self.X_train)
        l = L - k
        denom_C = math.comb(L - 1, l)
        weights = np.asarray(
            [math.comb(L - 1 - m, l - 1) / denom_C for m in range(1, k + 1)]
        )
        return weights
