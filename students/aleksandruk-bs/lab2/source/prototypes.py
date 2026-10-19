"""Жадное удаление не-эталонов для 1NN.

Ищем подмножество Ω ⊆ X^ℓ, на котором 1NN ошибается не чаще, чем на полной выборке.
Функционал считается по всем ℓ объектам, знаменатель не сжимается вместе с Ω:

    Q(Ω) = (1/ℓ) sum_{i=1}^{ℓ} [ a(x_i; Ω \\ {x_i}) != y_i ]

Если x_i уже вынут из Ω, множество соседей — это Ω целиком.
Иначе объект исключается, чтобы не измерять расстояние до самого себя.

Шаг: Ω := X^ℓ. Пока находится x ∈ Ω, для которого Q(Ω \\ {x}) не выше текущего Q,
удаляем такой x с наименьшим Q. При равенстве сначала снимаем ошибочный объект,
затем объект с большим отступом (он глубже внутри своего класса).
Последний объект класса не удаляем: иначе этот класс нельзя будет назначить.

Отступ на полной выборке (ρ_свой — ближайший другой свой, ρ_чужой — ближайший чужой):

    M(x_i) = (ρ_чужой - ρ_свой) / (ρ_чужой + ρ_свой)

M < 0 значит, что 1NN на полной выборке ошибается.
Роли по этому отступу назначает lecture_roles: эталон остался в Ω,
шум удалён при M < 0, неинформативный удалён при M ≥ 0.
В журнале шагов отдельно отмечено, упало Q или осталось тем же.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Selection:
    mask: np.ndarray
    roles: np.ndarray
    margins: np.ndarray
    removed_order: list[int]
    history_removed: list[int]
    history_loo: list[float]
    history_kind: list[str]


def metric_margins(dist: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Отступ 1NN. dist[i, i] должен быть +inf."""
    margins = np.empty(len(y), dtype=np.float64)
    for label in np.unique(y):
        own = y == label
        foreign = ~own
        rho_own = dist[np.ix_(own, own)].min(axis=1)
        rho_foreign = dist[np.ix_(own, foreign)].min(axis=1)
        denom = rho_foreign + rho_own
        value = np.divide(rho_foreign - rho_own, denom, out=np.full(own.sum(), -1.0), where=denom > 0)
        margins[own] = value
    return margins


def _two_nearest(dist: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blocked = dist.copy()
    blocked[:, ~mask] = np.inf
    np.fill_diagonal(blocked, np.inf)
    part = np.argpartition(blocked, kth=1, axis=1)[:, :2]
    rows = np.arange(dist.shape[0])[:, None]
    pair = blocked[rows, part]
    order = np.argsort(pair, axis=1, kind="mergesort")
    top = part[rows, order]
    return top[:, 0], top[:, 1]


def _error_count(y: np.ndarray, neighbor: np.ndarray) -> int:
    return int(np.sum(y[neighbor] != y))


def greedy_condense(dist: np.ndarray, y: np.ndarray) -> Selection:
    """Жадное сжатие. Q не растёт. Возвращает маску эталонов и роли всех объектов."""
    n = len(y)
    if n < 3:
        raise ValueError("для двух ближайших соседей нужно хотя бы 3 объекта")

    margins = metric_margins(dist, y)
    mask = np.ones(n, dtype=bool)
    roles = np.array(["эталон"] * n, dtype=object)
    classes = np.unique(y)

    neighbor, second = _two_nearest(dist, mask)
    current_errors = _error_count(y, neighbor)
    history_removed = [0]
    history_loo = [current_errors / n]
    history_kind = ["старт"]
    removed_order: list[int] = []

    while int(mask.sum()) > len(classes):
        counts = {int(label): int(np.sum(y[mask] == label)) for label in classes}
        candidates = [i for i in np.where(mask)[0] if counts[int(y[i])] > 1]
        if not candidates:
            break

        wrong = y[neighbor] != y
        best: tuple[int, int, float, int] | None = None
        for index in candidates:
            affected = neighbor == index
            if not np.any(affected):
                new_errors = current_errors
            else:
                old_wrong = int(np.sum(wrong[affected]))
                new_wrong = int(np.sum(y[second[affected]] != y[affected]))
                new_errors = current_errors - old_wrong + new_wrong
            # Минимизируем число ошибок, затем предпочитаем текущую ошибку 1NN,
            # затем больший отступ (объект глубже внутри класса).
            rank = (new_errors, 0 if wrong[index] else 1, -float(margins[index]), int(index))
            if best is None or rank < best:
                best = rank

        assert best is not None
        new_errors, _, _, index = best
        if new_errors > current_errors:
            break

        kind = "Q уменьшилось" if new_errors < current_errors else "Q не изменилось"
        roles[index] = kind
        mask[index] = False
        removed_order.append(int(index))

        neighbor, second = _two_nearest(dist, mask)
        current_errors = _error_count(y, neighbor)
        if current_errors != new_errors:
            raise RuntimeError(
                f"пересчёт 1NN дал {current_errors} ошибок, шаг ожидал {new_errors}"
            )
        history_removed.append(len(removed_order))
        history_loo.append(current_errors / n)
        history_kind.append(kind)

    return Selection(
        mask=mask,
        roles=roles,
        margins=margins,
        removed_order=removed_order,
        history_removed=history_removed,
        history_loo=history_loo,
        history_kind=history_kind,
    )


def predict_1nn(X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray) -> np.ndarray:
    """1NN по евклидовой метрике. Обучение здесь — это само множество эталонов."""
    diff = X_query[:, None, :] - X_train[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return y_train[np.argmin(dist, axis=1)]


def lecture_roles(mask: np.ndarray, margins: np.ndarray) -> np.ndarray:
    """Роли по определению из лекции, после того как жадный алгоритм остановился.

    Эталон остался в Ω. Шум — удалённый объект с M < 0: ближайший чужой ближе своего.
    Неинформативный — удалённый объект с M ≥ 0: свой класс и так узнаётся без него.
    """
    roles = np.empty(len(mask), dtype=object)
    roles[mask] = "эталон"
    removed = ~mask
    roles[removed & (margins < 0)] = "шум"
    roles[removed & (margins >= 0)] = "неинформативный"
    return roles


def predict_loo_1nn(dist: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Метка 1NN для каждого объекта исходной выборки. mask=None — вся выборка."""
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    neighbor, _ = _two_nearest(dist, mask)
    return y[neighbor]


def loo_1nn_risk(dist: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Q(Ω). Знаменатель — объём исходной выборки, а не |Ω|."""
    pred = predict_loo_1nn(dist, y, mask)
    return float(np.mean(pred != y))
