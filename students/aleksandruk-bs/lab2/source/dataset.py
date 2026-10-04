"""Загрузка Wine. Файл датасета в репозиторий не кладётся: он уже в sklearn."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES_RU = [
    "спирт",
    "яблочная кислота",
    "зола",
    "щёлочность золы",
    "магний",
    "фенолы",
    "флаваноиды",
    "нефлаваноиды",
    "проантоцианины",
    "интенсивность цвета",
    "оттенок",
    "OD280/OD315",
    "пролин",
]


@dataclass
class WineData:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    feature_names_ru: list[str]
    class_names: list[str]


def load_wine_data() -> WineData:
    bunch = load_wine()
    return WineData(
        X=bunch.data.astype(np.float64),
        y=bunch.target.astype(np.int64),
        feature_names=list(bunch.feature_names),
        feature_names_ru=list(FEATURE_NAMES_RU),
        class_names=[f"класс {name[-1]}" for name in bunch.target_names],
    )


def standardize(X: np.ndarray, scaler: StandardScaler | None = None) -> tuple[np.ndarray, StandardScaler]:
    """Нулевое среднее и единичный разброс. Без этого евклидова метрика смотрит только на крупные признаки."""
    if scaler is None:
        scaler = StandardScaler()
        return scaler.fit_transform(X), scaler
    return scaler.transform(X), scaler


def stratified_split(
    data: WineData,
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Отложенная выборка. Масштаб считается только по обучению."""
    X_train, X_test, y_train, y_test = train_test_split(
        data.X,
        data.y,
        test_size=test_size,
        random_state=random_state,
        stratify=data.y,
    )
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    return X_train_std, X_test_std, y_train, y_test
