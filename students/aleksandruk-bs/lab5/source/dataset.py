"""Диабет индейцев пима: бинарная классификация по восьми числовым признакам.

Таблица National Institute of Diabetes and Digestive and Kidney Diseases.
Скачивается при запуске и в репозиторий не кладётся.
Метка 1 — диабет, 0 — нет. В функционале y = 2 y_файл − 1, то есть −1 и +1.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.request import urlopen

import numpy as np
from sklearn.model_selection import train_test_split


FEATURE_NAMES = (
    "беременности",
    "глюкоза",
    "давление",
    "толщина кожи",
    "инсулин",
    "индекс массы",
    "наследственность",
    "возраст",
)

SOURCE_URLS = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
    "https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv",
)

_TABLE: np.ndarray | None = None


@dataclass
class Split:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    zero_counts: tuple[int, ...]


def _parse_table(text: str) -> np.ndarray:
    rows = []
    for line in text.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0]:
            continue
        try:
            rows.append([float(part) for part in parts])
        except ValueError:
            continue
    data = np.asarray(rows, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 9 or data.shape[0] < 700:
        raise ValueError(f"ожидалась таблица 768×9, получено {data.shape}")
    return data


def _download() -> np.ndarray:
    global _TABLE
    if _TABLE is not None:
        return _TABLE
    errors = []
    for url in SOURCE_URLS:
        try:
            with urlopen(url, timeout=30) as response:
                _TABLE = _parse_table(response.read().decode("utf-8"))
            return _TABLE
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("не удалось скачать Pima Indians Diabetes:\n" + "\n".join(errors))


def load_diabetes(test_size: float = 0.3, seed: int = 42) -> Split:
    """768 пациенток. Признаки нормируются по обучению, свободный член в X не входит."""
    table = _download()
    X = table[:, :-1]
    y = table[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    mean = X_train.mean(axis=0)
    centered = X_train - mean
    scale = np.sqrt(np.mean(centered * centered, axis=0))
    scale = np.where(scale > 0, scale, 1.0)
    zero_counts = tuple(int(np.sum(X[:, column] == 0)) for column in range(X.shape[1]))
    return Split(
        (X_train - mean) / scale,
        (X_test - mean) / scale,
        y_train,
        y_test,
        FEATURE_NAMES,
        zero_counts,
    )


def design_matrix(X: np.ndarray) -> np.ndarray:
    """Единичный столбец справа: последний коэффициент — свободный член."""
    X = np.asarray(X, dtype=np.float64)
    return np.column_stack([X, np.ones(len(X))])


def to_signed(y: np.ndarray) -> np.ndarray:
    """0/1 → −1/+1. Класс «диабет» остаётся положительным."""
    return np.where(np.asarray(y) >= 0.5, 1.0, -1.0)
