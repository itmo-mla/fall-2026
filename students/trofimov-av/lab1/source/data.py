"""Загрузка и подготовка данных для лабораторной работы № 1."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_iris


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class IrisBinaryDataset:
    """Два класса Iris в представлении, удобном для бинарной классификации."""

    features: FloatArray
    labels: FloatArray
    feature_names: tuple[str, ...]
    class_names: tuple[str, str]


def load_iris_versicolor_virginica() -> IrisBinaryDataset:
    """Загрузить два наиболее похожих класса Iris.

    Для наглядной двумерной визуализации используются длина и ширина
    лепестка. Метки versicolor и virginica преобразуются соответственно
    в -1 и +1.
    """

    iris = load_iris()
    class_mask = iris.target != 0
    petal_columns = (2, 3)

    features = np.asarray(
        iris.data[class_mask][:, petal_columns], dtype=np.float64
    )
    labels = np.where(iris.target[class_mask] == 1, -1.0, 1.0)
    feature_names = tuple(str(iris.feature_names[i]) for i in petal_columns)
    class_names = (str(iris.target_names[1]), str(iris.target_names[2]))

    return IrisBinaryDataset(
        features=features,
        labels=labels,
        feature_names=feature_names,
        class_names=class_names,
    )
