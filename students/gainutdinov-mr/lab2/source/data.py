"""Подготовка данных по порядку действий из KNN.ipynb."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Self

import kagglehub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataProcessor:
    """Получает таблицу из Kaggle и подготавливает её для экспериментов.

    Как в ноутбуке, load получает каталог через KaggleHub, берёт первый
    файл из каталога и читает его через pandas. В выбранном датасете один
    CSV. В конструкторе задаётся dataset_id, имя файла не требуется.
    Кэшем и скачиванием управляет KaggleHub; таблица хранится в frame.

    Индексы объектов относятся к исходным строкам CSV и сохраняются при
    разделении и получении подвыборок. _raw_splits хранит исходные координаты,
    а mean_/std_ хранят статистики полного train для масштабирования.
    """

    def __init__(
        self,
        dataset_id: str,
        *,
        target_column: str = "class",
        excluded_columns: Sequence[str] = (),
        random_state: int = 42,
    ) -> None:
        self.dataset_id = dataset_id
        self.target_column = target_column
        self.excluded_columns = tuple(excluded_columns)
        self.random_state = random_state

        self.frame: pd.DataFrame | None = None
        self.feature_names: list[str] = []
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self._raw_splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def load(self) -> pd.DataFrame:
        """Загрузить таблицу так же, как в ячейках ноутбука.

        Получить path через kagglehub.dataset_download(self.dataset_id),
        преобразовать его в Path и получить список файлов через iterdir.
        Прочитать первый файл через pd.read_csv, сохранить в frame
        и вернуть таблицу. В выбранном датасете находится один CSV.

        KaggleHub возвращает путь к данным в своём кэше и при необходимости
        скачивает их. Копировать файлы в репозиторий не нужно.
        """
        path = Path(kagglehub.dataset_download(self.dataset_id))
        filenames = list(path.iterdir())
        if not filenames:
            raise FileNotFoundError(f"В каталоге датасета нет файлов: {path}")

        self.frame = pd.read_csv(filenames[0])
        self.feature_names = []
        self.mean_ = None
        self.std_ = None
        self._raw_splits.clear()
        return self.frame

    def prepare(self, *, train_size: float = 0.7) -> Self:
        """Подготовить загруженную таблицу и вернуть self.

        При необходимости вызвать load. Выделить target_column как y,
        исключить его и excluded_columns из X, сохранить feature_names.
        Выполнить стратифицированное разбиение с train_size и random_state.
        В _raw_splits сохранить (X_raw, y, исходные индексы) для train/test.

        Вычислить mean_/std_ по полному train; нулевые std заменить на 1.
        Исходные координаты сохранить для LOO. Статистики test и выбранных
        эталонов при расчёте параметров масштабирования не используются.
        """
        frame = self.load() if self.frame is None else self.frame
        features = frame.drop(columns=[self.target_column, *self.excluded_columns])
        X = features.to_numpy()
        y = frame[self.target_column].to_numpy()
        indices = np.arange(len(frame))

        (
            X_train_raw,
            X_test_raw,
            y_train,
            y_test,
            train_indices,
            test_indices,
        ) = train_test_split(
            X,
            y,
            indices,
            train_size=train_size,
            stratify=y,
            random_state=self.random_state,
        )

        self.feature_names = features.columns.tolist()
        self._raw_splits = {
            "train": (X_train_raw, y_train, train_indices),
            "test": (X_test_raw, y_test, test_indices),
        }
        self.mean_ = X_train_raw.mean(axis=0)
        self.std_ = X_train_raw.std(axis=0)
        self.std_[self.std_ == 0] = 1
        return self

    def get_split(
        self,
        part: Literal["train", "test"],
        *,
        standardized: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Вернуть (X, y, исходные индексы) после prepare.

        При standardized=True применить mean_/std_ полного train.
        При False вернуть исходные координаты: в LOO масштабирование
        выполняется заново внутри каждого разбиения.

        X имеет форму (n, d), а y и индексы имеют форму (n,). Индексы относятся
        к исходной таблице CSV. Возвращаемые массивы должны быть копиями,
        чтобы изменение данных снаружи не меняло состояние процессора.
        """
        if part not in ("train", "test"):
            raise ValueError("part должен быть 'train' или 'test'.")
        if not self._raw_splits:
            raise RuntimeError("Сначала вызовите prepare().")

        X, y, indices = self._raw_splits[part]
        if standardized:
            X = (X - self.mean_) / self.std_
        else:
            X = X.copy()
        return X, y.copy(), indices.copy()

    def sample(
        self,
        part: Literal["train", "test"],
        size: int,
        *,
        standardized: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Вернуть стратифицированную подвыборку (X, y, исходные индексы).

        Выбрать size объектов только из указанной части с сохранением
        долей классов и random_state. standardized имеет тот же смысл,
        что в get_split; параметры масштабирования на подвыборке не менять.
        Как при подготовке данных для отбора эталонов в ноутбуке,
        если объектов не больше size, вернуть всю указанную часть.
        """
        if not isinstance(size, (int, np.integer)) or size < 1:
            raise ValueError("size должен быть положительным целым числом.")

        X, y, indices = self.get_split(part, standardized=standardized) # Если тут мы получаем нужны сплит - train или test уже после результата train_test_split
        if size >= len(X):
            return X, y, indices

        X_sample, _, y_sample, _, sample_indices, _ = train_test_split( # Тогда зачем делать train_test_split еще и тут?
            X,
            y,
            indices,
            train_size=int(size),
            stratify=y,
            random_state=self.random_state,
        )
        # Ответ - для того, чтобы не делать предотбор, который мы и так уже могли бы сделать при помощи X[:size], y[:size], indicies[:size]
        return X_sample, y_sample, sample_indices

    @staticmethod
    def standardize_pair(
        X_train: np.ndarray,
        X_other: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Вернуть (X_train_scaled, X_other_scaled) для отдельного разбиения.

        Средние/std вычислять только по X_train; нулевые std заменить на 1.
        Обе матрицы преобразовать с одинаковыми параметрами. Это операция
        для LOO и отдельных экспериментов: она сохраняет входные массивы
        и состояние DataProcessor.
        """
        if X_train.ndim != 2 or X_other.ndim != 2:
            raise ValueError("X_train и X_other должны быть двумерными массивами.")
        if len(X_train) == 0:
            raise ValueError("X_train должен содержать хотя бы один объект.")
        if X_train.shape[1] != X_other.shape[1]:
            raise ValueError("Число признаков в X_train и X_other должно совпадать.")

        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1
        return (X_train - mean) / std, (X_other - mean) / std
