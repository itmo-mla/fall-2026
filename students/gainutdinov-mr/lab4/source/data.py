"""Загрузка и подготовка данных для лабораторной работы № 4."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATASET_ID = "mikhail1681/walmart-sales"


def load_data(
    csv_path: str | Path | None = None,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Подготовить Walmart Sales для регрессии Weekly_Sales и PCA.

    CSV читается локально либо скачивается через кэш KaggleHub. Дата
    преобразуется в год, месяц и номер недели, идентификатор магазина
    исключается. Строки без целевой переменной удаляются. Пропуски
    заполняются медианами, признаки стандартизуются по обучающей выборке.
    Возвращает X_train, X_test, y_train, y_test; продажи не масштабируются.
    """
    if csv_path is None:
        import kagglehub

        directory = Path(kagglehub.dataset_download(DATASET_ID))
        csv_files = sorted(directory.rglob("*.csv"))
        if len(csv_files) != 1:
            raise ValueError(
                "Ожидался один CSV в датасете; укажите csv_path явно."
            )
        csv_path = csv_files[0]

    frame = pd.read_csv(csv_path).drop_duplicates()
    target = pd.to_numeric(frame["Weekly_Sales"], errors="coerce")
    valid = np.isfinite(target)
    frame = frame.loc[valid].copy()
    y = target.loc[valid].to_numpy(dtype=float)

    dates = pd.to_datetime(frame.pop("Date"), dayfirst=True, errors="coerce")
    frame["Year"] = dates.dt.year
    frame["Month"] = dates.dt.month
    frame["Week"] = dates.dt.isocalendar().week.astype(float)
    features = frame.drop(columns=["Weekly_Sales", "Store"])
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)

    X_train, X_test, y_train, y_test = train_test_split(
        features.to_numpy(dtype=float),
        y,
        test_size=test_size,
        random_state=random_state,
    )
    preprocessing = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler()
    )
    X_train = preprocessing.fit_transform(X_train)
    X_test = preprocessing.transform(X_test)
    return X_train, X_test, y_train, y_test
