"""Загрузка и подготовка данных для лабораторной работы № 4."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_ID = "mikhail1681/walmart-sales"


def load_data(
    csv_path: str | Path | None = None,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Подготовить Walmart Sales для регрессии Weekly_Sales и PCA.

    CSV читается локально либо скачивается через кэш KaggleHub. Дата
    преобразуется в год, месяц и номер недели, магазин кодируется one-hot.
    Строки без целевой переменной удаляются. Пропуски
    заполняются медианами, признаки центрируются и масштабируются по train.
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
    features = frame.drop(columns=["Weekly_Sales"])
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    # Номер магазина — категория: отдельный признак для каждого магазина.
    encoding = ColumnTransformer([
        ("store", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ), ["Store"]),
        ("numeric", SimpleImputer(strategy="median"), features.columns.drop("Store")),
    ])
    preprocessing = make_pipeline(encoding, StandardScaler())
    X_train = preprocessing.fit_transform(X_train)
    X_test = preprocessing.transform(X_test)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # print('sadsd')
    X_train, X_test, y_train, y_test = load_data(test_size=0.3)

    df = pd.DataFrame(X_train)
    print(df.describe())
    print(df.info())
