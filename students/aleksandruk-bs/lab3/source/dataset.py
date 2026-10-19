"""Social Network Ads: покупка по возрасту и зарплате.

Тот же CSV, что лежит на Kaggle (Social Network Ads, 400 строк).
Скрипт качает его по открытой ссылке, без ключа Kaggle, и в репозиторий не кладёт.
Метки Purchased 0/1 переводятся в {-1, +1}.
"""

from __future__ import annotations

import io
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Зеркала одного и того же файла. Kaggle API требует kaggle.json, эти ссылки — нет.
SOURCE_URLS = (
    "https://huggingface.co/datasets/Rodrigopiva/Social_Network_Ads.csv/resolve/main/Social_Network_Ads.csv",
    "https://raw.githubusercontent.com/ApoorvRusia/Logistic-Regression-Classifier-On-Social-Network-Advertising/master/Social_Network_Ads.csv",
)
KAGGLE_PAGE = "https://www.kaggle.com/datasets/akram24/social-network-ads"

_TABLE: pd.DataFrame | None = None


@dataclass
class Split:
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    n_neg: int
    n_pos: int


def _download_table() -> pd.DataFrame:
    global _TABLE
    if _TABLE is not None:
        return _TABLE
    errors: list[str] = []
    for url in SOURCE_URLS:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=40) as response:
                raw = response.read()
            frame = pd.read_csv(io.BytesIO(raw))
            frame.columns = [str(column).strip() for column in frame.columns]
            needed = {"Age", "EstimatedSalary", "Purchased"}
            if not needed <= set(frame.columns):
                raise ValueError(f"в файле нет колонок {needed}")
            _TABLE = frame
            return frame
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("не удалось скачать Social Network Ads:\n" + "\n".join(errors))


def _to_pm1(y: np.ndarray) -> np.ndarray:
    labels = set(np.unique(y).tolist())
    if labels <= {0, 1}:
        return np.where(y == 0, -1.0, 1.0)
    if labels == {-1, 1}:
        return y.astype(np.float64)
    raise ValueError(f"ожидались метки 0/1, получено {labels}")


def social_ads(test_size: float = 0.3, seed: int = 42) -> Split:
    """Возраст и оценка зарплаты. Пол и User ID не входят: модель остаётся двумерной."""
    frame = _download_table()
    X = frame[["Age", "EstimatedSalary"]].to_numpy(dtype=np.float64)
    y = _to_pm1(frame["Purchased"].to_numpy())
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    scaler = StandardScaler().fit(X_train)
    y_train = y_train.astype(np.float64)
    y_test = y_test.astype(np.float64)
    return Split(
        name="реклама",
        X_train=scaler.transform(X_train),
        X_test=scaler.transform(X_test),
        y_train=y_train,
        y_test=y_test,
        feature_names=("возраст", "зарплата"),
        n_neg=int(np.sum(y == -1)),
        n_pos=int(np.sum(y == 1)),
    )


def social_ads_with_noise(test_size: float = 0.3, seed: int = 42) -> Split:
    """Те же объекты плюс столбец чистого шума: по |w| видно, берёт ли его линейный SVM."""
    base = social_ads(test_size=test_size, seed=seed)
    rng = np.random.default_rng(seed)
    noise_train = rng.normal(0.0, 1.0, size=(len(base.y_train), 1))
    noise_test = rng.normal(0.0, 1.0, size=(len(base.y_test), 1))
    return Split(
        name="реклама и шум",
        X_train=np.hstack([base.X_train, noise_train]),
        X_test=np.hstack([base.X_test, noise_test]),
        y_train=base.y_train,
        y_test=base.y_test,
        feature_names=("возраст", "зарплата", "шум"),
        n_neg=base.n_neg,
        n_pos=base.n_pos,
    )
