from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class WineSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    frame: pd.DataFrame

    @property
    def train_counts(self) -> np.ndarray:
        return np.bincount(self.y_train, minlength=len(self.class_names))

    @property
    def test_counts(self) -> np.ndarray:
        return np.bincount(self.y_test, minlength=len(self.class_names))


def load_wine_frame() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Датасет sklearn"""
    bunch = load_wine()
    frame = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    frame["target"] = bunch.target.astype(int)
    class_names = [str(name) for name in bunch.target_names]
    return frame, class_names, list(bunch.feature_names)


def scale_by_train(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Масштабирование"""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    std = np.where(std > 0, std, 1.0)
    return (X_train - mean) / std, (X_test - mean) / std


def prepare_wine(test_size: float = 0.25, seed: int = 42) -> WineSplit:
    frame, class_names, feature_names = load_wine_frame()
    X = frame[feature_names].to_numpy(dtype=float)
    y = frame["target"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train, X_test = scale_by_train(X_train, X_test)

    if not (np.isfinite(X_train).all() and np.isfinite(X_test).all()):
        raise ValueError("После масштабирования остались нечисловые значения")

    return WineSplit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        class_names=class_names,
        frame=frame,
    )
