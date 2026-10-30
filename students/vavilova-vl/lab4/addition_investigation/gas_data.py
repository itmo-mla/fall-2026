"""Загрузка и очистка данных газоконденсатных испытаний скважин."""

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data.csv"
TARGET = "Дебит газа(ст)"

# Исходный столбец → короткое имя для таблиц и графиков.
FEATURES = {
    "Глубина манометра": "Глубина",
    "Dшт": "Dшт",
    "Руст(ср)": "Руст",
    "Руст(кон)": "Руст_кон",
    "Рзаб(ср)": "Рзаб",
    "Рзаб(кон)": "Рзаб_кон",
    "Pлин(ср)": "Рлин",
    "Рлин(кон)": "Рлин_кон",
    "Туст": "Туст",
    "Тзаб": "Тзаб",
    "Tлин": "Тлин",
    "Нэф": "Нэф",
    "Рпл. Тек (послед точка на КВД)": "Рпл",
    "Pсб": "Рсб",
    "Ro_c": "ρ_конд",
    "Удельная плотность газа": "ρ_газ",
}


@dataclass
class GasData:
    X: np.ndarray            # ℓ × n, признаки в исходных единицах; пропуски — NaN
    y: np.ndarray            # дебит газа, тыс. м³/сут
    groups: np.ndarray       # номер серии испытаний: режимы одной скважины
    feature_names: list[str]
    n_raw: int               # строк в файле до очистки
    fixes: list[str]         # что исправлено в данных

    @property
    def shape(self) -> tuple[int, int]:
        return self.X.shape


def load_gas_data(path: Path = DATA_PATH) -> GasData:
    df = pd.read_csv(path, sep=";", decimal=",").dropna(how="all")
    df.columns = df.columns.str.strip()
    n_raw = len(df)
    fixes = []

    # Серия испытаний — подряд идущие режимы одной скважины: одна глубина манометра
    # и одна эффективная толщина пласта. Глубину у строк без неё берём из предыдущей.
    series = df["Глубина манометра"].ffill().astype(str) + "|" + df["Нэф"].astype(str)
    groups = pd.factorize(series)[0]

    # У трёх замеров нет забойных данных (Рзаб, Тзаб, глубина) — их не восстановить.
    keep = df["Рзаб(ср)"].notna().to_numpy()
    fixes.append(f"удалено {int((~keep).sum())} замера без забойного давления и температуры")
    df, groups = df[keep].copy(), groups[keep]

    # Явные опечатки: плотность конденсата 186 кг/м³ при остальных 770–801,
    # линейное давление в конце режима 41–48 атм при среднем за режим около 100.
    bad_rho = df["Ro_c"] < 500
    df.loc[bad_rho, "Ro_c"] = np.nan
    fixes.append(f"Ro_c < 500 кг/м³ ({int(bad_rho.sum())} шт.) считается опечаткой")
    bad_line = (df["Pлин(ср)"] - df["Рлин(кон)"]).abs() > 30
    df.loc[bad_line, "Рлин(кон)"] = df.loc[bad_line, "Pлин(ср)"]
    fixes.append(f"Рлин(кон), отличающееся от Рлин(ср) больше чем на 30 атм ({int(bad_line.sum())} шт.), "
                 "заменено на Рлин(ср)")

    # Остальные пропуски заполняет Preprocessor по обучающей выборке.
    features = df[list(FEATURES)]
    n_missing = int(features.isna().sum().sum())
    fixes.append(f"{n_missing} пропуска остаются NaN и заполняются по обучающей выборке")

    return GasData(
        X=features.to_numpy(dtype=float),
        y=df[TARGET].to_numpy(dtype=float),
        groups=pd.factorize(groups)[0],
        feature_names=list(FEATURES.values()),
        n_raw=n_raw,
        fixes=fixes,
    )


class Preprocessor:
    """Заполнение пропусков и стандартизация по статистикам обучающей выборки."""

    def fit(self, X: np.ndarray, groups: np.ndarray) -> "Preprocessor":
        self.median_ = np.nanmedian(X, axis=0)
        # Если в серии нет ни одного значения признака, её медиана — NaN, и берётся общая.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.group_median_ = {g: np.nanmedian(X[groups == g], axis=0) for g in np.unique(groups)}
        X = self._impute(X, groups)
        self.mean_, self.std_ = X.mean(axis=0), X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray, groups: np.ndarray) -> np.ndarray:
        return (self._impute(X, groups) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray, groups: np.ndarray) -> np.ndarray:
        return self.fit(X, groups).transform(X, groups)

    def _impute(self, X: np.ndarray, groups: np.ndarray) -> np.ndarray:
        X = X.copy()
        for i, j in zip(*np.nonzero(np.isnan(X))):
            value = self.group_median_.get(groups[i], self.median_)[j]
            X[i, j] = self.median_[j] if np.isnan(value) else value
        return X
