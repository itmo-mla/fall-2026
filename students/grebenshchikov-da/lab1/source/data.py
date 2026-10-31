from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Tuple
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/tungstor1/1b-parquet"
FEATURE_NAMES = [
    "year",
    "selling_price",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "torque_nm",
    "torque_rpm",
    "seats",
]


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame
    cleaning_log: List[str]


def download_dataset(data_dir: Path) -> Path:
    """Скачивает датасет, если его ещё нет на диске."""
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / "1B.parquet"
    archive_path = data_dir / "dataset.zip"

    if parquet_path.exists():
        return parquet_path

    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request) as response:
        archive_path.write_bytes(response.read())

    with zipfile.ZipFile(archive_path) as archive:
        archive.extract("1B.parquet", data_dir)

    archive_path.unlink()
    return parquet_path


def parse_torque_nm(value: object) -> float:
    """Переводит крутящий момент из Nm или kgm в Н·м."""
    text = str(value).lower().strip().replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match is None:
        return np.nan

    torque = float(match.group())
    if "nm" in text:
        return torque

    if "kgm" in text:
        # В трёх строках 115, 130 и 145 ошибочно подписаны как kgm.
        # Для обычных машин такие значения kgm физически неправдоподобны.
        if torque > 60:
            return torque
        return torque * 9.80665

    return np.nan


def parse_torque_rpm(value: object) -> float:
    """Извлекает нижнюю границу оборотов максимального момента."""
    text = str(value).lower().strip().replace(",", "")
    match = re.search(r"(?:@|\bat\b)\s*(\d+(?:\.\d+)?)", text)
    if match is None:
        return np.nan
    return float(match.group(1))


def clean_obvious_errors(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Исправляет значения, смысл которых однозначен без статистики."""
    frame = frame.copy()
    log: List[str] = []

    wrong_year = frame["year"] == 3011
    if wrong_year.any():
        frame.loc[wrong_year, "year"] = 2011
        log.append("year: значение 3011 заменено на 2011")

    mileage_text = frame["mileage"].astype(str).str.strip()
    comma_values = mileage_text.str.contains(",", regex=False)
    for value in mileage_text[comma_values].unique():
        log.append(f"mileage: {value} преобразовано в {value.replace(',', '.')}")
    frame["mileage"] = pd.to_numeric(
        mileage_text.str.replace(",", ".", regex=False),
        errors="coerce",
    )

    frame["torque_nm"] = frame["torque"].map(parse_torque_nm)
    frame["torque_rpm"] = frame["torque"].map(parse_torque_rpm)

    # Для лабораторной решается бинарная задача Diesel против Petrol.
    frame = frame[frame["fuel"].isin(["Diesel", "Petrol"])].copy()
    frame["target"] = np.where(frame["fuel"] == "Diesel", 1, -1)
    return frame, log


class CarPreprocessor:
    """Заполняет пропуски и масштабирует признаки по train-выборке."""

    def fit(self, frame: pd.DataFrame) -> "CarPreprocessor":
        valid_price = frame[frame["selling_price"].notna()]
        self.price_by_brand_year = valid_price.groupby(["name", "year"])[
            "selling_price"
        ].median()
        self.price_by_brand = valid_price.groupby("name")["selling_price"].median()
        self.price_global = float(valid_price["selling_price"].median())

        valid_mileage = frame[frame["mileage"].notna()]
        self.mileage_by_brand_year = valid_mileage.groupby(["name", "year"])[
            "mileage"
        ].median()
        self.mileage_by_brand = valid_mileage.groupby("name")["mileage"].median()
        self.mileage_global = float(valid_mileage["mileage"].median())

        self.power_reference = frame[frame["max_power"] > 0].copy()
        self.power_global = float(self.power_reference["max_power"].median())

        valid_rpm = frame[frame["torque_rpm"].notna()]
        self.rpm_by_car = valid_rpm.groupby(
            ["name", "year", "engine", "torque_nm"]
        )["torque_rpm"].median()
        self.rpm_by_brand_engine = valid_rpm.groupby(["name", "engine"])[
            "torque_rpm"
        ].median()
        self.rpm_by_brand = valid_rpm.groupby("name")["torque_rpm"].median()
        self.rpm_global = float(valid_rpm["torque_rpm"].median())

        cleaned, _ = self._fill_values(frame, collect_log=False)
        matrix = cleaned[FEATURE_NAMES].to_numpy(dtype=float)
        self.remaining_medians = np.nanmedian(matrix, axis=0)
        matrix = np.where(np.isnan(matrix), self.remaining_medians, matrix)
        self.means = matrix.mean(axis=0)
        self.stds = matrix.std(axis=0)
        self.stds[self.stds == 0] = 1.0
        return self

    @staticmethod
    def _group_value(
        row: pd.Series,
        by_brand_year: pd.Series,
        by_brand: pd.Series,
        global_value: float,
    ) -> float:
        value = by_brand_year.get((row["name"], row["year"]), np.nan)
        if pd.isna(value):
            value = by_brand.get(row["name"], np.nan)
        if pd.isna(value):
            value = global_value
        return float(value)

    def _similar_power(self, row: pd.Series) -> float:
        candidates = self.power_reference
        exact_columns = [
            "name",
            "year",
            "engine",
            "mileage",
            "torque_nm",
            "seats",
            "transmission",
        ]
        exact = candidates
        for column in exact_columns:
            exact = exact[exact[column] == row[column]]
        if not exact.empty:
            return float(exact["max_power"].median())

        same_engine = candidates[
            (candidates["name"] == row["name"])
            & (candidates["year"] == row["year"])
            & (candidates["engine"] == row["engine"])
        ]
        if not same_engine.empty:
            return float(same_engine["max_power"].median())

        same_brand_year = candidates[
            (candidates["name"] == row["name"])
            & (candidates["year"] == row["year"])
        ]
        if not same_brand_year.empty:
            return float(same_brand_year["max_power"].median())

        return self.power_global

    def _similar_rpm(self, row: pd.Series) -> float:
        exact_key = (row["name"], row["year"], row["engine"], row["torque_nm"])
        value = self.rpm_by_car.get(exact_key, np.nan)
        if pd.isna(value):
            value = self.rpm_by_brand_engine.get(
                (row["name"], row["engine"]),
                np.nan,
            )
        if pd.isna(value):
            value = self.rpm_by_brand.get(row["name"], np.nan)
        if pd.isna(value):
            value = self.rpm_global
        return float(value)

    def _fill_values(
        self,
        frame: pd.DataFrame,
        collect_log: bool,
    ) -> Tuple[pd.DataFrame, List[str]]:
        frame = frame.copy()
        log: List[str] = []

        for index in frame.index[frame["selling_price"].isna()]:
            row = frame.loc[index]
            value = self._group_value(
                row,
                self.price_by_brand_year,
                self.price_by_brand,
                self.price_global,
            )
            frame.at[index, "selling_price"] = value
            if collect_log:
                log.append(
                    f"Sales_ID {row['Sales_ID']}: selling_price заменён на {value:.2f}"
                )

        for index in frame.index[frame["mileage"].isna()]:
            row = frame.loc[index]
            value = self._group_value(
                row,
                self.mileage_by_brand_year,
                self.mileage_by_brand,
                self.mileage_global,
            )
            frame.at[index, "mileage"] = value
            if collect_log:
                log.append(f"Sales_ID {row['Sales_ID']}: mileage заменён на {value:.2f}")

        for index in frame.index[frame["max_power"] <= 0]:
            row = frame.loc[index]
            value = self._similar_power(row)
            frame.at[index, "max_power"] = value
            if collect_log:
                log.append(
                    f"Sales_ID {row['Sales_ID']}: max_power заменён на {value:.2f}"
                )

        for index in frame.index[frame["torque_rpm"].isna()]:
            row = frame.loc[index]
            value = self._similar_rpm(row)
            frame.at[index, "torque_rpm"] = value
            if collect_log:
                log.append(
                    f"Sales_ID {row['Sales_ID']}: torque_rpm заменён на {value:.0f}"
                )

        return frame, log

    def transform(self, frame: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
        cleaned, log = self._fill_values(frame, collect_log=True)
        matrix = cleaned[FEATURE_NAMES].to_numpy(dtype=float)
        matrix = np.where(np.isnan(matrix), self.remaining_medians, matrix)
        matrix = (matrix - self.means) / self.stds
        return matrix, cleaned, log


def load_prepared_data(
    project_dir: Path,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> PreparedData:
    parquet_path = download_dataset(project_dir / "data")
    raw = pd.read_parquet(parquet_path)
    cleaned, cleaning_log = clean_obvious_errors(raw)

    # Одинаковые наборы признаков считаются одной группой. Все повторения
    # попадают только в одну выборку и не упрощают итоговую проверку модели.
    grouped = cleaned.copy()
    grouped["_feature_group"] = pd.util.hash_pandas_object(
        grouped[FEATURE_NAMES],
        index=False,
    ).astype("uint64")
    group_targets = grouped[["_feature_group", "target"]].drop_duplicates(
        "_feature_group"
    )
    if grouped.groupby("_feature_group")["target"].nunique().max() != 1:
        raise ValueError("Одинаковые признаки встретились у разных классов")

    train_validation_groups, test_groups = train_test_split(
        group_targets,
        test_size=test_size,
        random_state=random_state,
        stratify=group_targets["target"],
    )

    validation_fraction = validation_size / (1.0 - test_size)
    train_groups, validation_groups = train_test_split(
        train_validation_groups,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=train_validation_groups["target"],
    )

    def select_groups(group_frame: pd.DataFrame) -> pd.DataFrame:
        selected = grouped["_feature_group"].isin(group_frame["_feature_group"])
        return grouped.loc[selected].drop(columns="_feature_group").copy()

    train_frame = select_groups(train_groups)
    validation_frame = select_groups(validation_groups)
    test_frame = select_groups(test_groups)

    preprocessor = CarPreprocessor().fit(train_frame)
    X_train, train_frame, train_log = preprocessor.transform(train_frame)
    X_validation, validation_frame, validation_log = preprocessor.transform(
        validation_frame
    )
    X_test, test_frame, test_log = preprocessor.transform(test_frame)
    cleaning_log.extend(train_log)
    cleaning_log.extend(validation_log)
    cleaning_log.extend(test_log)

    return PreparedData(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=train_frame["target"].to_numpy(dtype=int),
        y_validation=validation_frame["target"].to_numpy(dtype=int),
        y_test=test_frame["target"].to_numpy(dtype=int),
        feature_names=FEATURE_NAMES,
        train_frame=train_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
        cleaning_log=cleaning_log,
    )
