from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


COLUMN_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]

# Нули в этих признаках физически невозможны и трактуются как пропуски.
ZERO_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

DATASET_HANDLE = "kumargh/pimaindiansdiabetescsv"
CSV_NAME = "pima-indians-diabetes.csv"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class DiabetesSplit:
    """Стандартизированные выборки диабета и их метаданные"""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    frame: pd.DataFrame
    zero_rows: int

    @property
    def train_counts(self) -> np.ndarray:
        return _counts(self.y_train, len(self.class_names))

    @property
    def test_counts(self) -> np.ndarray:
        return _counts(self.y_test, len(self.class_names))


def _counts(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Частоты классов после обратного перевода −1/+1 в 0/1"""
    return np.bincount(((y + 1) // 2).astype(int), minlength=n_classes)


def download_dataset(data_dir: Path = DATA_DIR) -> Path:
    """Возвращает путь к CSV: kagglehub, при сбое — локальная копия"""
    try:
        import kagglehub

        folder = Path(kagglehub.dataset_download(
            DATASET_HANDLE, output_dir=str(data_dir)
        ))
        csv_path = folder / CSV_NAME
        if csv_path.exists():
            return csv_path
        raise FileNotFoundError(f"В каталоге {folder} нет {CSV_NAME}")
    except Exception as exc:
        local_copy = data_dir / CSV_NAME
        if local_copy.exists():
            print(f"kagglehub недоступен ({type(exc).__name__}); "
                  f"используется локальная копия {local_copy}")
            return local_copy
        raise RuntimeError(
            "Не удалось получить датасет. Установите kagglehub и проверьте "
            f"подключение к сети либо положите {CSV_NAME} в {data_dir}"
        ) from exc


def load_raw_frame(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Чтение CSV без заголовка с заданными именами столбцов"""
    csv_path = download_dataset(data_dir)
    frame = pd.read_csv(csv_path, header=None, names=COLUMN_NAMES)
    if frame.shape != (768, 9):
        raise ValueError(f"Ожидалась таблица (768, 9), получено {frame.shape}")
    return frame


def fill_missing(train: pd.DataFrame, test: pd.DataFrame
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Нули заменяются медианой, вычисленной только по обучающей части"""
    train, test = train.copy(), test.copy()
    for part in (train, test):
        part[ZERO_COLUMNS] = part[ZERO_COLUMNS].replace(0, np.nan)
    medians = train[ZERO_COLUMNS].median()
    train[ZERO_COLUMNS] = train[ZERO_COLUMNS].fillna(medians)
    test[ZERO_COLUMNS] = test[ZERO_COLUMNS].fillna(medians)
    return train, test


def scale_by_train(X_train: np.ndarray, X_test: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Z-масштабирование; статистики берутся только с обучающей выборки"""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    std = np.where(std > 0, std, 1.0)
    return (X_train - mean) / std, (X_test - mean) / std


def encode_labels(y: np.ndarray) -> np.ndarray:
    """Метки 0/1 -> −1/+1, как требует постановка SVM"""
    return np.where(np.asarray(y) == 1, 1.0, -1.0)


def prepare_diabetes(
    test_size: float = 0.30, seed: int = 42, data_dir: Path = DATA_DIR
) -> DiabetesSplit:
    """Загрузка, очистка, стратифицированное деление и стандартизация"""
    frame = load_raw_frame(data_dir)
    zero_rows = int(frame[ZERO_COLUMNS].eq(0).any(axis=1).sum())

    feature_names = [name for name in COLUMN_NAMES if name != "Outcome"]
    train_part, test_part, y_train_raw, y_test_raw = train_test_split(
        frame[feature_names], frame["Outcome"].to_numpy(dtype=int),
        test_size=test_size, random_state=seed, stratify=frame["Outcome"],
    )
    train_part, test_part = fill_missing(train_part, test_part)

    X_train = train_part.to_numpy(dtype=float)
    X_test = test_part.to_numpy(dtype=float)
    X_train, X_test = scale_by_train(X_train, X_test)

    if not (np.isfinite(X_train).all() and np.isfinite(X_test).all()):
        raise ValueError("После подготовки остались нечисловые значения")

    return DiabetesSplit(
        X_train=X_train,
        y_train=encode_labels(y_train_raw),
        X_test=X_test,
        y_test=encode_labels(y_test_raw),
        feature_names=feature_names,
        class_names=["Без диабета (−1)", "Диабет (+1)"],
        frame=frame,
        zero_rows=zero_rows,
    )
