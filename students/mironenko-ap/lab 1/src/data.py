"""Загрузка, препроцессинг, разбиение — всё в одной функции.

Датасет: Breast Cancer Wisconsin (Diagnostic).
Один CSV: колонка id, 30 числовых признаков, целевая diagnosis ∈ {M, B}.
Тест выделяется стратифицированным сплитом из общего файла.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from .core import log

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "data.csv"
KAGGLE_DATASET = "uciml/breast-cancer-wisconsin-data"

DROP_COLS = ["id"]
TARGET_COL = "diagnosis"


def _ensure_csv(path: Path) -> Path:
    """Возвращает путь к data.csv. Если файла нет — качает с Kaggle."""
    if path.exists():
        return path
    try:
        import kagglehub
    except ImportError as e:
        raise RuntimeError(
            "Нужен пакет kagglehub: pip install kagglehub "
            "(или положите data.csv в data/ вручную)"
        ) from e

    path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    src = root / "data.csv"
    if not src.exists():
        hits = list(root.rglob("data.csv"))
        if not hits:
            raise FileNotFoundError(f"data.csv не найден внутри {root}")
        src = hits[0]
    path.write_bytes(src.read_bytes())
    log(f"[load] скачано с Kaggle → {path}")
    return path


def _load(path=DATA_PATH):
    path = _ensure_csv(Path(path))
    if not path.exists():
        raise FileNotFoundError(f"Не найден {path}")

    df = pd.read_csv(path)

    # Убираем безымянные колонки-индексы в любом месте датасета.
    unnamed_cols = [
        col for col in df.columns
        if str(col).startswith("Unnamed") or str(col).strip() == ""
    ]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    log(f"[load] {path.name}: {df.shape}")
    return df


def _preprocess(df):
    df = df.copy()

    s = df[TARGET_COL].astype(str).str.strip().str.upper()
    y = s.map({"M": 1, "B": -1})
    if y.isna().any():
        raise ValueError(
            f"Неизвестные метки: {s[y.isna()].unique().tolist()}"
        )
    y = y.to_numpy(dtype=np.int64)

    drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop + [TARGET_COL])

    # Признаки числовые, но подстрахуемся на случай пропусков.
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    X = X.astype(np.float64)
    log(f"[preprocess] X={X.shape}, pos_ratio={np.mean(y == 1):.3f}")
    return X.to_numpy(), y, list(X.columns)


def _split(y, val_size=0.15, test_size=0.15, seed=42):
    """Стратифицированное разбиение на train / val / test."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == -1)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def parts(idx):
        n = len(idx)
        n_te = int(n * test_size)
        n_va = int(n * val_size)
        te = idx[:n_te]
        va = idx[n_te:n_te + n_va]
        tr = idx[n_te + n_va:]
        return tr, va, te

    tr_p, va_p, te_p = parts(pos)
    tr_n, va_n, te_n = parts(neg)
    tr = np.r_[tr_p, tr_n]
    va = np.r_[va_p, va_n]
    te = np.r_[te_p, te_n]
    rng.shuffle(tr)
    rng.shuffle(va)
    rng.shuffle(te)
    return tr, va, te


def _standardize(X_tr, X_val, X_te):
    """Статистики считаем только на train, применяем ко всем трём."""
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std[std == 0] = 1.0
    return (X_tr - mean) / std, (X_val - mean) / std, (X_te - mean) / std


def load_data(path=DATA_PATH, val_size=0.15, test_size=0.15, seed=42):
    df = _load(path)
    X, y, names = _preprocess(df)
    tr, va, te = _split(y, val_size, test_size, seed)
    X_tr, X_val, X_te = _standardize(X[tr], X[va], X[te])
    log(f"[split] train={len(tr)}, val={len(va)}, test={len(te)}")
    return dict(
        X_train=X_tr, y_train=y[tr],
        X_val=X_val, y_val=y[va],
        X_test=X_te, y_test=y[te],
        feat_names=names,
    )
