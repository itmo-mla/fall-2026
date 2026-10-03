from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "E-commerce_Customer_Segmentation_2026.csv"

TARGET_COL = "churn_risk_category"
HIGH_RISK_CATEGORIES = {"Medium", "High", "Very High"}

NUMERIC_FEATURES = [
    "age",
    "tenure_months",
    "total_purchases",
    "avg_order_value_usd",
    "total_spent_usd",
    "days_since_last_purchase",
    "return_count",
    "complaint_count",
    "satisfaction_score",
    "email_open_rate",
    "click_through_rate",
    "conversion_rate",
]

CATEGORICAL_FEATURES = [
    "gender",
    "country",
    "income_bracket",
    "education_level",
    "employment_type",
    "marital_status",
    "shopping_channel",
    "device_used",
    "payment_method",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_raw(path=DEFAULT_DATA_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Датасет не найден: {path}. Скачать 'E-commerce Customer Segmentation 2026' "
            "с Kaggle и положить CSV в lab1/data/."
        )
    return pd.read_csv(path)


def binarize_target(series):
    # 0 = низкий риск (Very Low, Low), 1 = повышенный риск (Medium, High, Very High)
    return series.isin(HIGH_RISK_CATEGORIES).astype(int).to_numpy()


def get_feature_names(preprocessor):
    return list(preprocessor.get_feature_names_out())


def load_dataset(path=DEFAULT_DATA_PATH, test_size=0.2, random_state=42):
    df = load_raw(path)

    X = df[ALL_FEATURES]
    y01 = binarize_target(df[TARGET_COL])

    X_train_df, X_test_df, y01_train, y01_test = train_test_split(
        X, y01, test_size=test_size, stratify=y01, random_state=random_state
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    y_train = np.where(y01_train == 1, 1.0, -1.0)
    y_test = np.where(y01_test == 1, 1.0, -1.0)

    return X_train, X_test, y_train, y_test, preprocessor
