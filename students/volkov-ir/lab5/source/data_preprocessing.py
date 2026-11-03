import os
import kagglehub
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

KAGGLE_DATASET = "fedesoriano/heart-failure-prediction"
CSV_NAME = "heart.csv"

TARGET = "HeartDisease"
CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]


def load_heart():
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    df = pd.read_csv(os.path.join(path, CSV_NAME))

    # холестерин 0 - помечаем отдельным признаком
    df["CholesterolMissing"] = (df["Cholesterol"] == 0).astype(float)
    # давление 0 - заполним медианой
    df[["Cholesterol", "RestingBP"]] = df[["Cholesterol", "RestingBP"]].replace(0, np.nan)

    # one-hot
    df = pd.get_dummies(df, columns=CATEGORICAL, drop_first=True)
    feature_names = [c for c in df.columns if c != TARGET]

    X = df[feature_names].to_numpy(dtype=float)
    # перевод таргетов в +-1
    y = 2 * df[TARGET].to_numpy(dtype=float) - 1
    return X, y, feature_names


def prepare_data(test_size=0.25, seed=42):
    X, y, feature_names = load_heart()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

    # медиана и среднее с std
    imputer = SimpleImputer(strategy="median").fit(X_train)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # столбец из единиц первым
    X_train = np.hstack([np.ones((len(X_train), 1)), X_train])
    X_test = np.hstack([np.ones((len(X_test), 1)), X_test])

    return X_train, X_test, y_train, y_test, ["bias"] + feature_names


# if __name__ == "__main__":
#     X_train, X_test, y_train, y_test, feature_names = prepare_data()
#     print("признаки:", feature_names)
#     print("train:", X_train.shape, "test:", X_test.shape)
#     print("доля класса +1: train", round((y_train == 1).mean(), 3), "test", round((y_test == 1).mean(), 3))
#     print("mean:", X_train[:, 1:].mean(axis=0).round(3), "std:", X_train[:, 1:].std(axis=0).round(3))
#     print("число обусловленности X^T X:", round(np.linalg.cond(X_train.T @ X_train), 1))
