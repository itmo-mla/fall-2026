import os

import kagglehub
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# качается при запуске, кэш в ~/.cache/kagglehub
KAGGLE_DATASET = "parulpandey/palmer-archipelago-antarctica-penguin-data"
CSV_NAME = "penguins_size.csv"

TARGET = "species"
# только числовые: island выдаёт вид, sex с пропусками
FEATURES = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]


def load_penguins():
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    df = pd.read_csv(os.path.join(path, CSV_NAME))

    df = df[FEATURES + [TARGET]]
    # 2 пингвина без всех признаков — заполнять нечем
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    X = df[FEATURES].to_numpy(dtype=float)
    # виды -> 0, 1, 2 по алфавиту
    class_names, y = np.unique(df[TARGET].to_numpy(), return_inverse=True)
    return X, y, FEATURES, list(class_names)


def prepare_data(test_size=0.25, seed=42):
    X, y, feature_names, class_names = load_penguins()

    # test не трогаем до финальной оценки; stratify — одинаковые доли видов в train и test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

    # среднее и std только по train, иначе утечка из test
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_names, class_names


# самопроверка: размеры, баланс классов, mean≈0 и std≈1 после стандартизации
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, class_names = prepare_data()
    print("классы:", class_names)
    print("train:", X_train.shape, np.bincount(y_train), "test:", X_test.shape, np.bincount(y_test))
    print("mean:", X_train.mean(axis=0).round(3), "std:", X_train.std(axis=0).round(3))
