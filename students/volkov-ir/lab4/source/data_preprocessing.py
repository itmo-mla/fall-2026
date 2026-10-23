import os
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

KAGGLE_DATASET = "uciml/autompg-dataset"
CSV_NAME = "auto-mpg.csv"

TARGET = "mpg"
# origin – код страны (1, 2, 3), а не величина, car name – текст, оба не берем
FEATURES = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year"]


def load_auto_mpg():
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    df = pd.read_csv(os.path.join(path, CSV_NAME))

    df = df[FEATURES + [TARGET]]
    # у 6 машин мощность записана как "?", заполнять нечем
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    X = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    return X, y, FEATURES


def prepare_data(test_size=0.25, seed=42):
    X, y, feature_names = load_auto_mpg()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_names


# if __name__ == "__main__":
#     X_train, X_test, y_train, y_test, feature_names = prepare_data()
#     print("признаки:", feature_names)
#     print("train:", X_train.shape, "test:", X_test.shape)
#     print("mean:", X_train.mean(axis=0).round(3), "std:", X_train.std(axis=0).round(3))
#     print("mpg train: среднее", round(y_train.mean(), 2), "std", round(y_train.std(), 2))
#     print("корреляции признаков на train:")
#     print(pd.DataFrame(np.corrcoef(X_train, rowvar=False), index=feature_names, columns=feature_names).round(2).to_string())