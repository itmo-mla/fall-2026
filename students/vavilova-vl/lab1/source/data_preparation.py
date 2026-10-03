import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split


# =========================
# 1. Загрузка датасета
# =========================

df = pd.read_csv(Path(__file__).with_name("penguins.csv"))

print("Размер датасета:", df.shape)

print("\nСтолбцы:")
print(df.columns.tolist())

print("\nТипы данных:")
print(df.dtypes)

print("\nПропуски:")
print(df.isnull().sum())


# =========================
# 2. Выбор классов
# =========================

df = df[df["species"].isin(["Adelie", "Gentoo"])].copy()

print("\nРаспределение классов:")
print(df["species"].value_counts())


# =========================
# 3. Выбор признаков
# =========================

features = [
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g"
]


# =========================
# 4. Удаление пропусков
# =========================

df = df.dropna(subset=features).copy()

print("\nРазмер после удаления пропусков:", df.shape)

print("\nПропуски после очистки:")
print(df[features].isnull().sum())


# =========================
# 5. Формирование X и y
# =========================

X = df[features].values

y = df["species"].map({
    "Adelie": -1,
    "Gentoo": 1
}).values

print("\nРазмер X:", X.shape)
print("Размер y:", y.shape)


# =========================
# 6. Train / Test
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train_raw = X_train.copy()
X_test_raw = X_test.copy()

print("\nРазмеры выборок:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# =========================
# 7. Стандартизация
# =========================

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std


# =========================
# 8. Добавление bias
# =========================

X_train = np.hstack([
    np.ones((X_train.shape[0], 1)),
    X_train
])

X_test = np.hstack([
    np.ones((X_test.shape[0], 1)),
    X_test
])


print("\nПосле подготовки:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("\nПервый объект:")
print(X_train[0])

print("\nКласс первого объекта:")
print(y_train[0])