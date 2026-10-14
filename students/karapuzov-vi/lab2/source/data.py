import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv("species.csv")

print("\n=== Пропуски ===")
print(df.isna().sum())
print("\n=== Статистика ===")
print(df.describe())

print("-----")
print("колиичество строк до дропа по гендеру:", len(df))

df = df.dropna(subset=['sex'])
print("количество строк после дропа гендера:", len(df))

print("значения класса species(целевая переменная) и их возможные значения:")
print(len(df["species"].unique()))
for sp in df["species"].unique():
    print(sp)

sns.countplot(x='species', data=df)
plt.title('Распределение классов')
plt.savefig("plots/target_distribution.png", format="png")
plt.close()

print("График распределения таргетных классов показал -> явного дисбаланса не наблюдается")

plt.figure()
sns.scatterplot(data=df, x="culmen_length_mm", y="culmen_depth_mm", hue="species")
plt.title("Виды по длине и глубине клюва")
plt.xlabel("длина клюва, мм")
plt.ylabel("глубина клюва, мм")
plt.savefig("plots/bill_by_species.png", format="png")
plt.close()

feature_columns = [
    "culmen_length_mm",
    "culmen_depth_mm",
]
X = df[feature_columns]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["species"])

print("X.shape:", X.shape)
print("кодировка species:")
for idx, name in enumerate(label_encoder.classes_):
    print(idx, "->", name)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Засетим среднее значение для все полей с пропусками")
print("среднее считаем по train и тем же числом заполняем test")
train_means = X_train.mean()
X_train = X_train.fillna(train_means)
X_test = X_test.fillna(train_means)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("метки train:", np.unique(y_train, return_counts=True))
print("метки test:", np.unique(y_test, return_counts=True))
print("пропуски train/test:", np.isnan(X_train).sum(), np.isnan(X_test).sum())
