#%%
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

DATA_URL = "https://raw.githubusercontent.com/AbhishekDataAnalyst/Credit-Card-Approval/main/Credit_card.csv"

try:
    IMG_DIR = Path(__file__).parent / "images"
except NameError:
    IMG_DIR = Path.cwd() / "images"
IMG_DIR.mkdir(exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

df = pd.read_csv(DATA_URL)

print(df.shape)
print(df.head())
#%%
#чистка датасета от id
df = df.drop("Ind_ID", axis=1)
#таргет
y = df["Car_Owner"]
X = df.drop("Car_Owner", axis=1)

print(y.value_counts())
#%%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=32,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
#%%
#заполнение пропусков

X_train = X_train.copy()
X_test = X_test.copy()

gender_mode = X_train["GENDER"].mode()[0]
income_median = X_train["Annual_income"].median()
birthday_median = X_train["Birthday_count"].median()

X_train["GENDER"] = X_train["GENDER"].fillna(gender_mode)
X_test["GENDER"] = X_test["GENDER"].fillna(gender_mode)
#профессия
X_train["Type_Occupation"] = X_train["Type_Occupation"].fillna("Unknown")
X_test["Type_Occupation"] = X_test["Type_Occupation"].fillna("Unknown")
#годовой доход
X_train["Annual_income"] = X_train["Annual_income"].fillna(income_median)
X_test["Annual_income"] = X_test["Annual_income"].fillna(income_median)
#возраст
X_train["Birthday_count"] = X_train["Birthday_count"].fillna(birthday_median)
X_test["Birthday_count"] = X_test["Birthday_count"].fillna(birthday_median)

print(X_train.isna().sum().sum(), "пропусков в train")
print(X_test.isna().sum().sum(), "пропусков в test")
#%%
#кодировка категоральных признаков

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

X_train = X_train.astype(int)
X_test = X_test.astype(int)

print("Количество признаков:", X_train.shape[1])
#%%
#кодировка классов
y_train = y_train.map({"Y": 1, "N": -1}).values
y_test = y_test.map({"Y": 1, "N": -1}).values

print(pd.Series(y_train).value_counts())
print(pd.Series(y_test).value_counts())
#%%
#стандартизация
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#%%
#SVM решение двойственной задачи по лямбда
class SVM:
    def __init__(self, C=1.0, kernel="linear", gamma=0.1, degree=3, maxiter=1000):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.maxiter = maxiter

    #трюк с ядром: скалярное произведение заменяется на K(x, x')
    def kernel_matrix(self, X1, X2):
        if self.kernel == "linear":
            return X1 @ X2.T
        if self.kernel == "poly":
            return (self.gamma * X1 @ X2.T + 1) ** self.degree
        if self.kernel == "rbf":
            sq_dist = (
                np.sum(X1 ** 2, axis=1)[:, None]
                + np.sum(X2 ** 2, axis=1)[None, :]
                - 2 * X1 @ X2.T
            )
            return np.exp(-self.gamma * np.maximum(sq_dist, 0))
        raise ValueError("Неизвестное ядро: " + self.kernel)

    #двойственная функция (минимизируем минус лагранжиан)
    @staticmethod
    def dual_loss(lambdas, Q):
        return 0.5 * lambdas @ Q @ lambdas - np.sum(lambdas)

    @staticmethod
    def dual_gradient(lambdas, Q):
        return Q @ lambdas - 1.0

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        N = len(y)

        K = self.kernel_matrix(X, X)
        Q = np.outer(y, y) * K

        #двойственная задача: min 1/2 l^T Q l - sum(l)
        #при 0 <= l_i <= C и sum(l_i * y_i) = 0
        result = minimize(
            self.dual_loss,
            np.zeros(N),
            args=(Q,),
            jac=self.dual_gradient,
            method="SLSQP",
            bounds=[(0.0, self.C)] * N,
            constraints={
                "type": "eq",
                "fun": lambda l: l @ y,
                "jac": lambda l: y
            },
            options={"maxiter": self.maxiter, "ftol": 1e-9}
        )
        if not result.success:
            print("Предупреждение оптимизатора:", result.message)

        lambdas = result.x
        eps = 1e-6

        #опорные: lambda > 0, граничные: 0 < lambda < C
        support = lambdas > eps
        boundary = support & (lambdas < self.C - eps)

        #w0 по граничным опорным векторам, усредняем
        scores = K[:, support] @ (lambdas[support] * y[support])
        self.w0 = np.mean(scores[boundary] - y[boundary])

        self.lambdas = lambdas
        self.support = support
        self.boundary = boundary
        self.support_vectors = X[support]
        self.support_labels = y[support]
        self.support_lambdas = lambdas[support]

        #для линейного ядра восстанавливается w явно
        if self.kernel == "linear":
            self.w = (self.support_lambdas * self.support_labels) @ self.support_vectors
        return self

    def decision_function(self, X):
        K = self.kernel_matrix(np.asarray(X, dtype=float), self.support_vectors)
        return K @ (self.support_lambdas * self.support_labels) - self.w0

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)
#%%
#метрики
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_metrics(name, y_true, y_pred):
    return {
        "Модель": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=1),
        "Recall": recall_score(y_true, y_pred, pos_label=1),
        "F1-score": f1_score(y_true, y_pred, pos_label=1)
    }
#%%
#линейный классификатор - собственный SVM против sklearn SVC
from sklearn.svm import SVC

C = 1.0

start = perf_counter()
svm_linear = SVM(C=C, kernel="linear").fit(X_train, y_train)
time_own = perf_counter() - start

start = perf_counter()
sk_linear = SVC(C=C, kernel="linear").fit(X_train, y_train)
time_sk = perf_counter() - start

y_pred_own = svm_linear.predict(X_test)
y_pred_sk = sk_linear.predict(X_test)

comparison_linear = pd.DataFrame([
    get_metrics("SVM линейный (собственный)", y_test, y_pred_own),
    get_metrics("sklearn SVC линейный", y_test, y_pred_sk),
])
comparison_linear["Опорных"] = [len(svm_linear.support_vectors), len(sk_linear.support_)]
comparison_linear["Время, с"] = [time_own, time_sk]

print(comparison_linear)
print("Совпадение предсказаний:", np.mean(y_pred_own == y_pred_sk))
print("Ограничение sum(lambda*y):", svm_linear.lambdas @ y_train)
print("Максимальная разница w:", np.max(np.abs(svm_linear.w - sk_linear.coef_[0])))
print("w0 собственный / sklearn:", svm_linear.w0, -sk_linear.intercept_[0])
#%%
#трюк с ядром - полиномиальное и гауссово ядра
GAMMA = 1.0 / X_train.shape[1]

rows = []

for kernel in ["poly", "rbf"]:
    start = perf_counter()
    own = SVM(C=C, kernel=kernel, gamma=GAMMA, degree=3).fit(X_train, y_train)
    t_own = perf_counter() - start

    start = perf_counter()
    ref = SVC(C=C, kernel=kernel, gamma=GAMMA, degree=3, coef0=1.0).fit(X_train, y_train)
    t_ref = perf_counter() - start

    m_own = get_metrics("SVM " + kernel + " (собственный)", y_test, own.predict(X_test))
    m_ref = get_metrics("sklearn SVC " + kernel, y_test, ref.predict(X_test))
    m_own["Опорных"], m_own["Время, с"] = len(own.support_vectors), t_own
    m_ref["Опорных"], m_ref["Время, с"] = len(ref.support_), t_ref
    rows += [m_own, m_ref]

    print(kernel, "совпадение предсказаний:", np.mean(own.predict(X_test) == ref.predict(X_test)))

comparison_kernel = pd.DataFrame(rows)
print(comparison_kernel)
#%%
#SVM обучается уже на двух признаках
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
Z_train = pca.fit_transform(X_train)
Z_test = pca.transform(X_test)

settings = [
    ("linear", "Линейное ядро", {}),
    ("poly", "Полиномиальное ядро, d = 3", {"degree": 3, "gamma": 0.5}),
    ("rbf", "Гауссово ядро", {"gamma": 0.5}),
]

models_2d = []
for kernel, title, params in settings:
    own = SVM(C=C, kernel=kernel, maxiter=200, **params).fit(Z_train, y_train)
    ref = SVC(C=C, kernel=kernel, coef0=1.0, **params).fit(Z_train, y_train)
    models_2d.append((title, own, ref))
    print(title, "| accuracy на тесте (собственный / sklearn):",
          accuracy_score(y_test, own.predict(Z_test)),
          accuracy_score(y_test, ref.predict(Z_test)))
#%%
#графики - сверху собственный SVM, снизу из библиоткеи
gx, gy = np.meshgrid(
    np.linspace(Z_train[:, 0].min() - 0.5, Z_train[:, 0].max() + 0.5, 250),
    np.linspace(Z_train[:, 1].min() - 0.5, Z_train[:, 1].max() + 0.5, 250)
)
grid = np.c_[gx.ravel(), gy.ravel()]

fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)

for j, (title, own, ref) in enumerate(models_2d):
    for i, (model, name) in enumerate([(own, "собственный SVM"), (ref, "sklearn SVC")]):
        ax = axes[i, j]
        values = model.decision_function(grid).reshape(gx.shape)
        sv = model.support_vectors if i == 0 else model.support_vectors_

        ax.contourf(gx, gy, values >= 0, levels=[-0.5, 0.5, 1.5], colors=["#f7dfe0", "#dce7fa"])
        ax.contour(gx, gy, values, levels=[-1, 0, 1], colors="black",
                   linestyles=["--", "-", "--"], linewidths=1)

        for c, color in ((-1, "#D95459"), (1, "#3268BF")):
            m = y_train == c
            ax.scatter(Z_train[m, 0], Z_train[m, 1], s=10, c=color, alpha=0.5)
        ax.scatter(sv[:, 0], sv[:, 1], s=50, facecolors="none", edgecolors="black", linewidths=0.6)

        ax.set_title(name + ": " + title)
        ax.set_xlabel("Главная компонента 1")
        ax.set_ylabel("Главная компонента 2")

plt.tight_layout()
plt.savefig(IMG_DIR / "svm_solution.png", dpi=150, bbox_inches="tight")
plt.show()
#%%
#типы объектов для линейного SVM: периферийные, опорные граничные, опорные нарушители
lam = svm_linear.lambdas
eps = 1e-6
mrg = y_train * (X_train @ svm_linear.w - svm_linear.w0)

peripheral = lam <= eps
border = (lam > eps) & (lam < C - eps)
violators = lam >= C - eps

print("Периферийные:", peripheral.sum())
print("Опорные граничные:", border.sum())
print("Опорные нарушители:", violators.sum())
print("Нарушители с отступом от 0 до 1 (классифицированы верно):", np.sum(violators & (mrg > 0)))
print("Нарушители с отступом < 0 (классифицированы неверно):", np.sum(violators & (mrg < 0)))
#%%
