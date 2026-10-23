import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import time
from sklearn.model_selection import GridSearchCV # для выбора оптимальной модели
from sklearn.datasets import load_breast_cancer
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

images_dir = Path(__file__).resolve().parent / "images"
images_dir.mkdir(parents=True, exist_ok=True)

data = load_breast_cancer()

X = data.data
feature_names = list(data.feature_names)

# sklearn: malignant = 0, benign = 1
y = np.where(data.target == 0, 1.0, -1.0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# стандартизация: среднее и отклонение считаем только по обучающей выборке
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

def linear_kernel(A, B):
    return (A @ B.T)

def quad_kernel(A, B):
    return (A @ B.T) ** 2

def rbf_kernel(A,B, gamma):
    n, m  = A.shape[0], B.shape[0]
    K = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            diff = A[i]-B[j]
            K[i, j] = np.exp(-gamma * np.sum(diff ** 2))
    return K

def poly_kernel(A, B, d, gamma = 1/30):
    return (gamma * (A @ B.T)) ** d
 
def poly_monoms(A, B, d, gamma=1.00):
    return (gamma * (A @ B.T) + 1) ** d
 
def sigmoid_activ(A, B, k1, k0):   # при k0, k1 >= 0
    return np.tanh(k1 * (A @ B.T) - k0)

# функция для выбора ядра, выбрала 30, потому что в датасете 30 столбцов
def get_kernel(name, gamma=1/30):
    if name == "linear":
        return linear_kernel
    if name == "quad":
        return quad_kernel
    if name == "rbf":
        return lambda A, B: rbf_kernel(A, B, gamma)

def qu(K,y):
    l = len(y)
    Q = np.zeros((l,l))
    for i in range(l):
        for j in range(l):
            Q[i, j] = y[i] * y[j] * K[i, j]
    return Q

C = 1.0 # начальное значение, дальше C подбирается по кросс-валидации
EPS = 1e-6

# находим вектор лямбда для решения двойственной задачи
def solve_dual_task(Q, y, C):
    l = len(y)

    def f(lam):
        return 0.5*lam @ Q @ lam - lam.sum()

    def grad(lam):
        return Q @ lam - 1.0

# минимизация перечитать
    res = minimize(f, x0=np.zeros(l), jac=grad, method="SLSQP", bounds=[(0.0, C)] * l, constraints=[{"type": "eq", "fun": lambda lam: lam @ y, "jac": lambda lam: y}], options={"maxiter": 1000, "ftol": 1e-6},)
    if not res.success:
        print("оптимизатор не сошёлся:", res.message)
    return np.clip(res.x, 0.0, C)

def compute_wo(lam, X, y, C, kernel, eps=1e-6):
    sv = lam > eps
    boundary = sv & (lam < C - eps)

    if not boundary.any():
        print("нет опорных-граничных объектов, w0 считается по всем опорным")
        boundary = sv

    # <w, x_i> = sum_j lam_j y_j K(x_j, x_i)
    K_sv_b = kernel(X[sv], X[boundary])              # (n_sv, n_boundary)
    w_dot_x = (lam[sv] * y[sv]) @ K_sv_b             # (n_boundary,)
 
    return np.mean(w_dot_x - y[boundary])

def fit(X, y, kernel, C, eps=1e-6):
    K = kernel(X, X)
    Q = qu(K, y)
    lam = solve_dual_task(Q, y, C)
    w0 = compute_wo(lam, X, y, C, kernel, eps)
 
    sv = lam > eps
    return {
        "lam": lam,
        "lam_sv": lam[sv],
        "X_sv": X[sv],
        "y_sv": y[sv],
        "w0": w0,
        "kernel": kernel,
        "C": C,
    }

#  a(x) = sum_i lam_i y_i K(x, x_i) - w0
def decision_function(model, X_new):
    K = model["kernel"](X_new, model["X_sv"])        # (m, n_sv)
    return K @ (model["lam_sv"] * model["y_sv"]) - model["w0"]
 
def predict(model, X_new):
    return np.where(decision_function(model, X_new) >= 0, 1.0, -1.0)
 
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def fit_sklearn(X, y, kernel_name, C, gamma=1/30):
    if kernel_name == "linear":
        model = SVC(kernel="linear", C=C, tol=1e-8)
    elif kernel_name == "quad":
        model = SVC(kernel="poly", degree=2, gamma=1.0, coef0=0.0, C=C, tol=1e-8)
    elif kernel_name == "rbf":
        model = SVC(kernel="rbf", gamma=gamma, C=C, tol=1e-8)
    else:
        raise ValueError(kernel_name)
    return model.fit(X, y)
 
print("обучение:", X_train.shape, " тест:", X_test.shape)
 
# # kernel = linear_kernel   # выбранное ядро

# t0 = time.time()
# model = fit(X_train, y_train, kernel, C)
# print(f"обучение заняло {time.time() - t0:.1f} с")
# print(f"точность на обучении: {accuracy(y_train, predict(model, X_train)):.4f}")
# print(f"точность на тесте:    {accuracy(y_test, predict(model, X_test)):.4f}")
 
# # сравнение с эталоном 
# sk_model = SVC(kernel="linear", C=C, tol=1e-8)
# sk_model.fit(X_train, y_train)
 
# own_pred = predict(model, X_test)
# sk_pred = sk_model.predict(X_test)
 
# print(f"\n{'':28}{'своя':>10}{'sklearn':>10}")
# print(f"{'точность на тесте':28}{accuracy(y_test, own_pred):>10.4f}{accuracy(y_test, sk_pred):>10.4f}")
# print(f"{'число опорных векторов':28}{len(model['lam_sv']):>10d}{int(sk_model.n_support_.sum()):>10d}")
# print(f"{'w0':28}{model['w0']:>10.4f}{-sk_model.intercept_[0]:>10.4f}")
# print(f"доля совпавших предсказаний: {np.mean(own_pred == sk_pred):.4f}")

# попоробуем подобрать с помощью GScv
Cs = [0.01, 0.1, 1, 10, 100]
G = [1e-3, 1e-2, 1/30, 0.1]
 
grids = {
    "linear":      {"kernel": ["linear"], "C": Cs},
    "quad":        {"kernel": ["poly"], "degree": [2], "gamma": [1.0], "coef0": [0.0], "C": Cs},
    "poly":        {"kernel": ["poly"], "degree": [2, 3, 4], "gamma": G, "coef0": [0.0], "C": Cs},
    "poly_monoms": {"kernel": ["poly"], "degree": [2, 3, 4], "gamma": G, "coef0": [1.0], "C": Cs},
    "sigmoid":     {"kernel": ["sigmoid"], "gamma": G, "coef0": [-0.5, -1.0], "C": Cs},
    "rbf":         {"kernel": ["rbf"], "gamma": G, "C": Cs},
}

def make_kernel(name, p):
    if name == "linear":
        return linear_kernel
    if name == "quad":
        return quad_kernel
    if name == "poly":
        return lambda A, B: poly_kernel(
            A, B, p["degree"], p["gamma"])
    if name == "poly_monoms":
        return lambda A, B: poly_monoms(
            A, B, p["degree"], p["gamma"])
    if name == "sigmoid":
        return lambda A, B: sigmoid_activ(A, B, p["gamma"], -p["coef0"])
    if name == "rbf":
        return lambda A, B: rbf_kernel(
            A, B, p["gamma"])
    raise ValueError(f"Неизвестное ядро: {name}")

searches = {}
models = {}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(
    f"\n{'ядро':12}"
    f"{'CV':>10}"
    f"{'тест своя':>12}"
    f"{'тест sklearn':>15}"
    f"{'опорных':>12}  лучшие параметры"
)

for name, grid in grids.items():
    pipeline = Pipeline([("svc", SVC(max_iter=2_000_000))])

    grid_pipeline = {"svc__" + key: value
        for key, value in grid.items()}

    s = GridSearchCV(
        estimator=pipeline,
        param_grid=grid_pipeline,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )
    s.fit(X_train, y_train)
    p = {
        key.replace("svc__", ""): value
        for key, value in s.best_params_.items()
    }
    m = fit(X_train, y_train, make_kernel(name, p), p["C"], eps=EPS)

    searches[name] = s
    models[name] = m

    own_acc = accuracy(
        y_test,
        predict(m, X_test))

    sk_acc = s.best_estimator_.score(
        X_test,
        y_test)
    
    print(
        f"{name:12}"
        f"{s.best_score_:>10.4f}"
        f"{own_acc:>12.4f}"
        f"{sk_acc:>15.4f}"
        f"{len(m['lam_sv']):>12d}  {p}")
chosen_name = max(
    searches,
    key=lambda n: searches[n].best_score_)

chosen = models[chosen_name]
chosen_sk = searches[chosen_name].best_estimator_

print("\nВыбранное ядро:", chosen_name)
print("Лучшие параметры:", searches[chosen_name].best_params_)

own_pred = predict(chosen, X_test)
sk_pred = chosen_sk.predict(X_test)

a_own = decision_function(chosen, X_test)
a_sk = chosen_sk.decision_function(X_test)

print("\nСравнение с эталоном:")
print(f"Accuracy своя:   {accuracy(y_test, own_pred):.4f}")
print(f"Accuracy sklearn:{accuracy(y_test, sk_pred):.4f}")

print(
    "Доля совпавших предсказаний:",
      f"{np.mean(own_pred == sk_pred):.4f}")

print(
    "Средняя разница решающих функций:",
    f"{np.mean(np.abs(a_own - a_sk)):.4e}")

print(
    "Максимальная разница решающих функций:",
    f"{np.max(np.abs(a_own - a_sk)):.4e}")

print(
    "Число опорных векторов:",
    len(chosen["lam_sv"]),
    "(своя),",
    chosen_sk.named_steps["svc"].support_.size,
    "(sklearn)")

images_dir = Path(__file__).resolve().parent / "images"
images_dir.mkdir(parents=True, exist_ok=True)
a = decision_function(chosen, X_train)
M = y_train * a

lam = chosen["lam"]
C_best = chosen["C"]

peripheral = lam <= EPS
boundary = (lam > EPS) & (lam < C_best - EPS)
bounded = lam >= C_best - EPS

errors = M < 0

rng = np.random.default_rng(42)
jitter = rng.uniform(-0.4, 0.4, len(a))

plt.figure(figsize=(12, 6))

plt.axvspan(
    -1, 1,
    color="gray",
    alpha=0.2,
    label="Полоса отступа"
)

plt.axvline(
    0,
    color="black",
    linewidth=1.5,
    label="Граница классификации a(x)=0"
)

plt.axvline(-1, color="gray", linestyle="--")
plt.axvline(1, color="gray", linestyle="--")

plt.scatter(
    a[y_train == 1],
    jitter[y_train == 1],
    c="tab:red",
    s=30,
    alpha=0.7,
    label="Злокачественные (M)"
)

plt.scatter(
    a[y_train == -1],
    jitter[y_train == -1],
    c="tab:blue",
    s=30,
    alpha=0.7,
    label="Доброкачественные (B)"
)

plt.scatter(
    a[boundary],
    jitter[boundary],
    s=110,
    facecolors="none",
    edgecolors="green",
    linewidths=1.5,
    label="Опорные-граничные: 0 < λ < C"
)

plt.scatter(
    a[bounded],
    jitter[bounded],
    s=120,
    marker="s",
    facecolors="none",
    edgecolors="black",
    label="Опорные: λ = C"
)

plt.scatter(
    a[errors],
    jitter[errors],
    s=150,
    marker="x",
    color="orange",
    linewidths=1.8,
    label="Ошибки классификации"
)

plt.xlabel("Значение решающей функции a(x)")
plt.ylabel("Случайное смещение точек")

plt.title(
    f"SVM ({chosen_name}): "
    f"периферийных {peripheral.sum()}, "
    f"граничных {boundary.sum()}, "
    f"λ=C: {bounded.sum()}, "
    f"ошибок {errors.sum()}"
)

plt.yticks([])
plt.legend(loc="best")
plt.grid(alpha=0.2)

plt.savefig(
    images_dir / "svm_margins.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()

plt.figure(figsize=(7, 6))

plt.scatter(
    a_sk,
    a_own,
    alpha=0.65,
    label="Тестовые объекты"
)

low = min(a_sk.min(), a_own.min())
high = max(a_sk.max(), a_own.max())
plt.plot(
    [low, high],
    [low, high],
    "k--",
    label="Идеальное совпадение"
)
plt.xlabel("Решающая функция sklearn")
plt.ylabel("Решающая функция собственной SVM")
plt.title(
    f"Сравнение SVM ({chosen_name})\n"
    f"Максимальная разница: "
    f"{np.max(np.abs(a_own - a_sk)):.2e}"
)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(
    images_dir / "svm_comparison.png",
    dpi=150,
    bbox_inches="tight"
)
plt.show()
