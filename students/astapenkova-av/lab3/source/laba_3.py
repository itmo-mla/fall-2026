import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from pathlib import Path

# картинки сохраняем в lab3/images, именно на эту папку ссылается README
images_dir = Path(__file__).resolve().parent.parent / "images"
images_dir.mkdir(parents=True, exist_ok=True)

# 1. Датасет для бинарной классификации: Breast Cancer Wisconsin (Diagnostic)
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

# 3. Ядра
def linear_kernel(A, B):
    return (A @ B.T)

def quad_kernel(A, B):
    return (A @ B.T) ** 2

def rbf_kernel(A,B, gamma):
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>, считаем сразу для всех пар без двойного цикла
    sq_dist = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2 * (A @ B.T)
    return np.exp(-gamma * np.maximum(sq_dist, 0.0))

def poly_monoms(A, B, d, gamma=1.00):
    return (gamma * (A @ B.T) + 1) ** d

def make_kernel(name, p):
    if name == "linear":
        return linear_kernel
    if name == "quad":
        return quad_kernel
    if name == "poly_monoms":
        return lambda A, B: poly_monoms(A, B, p["degree"], p["gamma"])
    if name == "rbf":
        return lambda A, B: rbf_kernel(A, B, p["gamma"])
    raise ValueError(f"Неизвестное ядро: {name}")

def qu(K,y):
    # Q_ij = y_i * y_j * K_ij, в матричном виде Q = (y y^T) * K (поэлементно)
    return np.outer(y, y) * K

EPS = 1e-6

# 2. Двойственная задача: находим вектор лямбда
def solve_dual_task(Q, y, C):
    l = len(y)

    def f(lam):
        return 0.5*lam @ Q @ lam - lam.sum()

    def grad(lam):
        return Q @ lam - 1.0

    # минимизируем 0.5 lam^T Q lam - sum(lam) при sum(lam_i y_i) = 0 и 0 <= lam_i <= C
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

# 4. Линейный классификатор: для линейного ядра вектор w восстанавливается явно,
# w = sum_i lam_i y_i x_i,  a(x) = sign(<w, x> - w0)
def compute_w(model):
    return (model["lam_sv"] * model["y_sv"]) @ model["X_sv"]

def predict_linear(w, w0, X_new):
    return np.where(X_new @ w - w0 >= 0, 1.0, -1.0)

# эталон: SVC из sklearn с теми же ядром и параметрами
def fit_sklearn(X, y, kernel_name, p):
    if kernel_name == "linear":
        model = SVC(kernel="linear", C=p["C"], tol=1e-8)
    elif kernel_name == "quad":
        model = SVC(kernel="poly", degree=2, gamma=1.0, coef0=0.0, C=p["C"], tol=1e-8)
    elif kernel_name == "poly_monoms":
        model = SVC(kernel="poly", degree=p["degree"], gamma=p["gamma"], coef0=1.0, C=p["C"], tol=1e-8)
    elif kernel_name == "rbf":
        model = SVC(kernel="rbf", gamma=p["gamma"], C=p["C"], tol=1e-8)
    else:
        raise ValueError(kernel_name)
    return model.fit(X, y)

# подбор параметров кросс-валидацией собственной реализации
def cross_val_accuracy(X, y, kernel, C, cv):
    scores = []
    for tr, va in cv.split(X, y):
        m = fit(X[tr], y[tr], kernel, C, eps=EPS)
        scores.append(accuracy(y[va], predict(m, X[va])))
    return np.mean(scores)

print("обучение:", X_train.shape, " тест:", X_test.shape)

grids = {
    "linear":      [{"C": c} for c in [0.01, 0.1, 1, 10]],
    "quad":        [{"C": c} for c in [0.01, 0.1, 1]],
    "poly_monoms": [{"degree": d, "gamma": 1/30, "C": c} for d in [2, 3] for c in [0.1, 1, 10]],
    "rbf":         [{"gamma": g, "C": c} for g in [0.01, 1/30, 0.1] for c in [1, 10]],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best = {}
cv_scores = {}   # все результаты кросс-валидации, нужны для графика отбора параметров
print(f"\n{'ядро':12}{'CV':>8}  лучшие параметры")
for name, grid in grids.items():
    scored = [(cross_val_accuracy(X_train, y_train, make_kernel(name, p), p["C"], cv), p) for p in grid]
    score, p = max(scored, key=lambda t: t[0])   # при равенстве берётся первый, т.е. меньший C
    best[name] = (score, p)
    cv_scores[name] = scored
    print(f"{name:12}{score:>8.4f}  {p}")

# обучаем все ядра с лучшими параметрами и сравниваем с эталоном
models = {}
sk_models = {}
print(f"\n{'ядро':12}{'тест своя':>12}{'тест sklearn':>15}{'опорных своя':>15}{'опорных sklearn':>18}")
for name, (score, p) in best.items():
    m = fit(X_train, y_train, make_kernel(name, p), p["C"], eps=EPS)
    sk = fit_sklearn(X_train, y_train, name, p)
    models[name] = m
    sk_models[name] = sk
    print(
        f"{name:12}"
        f"{accuracy(y_test, predict(m, X_test)):>12.4f}"
        f"{sk.score(X_test, y_test):>15.4f}"
        f"{len(m['lam_sv']):>15d}"
        f"{sk.support_.size:>18d}")

# 4. Линейный классификатор в явном виде
lin = models["linear"]
lin_sk = sk_models["linear"]
w = compute_w(lin)
w_sk = lin_sk.coef_[0]
lin_pred = predict_linear(w, lin["w0"], X_test)

print("\nЛинейный классификатор a(x) = sign(<w, x> - w0), C =", lin["C"])
print(f"Точность на тесте (своя):    {accuracy(y_test, lin_pred):.4f}")
print(f"Точность на тесте (sklearn): {lin_sk.score(X_test, y_test):.4f}")
print(f"||w - w_sklearn|| / ||w_sklearn||: {np.linalg.norm(w - w_sk) / np.linalg.norm(w_sk):.4e}")
print(f"w0 своя: {lin['w0']:.4f}, w0 sklearn: {-lin_sk.intercept_[0]:.4f}")
top = np.argsort(-np.abs(w))[:5]
print("Признаки с наибольшим |w|:", ", ".join(f"{feature_names[i]} ({w[i]:+.3f})" for i in top))

# 6. Сравнение с эталоном для выбранного ядра (лучшее на CV)
chosen_name = max(best, key=lambda n: best[n][0])
p_best = best[chosen_name][1]
chosen = models[chosen_name]
chosen_sk = sk_models[chosen_name]

print("\nВыбранное ядро:", chosen_name, p_best)

own_pred = predict(chosen, X_test)
sk_pred = chosen_sk.predict(X_test)

a_own = decision_function(chosen, X_test)
a_sk = chosen_sk.decision_function(X_test)

print("\nСравнение с эталоном:")
print(f"Accuracy своя:    {accuracy(y_test, own_pred):.4f}")
print(f"Accuracy sklearn: {accuracy(y_test, sk_pred):.4f}")
print(f"Точность на обучении (своя): {accuracy(y_train, predict(chosen, X_train)):.4f}")
print(f"Доля совпавших предсказаний: {np.mean(own_pred == sk_pred):.4f}")
print(f"Средняя разница решающих функций: {np.mean(np.abs(a_own - a_sk)):.4e}")
print(f"Максимальная разница решающих функций: {np.max(np.abs(a_own - a_sk)):.4e}")
print("Число опорных векторов:", len(chosen["lam_sv"]), "(своя),", chosen_sk.support_.size, "(sklearn)")
print(f"w0 своя: {chosen['w0']:.4f}, w0 sklearn: {-chosen_sk.intercept_[0]:.4f}")

# 5. Визуализация
# график 1: веса линейного классификатора, своя реализация и sklearn
idx = np.arange(len(w))
plt.figure(figsize=(12, 5))
plt.bar(idx - 0.2, w, width=0.4, label="Своя SVM", color="tab:blue")
plt.bar(idx + 0.2, w_sk, width=0.4, label="sklearn SVC", color="tab:orange")
plt.xticks(idx, feature_names, rotation=90)
plt.ylabel("Вес признака $w_j$")
plt.title(f"Линейный классификатор: веса признаков (C = {lin['C']})")
plt.legend()
plt.grid(alpha=0.3, axis="y")
plt.savefig(images_dir / "svm_linear_weights.png", dpi=150, bbox_inches="tight")
plt.show()

# график 2: отступы объектов для выбранного ядра
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
plt.axvspan(-1, 1, color="gray", alpha=0.2, label="Полоса отступа")
plt.axvline(0, color="black", linewidth=1.5, label="Граница классификации a(x)=0")
plt.axvline(-1, color="gray", linestyle="--")
plt.axvline(1, color="gray", linestyle="--")
plt.scatter(a[y_train == 1], jitter[y_train == 1], c="tab:red", s=30, alpha=0.7, label="Злокачественные (M)")
plt.scatter(a[y_train == -1], jitter[y_train == -1], c="tab:blue", s=30, alpha=0.7, label="Доброкачественные (B)")
plt.scatter(a[boundary], jitter[boundary], s=110, facecolors="none", edgecolors="green", linewidths=1.5, label="Опорные-граничные: 0 < λ < C")
plt.scatter(a[bounded], jitter[bounded], s=120, marker="s", facecolors="none", edgecolors="black", label="Опорные: λ = C")
plt.scatter(a[errors], jitter[errors], s=150, marker="x", color="orange", linewidths=1.8, label="Ошибки классификации")
plt.xlabel("Значение решающей функции a(x)")
plt.ylabel("Случайное смещение точек")
plt.title(
    f"SVM ({chosen_name}), обучающая выборка: "
    f"периферийных {peripheral.sum()}, "
    f"граничных {boundary.sum()}, "
    f"λ=C: {bounded.sum()}, "
    f"ошибок {errors.sum()}"
)
plt.yticks([])
plt.legend(loc="best")
plt.grid(alpha=0.2)
plt.savefig(images_dir / "svm_margins.png", dpi=150, bbox_inches="tight")
plt.show()

# график 3: решающие функции своей SVM и sklearn на тесте
plt.figure(figsize=(7, 6))
plt.scatter(a_sk, a_own, alpha=0.65, label="Тестовые объекты")
low = min(a_sk.min(), a_own.min())
high = max(a_sk.max(), a_own.max())
plt.plot([low, high], [low, high], "k--", label="Идеальное совпадение")
plt.xlabel("Решающая функция sklearn")
plt.ylabel("Решающая функция собственной SVM")
plt.title(
    f"Сравнение SVM ({chosen_name})\n"
    f"Максимальная разница: "
    f"{np.max(np.abs(a_own - a_sk)):.2e}"
)
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(images_dir / "svm_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# график 4: отбор параметров, точность на кросс-валидации для каждой точки сетки
def param_label(name, p):
    if name == "poly_monoms":
        return f"d={p['degree']}"
    if name == "rbf":
        return f"γ={p['gamma']:.3g}"
    return "C"

fig, axes = plt.subplots(1, len(cv_scores), figsize=(16, 4.5), sharey=True)
for ax, (name, scored) in zip(axes, cv_scores.items()):
    lines = {}
    for score, p in scored:
        lines.setdefault(param_label(name, p), []).append((p["C"], score))
    for label, pts in lines.items():
        cs, sc = zip(*pts)
        ax.plot(cs, sc, marker="o", label=label if name in ("poly_monoms", "rbf") else "CV accuracy")
    b_score, b_p = best[name]
    ax.scatter([b_p["C"]], [b_score], s=180, facecolors="none", edgecolors="red", linewidths=2, zorder=3, label="Лучшие параметры")
    ax.set_xscale("log")
    ax.set_xlabel("C")
    ax.set_title(name + (" (выбрано)" if name == chosen_name else ""))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
axes[0].set_ylabel("Accuracy на 5-fold CV")
fig.suptitle("Отбор параметров SVM кросс-валидацией собственной реализации")
fig.savefig(images_dir / "svm_cv_selection.png", dpi=150, bbox_inches="tight")
plt.show()

# график 5: матрица ошибок выбранной модели на тесте
classes = [(1.0, "Злокачественная (M)"), (-1.0, "Доброкачественная (B)")]
cm = np.array([[np.sum((y_test == t) & (own_pred == pr)) for pr, _ in classes] for t, _ in classes])
print("\nМатрица ошибок на тесте (строки — истинный класс M, B; столбцы — предсказанный):")
print(cm)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.colorbar(label="Число объектов")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xticks([0, 1], [c for _, c in classes])
plt.yticks([0, 1], [c for _, c in classes])
plt.xlabel("Предсказанный класс")
plt.ylabel("Истинный класс")
plt.title(f"Матрица ошибок на тесте, SVM ({chosen_name})\naccuracy = {accuracy(y_test, own_pred):.4f}")
plt.savefig(images_dir / "svm_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# график 6: разделяющая граница на двух признаках
# в 30-мерном пространстве границу не нарисовать, поэтому отдельно обучаем SVM на двух признаках
pair = ["worst radius", "worst concave points"]
cols = [feature_names.index(f) for f in pair]
X2_train, X2_test = X_train[:, cols], X_test[:, cols]

models_2d = [
    ("linear", f"Линейное ядро, C = {lin['C']}", make_kernel("linear", {}), lin["C"]),
    ("poly_monoms", f"poly_monoms: d = {p_best.get('degree', 3)}, γ = 1/2, C = {p_best['C']}",
     make_kernel("poly_monoms", {"degree": p_best.get("degree", 3), "gamma": 1 / 2}), p_best["C"]),
]

xx, yy = np.meshgrid(
    np.linspace(X2_train[:, 0].min() - 0.5, X2_train[:, 0].max() + 0.5, 300),
    np.linspace(X2_train[:, 1].min() - 0.5, X2_train[:, 1].max() + 0.5, 300))
grid_points = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, (name, title, kernel, C2) in zip(axes, models_2d):
    m2 = fit(X2_train, y_train, kernel, C2, eps=EPS)
    zz = decision_function(m2, grid_points).reshape(xx.shape)
    acc2 = accuracy(y_test, predict(m2, X2_test))

    ax.contourf(xx, yy, zz, levels=[zz.min(), 0, zz.max()], colors=["tab:blue", "tab:red"], alpha=0.12)
    ax.contour(xx, yy, zz, levels=[0], colors="black", linewidths=2)
    ax.contour(xx, yy, zz, levels=[-1, 1], colors="gray", linestyles="--")
    ax.scatter(*X2_train[y_train == 1].T, c="tab:red", s=18, alpha=0.6, label="Злокачественные (M), обучение")
    ax.scatter(*X2_train[y_train == -1].T, c="tab:blue", s=18, alpha=0.6, label="Доброкачественные (B), обучение")
    ax.scatter(*X2_test[y_test == 1].T, c="tab:red", marker="^", s=45, edgecolors="black", label="Злокачественные (M), тест")
    ax.scatter(*X2_test[y_test == -1].T, c="tab:blue", marker="^", s=45, edgecolors="black", label="Доброкачественные (B), тест")
    ax.scatter(*m2["X_sv"].T, s=90, facecolors="none", edgecolors="green", linewidths=1.2, label="Опорные векторы")
    ax.plot([], [], color="black", linewidth=2, label="Граница a(x) = 0")
    ax.plot([], [], color="gray", linestyle="--", label="Полоса отступа a(x) = ±1")
    ax.set_xlabel(pair[0] + " (стандартизирован)")
    ax.set_ylabel(pair[1] + " (стандартизирован)")
    ax.set_title(f"{title}\nопорных {len(m2['lam_sv'])}, accuracy на тесте {acc2:.4f}")
    ax.grid(alpha=0.2)
    print(f"2D, {name}: опорных {len(m2['lam_sv'])}, accuracy на тесте {acc2:.4f}")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.08))
fig.suptitle("Разделение опухолей SVM по двум признакам")
fig.savefig(images_dir / "svm_2d_boundary.png", dpi=150, bbox_inches="tight")
plt.show()
