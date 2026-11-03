from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

from data import data_load, split_and_scale
from logreg_irls import add_bias, sigmoid, newton_step, irls_step, fit, fit_with_norms
from compare_sklearn import fit_sklearn, compare

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

COLOR_0 = "#4472C4"
COLOR_1 = "#ED7D31"
LAM = 1.0


def savefig(name):
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / name)
    plt.close()


X, y = data_load()
X_train, X_test, y_train, y_test = split_and_scale(X, y)

print(f"пропуски в данных: {X.isna().sum().sum()}")
print(f"malignant (0): {(y == 0).sum()}, benign (1): {(y == 1).sum()}")

counts = [int((y == 0).sum()), int((y == 1).sum())]
plt.figure(figsize=(5, 4))
bars = plt.bar(["malignant", "benign"], counts, color=[COLOR_0, COLOR_1])
for b, c in zip(bars, counts):
    plt.text(b.get_x() + b.get_width() / 2, c + 3, str(c), ha="center")
plt.ylabel("количество объектов")
plt.title("Баланс классов")
savefig("00_class_balance.png")

mean_cols = [c for c in X.columns if c.startswith("mean ")]

plt.figure(figsize=(8, 7))
sns.heatmap(X[mean_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            cbar_kws={"label": "коэффициент корреляции"})
plt.title("Корреляция признаков (группа mean)")
savefig("01_corr_heatmap.png")

plt.figure(figsize=(8, 5))
sns.boxplot(data=X[mean_cols], color=COLOR_0)
plt.xticks(rotation=30, ha="right")
plt.ylabel("значение признака")
plt.title("Масштабы признаков до стандартизации")
savefig("02_feature_scales.png")

w, history = fit(X_train, y_train, lam=LAM)

plt.figure(figsize=(6, 4))
plt.plot(range(len(history)), history, marker="o", color=COLOR_0)
plt.yscale("log")
plt.xlabel("итерация")
plt.ylabel("log-loss (train)")
plt.title(f"Сходимость Ньютона-Рафсона / IRLS, lambda={LAM}")
savefig("03_convergence.png")

print(f"сошлось за {len(history) - 1} итераций, log-loss: {history[0]:.4f} -> {history[-1]:.4f}")

Xb_train = add_bias(X_train)
w_check = newton_step(Xb_train, y_train, np.zeros(Xb_train.shape[1]), LAM)
w_newton2 = newton_step(Xb_train, y_train, w_check, LAM)
w_irls2 = irls_step(Xb_train, y_train, w_check, LAM)
eq_diff = np.max(np.abs(w_newton2 - w_irls2))
print(f"эквивалентность Ньютона и IRLS на одном шаге: max|w_newton - w_irls| = {eq_diff:.2e}")

norms_reg = fit_with_norms(X_train, y_train, lam=LAM, max_iter=15)
norms_noreg = fit_with_norms(X_train, y_train, lam=0.0, max_iter=15)

plt.figure(figsize=(6, 4))
plt.plot(range(len(norms_noreg)), norms_noreg, marker="o", color=COLOR_1, label="lambda=0")
plt.plot(range(len(norms_reg)), norms_reg, marker="o", color=COLOR_0, label=f"lambda={LAM}")
plt.xlabel("итерация")
plt.ylabel("норма весов")
plt.title("Норма весов: регуляризованная модель vs без регуляризации")
plt.legend()
savefig("04_weight_norm_regularized_vs_not.png")

print(f"без регуляризации остановлено на {len(norms_noreg) - 1} итерации, норма весов {norms_noreg[-1]:.2f}")
print(f"с регуляризацией норма весов к концу: {norms_reg[-1]:.2f}")

skl = fit_sklearn(X_train, y_train, LAM)
result = compare(w, skl, X_test, y_test)

print(f"\nmax|w_mine - w_sklearn| = {result['coef_diff']:.2e}")
print(f"max|p_mine - p_sklearn| (test) = {result['prob_diff']:.2e}")
print(f"accuracy test: своя {result['acc_mine']:.4f}, sklearn {result['acc_skl']:.4f}")
print(f"log-loss test: своя {result['logloss_mine']:.4f}, sklearn {result['logloss_skl']:.4f}")

pca = PCA(n_components=2, random_state=42)
Z_train = pca.fit_transform(X_train)

w_2d, _ = fit(Z_train, y_train, lam=LAM)
skl_2d = fit_sklearn(Z_train, y_train, LAM)

x_min, x_max = Z_train[:, 0].min() - 1, Z_train[:, 0].max() + 1
y_min, y_max = Z_train[:, 1].min() - 1, Z_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.column_stack([xx.ravel(), yy.ravel()])

p_grid_mine = sigmoid(add_bias(grid) @ w_2d).reshape(xx.shape)
p_grid_skl = skl_2d.predict_proba(grid)[:, 1].reshape(xx.shape)

plt.figure(figsize=(7, 6))
plt.scatter(Z_train[y_train == 0, 0], Z_train[y_train == 0, 1], color=COLOR_0, label="malignant", s=14)
plt.scatter(Z_train[y_train == 1, 0], Z_train[y_train == 1, 1], color=COLOR_1, label="benign", s=14)
plt.contour(xx, yy, p_grid_mine, levels=[0.5], colors="black", linewidths=2)
plt.contour(xx, yy, p_grid_skl, levels=[0.5], colors="gray", linestyles="dashed", linewidths=2)
plt.xlabel("1-я главная компонента")
plt.ylabel("2-я главная компонента")
plt.legend()
plt.title("Решающая граница: своя реализация vs sklearn (PCA-проекция)")
savefig("05_decision_boundary.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, cm, name, acc in [
    (axes[0], result["cm_mine"], "своя реализация", result["acc_mine"]),
    (axes[1], result["cm_skl"], "sklearn", result["acc_skl"]),
]:
    cm_frac = cm / cm.sum()
    annot = np.array([[f"{cm[i, j]}\n{cm_frac[i, j]:.2f}" for j in range(2)] for i in range(2)])
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False,
                xticklabels=["malignant", "benign"], yticklabels=["malignant", "benign"], ax=ax)
    ax.set_xlabel("предсказанный класс")
    ax.set_ylabel("истинный класс")
    ax.set_title(f"{name}, accuracy={acc:.3f}")
savefig("06_confusion_matrices.png")

feature_names = list(X.columns)
order = np.argsort(-np.abs(w[1:]))[:10]
top_names = [feature_names[i] for i in order]
top_mine = w[1:][order]
top_skl = result["w_skl"][1:][order]

ypos = np.arange(len(top_names))
plt.figure(figsize=(9, 6))
plt.barh(ypos - 0.2, top_mine, height=0.4, color=COLOR_0, label="своя реализация")
plt.barh(ypos + 0.2, top_skl, height=0.4, color=COLOR_1, label="sklearn")
plt.yticks(ypos, top_names)
plt.gca().invert_yaxis()
plt.xlabel("значение коэффициента")
plt.legend(loc="upper right")
plt.title("Коэффициенты логрег: своя реализация vs sklearn (топ-10 по модулю)")
savefig("07_compare_coefficients.png")

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
acc_dummy = accuracy_score(y_test, dummy.predict(X_test))
print(f"\nbaseline (DummyClassifier, most_frequent) accuracy на test: {acc_dummy:.4f}")
print(f"своя реализация accuracy: {result['acc_mine']:.4f}, sklearn: {result['acc_skl']:.4f}")
