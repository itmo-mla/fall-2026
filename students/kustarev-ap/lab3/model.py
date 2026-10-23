import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from data import data_load, split_and_scale, FEATURE_NAMES
from kernels import linear, polynomial, rbf
from svm_dual import DualSVM, to_primal_weights, kkt_report
from cross_val import grid_search

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

COLOR_NEG = "#4472C4"
COLOR_POS = "#ED7D31"
CLASS_TICKS = ["поддельная", "подлинная"]
TAB10 = plt.get_cmap("tab10").colors


def savefig(name):
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / name)
    plt.close()


def class_colors(y):
    return [COLOR_NEG if v < 0 else COLOR_POS for v in y]


def fit_timed(kernel, C, X, y, **params):
    t0 = time.perf_counter()
    model = DualSVM(kernel, C=C, **params).fit(X, y)
    dt = time.perf_counter() - t0
    return model, dt


def plot_decision_regions(ax, model, X2, y2, title):
    x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
    y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    zz = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    ax.contourf(xx, yy, zz, levels=[-1e9, 0, 1e9], colors=[COLOR_NEG, COLOR_POS], alpha=0.18)
    ax.contour(xx, yy, zz, levels=[0], colors="black", linewidths=1.5)
    ax.contour(xx, yy, zz, levels=[-1, 1], colors="gray", linestyles="--", linewidths=1.0)

    ax.scatter(X2[:, 0], X2[:, 1], c=class_colors(y2), s=14, edgecolor="none", zorder=2)
    margin, violator = model.margin_idx, model.violator_idx
    ax.scatter(X2[margin, 0], X2[margin, 1], c=class_colors(y2[margin]), marker="o",
               s=130, edgecolor="black", linewidth=1.8, zorder=3)
    ax.scatter(X2[violator, 0], X2[violator, 1], c=class_colors(y2[violator]), marker="^",
               s=150, edgecolor="black", linewidth=1.8, zorder=3)

    ax.set_xlabel("1-я главная компонента")
    ax.set_ylabel("2-я главная компонента")
    ax.set_title(title)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_NEG, markersize=8, label="класс -1"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_POS, markersize=8, label="класс +1"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="black",
               markeredgewidth=1.8, markersize=13, label=f"опорно-граничные ({len(margin)} шт.)"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="none", markeredgecolor="black",
               markeredgewidth=1.8, markersize=13, label=f"опорно-нарушители ({len(violator)} шт.)"),
        Line2D([0], [0], color="black", linewidth=1.5, label="граница a(x)=0"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=1.0, label="граница зазора a(x)=+-1"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="best")


def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    total = cm.sum()
    annot = np.array([[f"{cm[i, j]}\n({100 * cm[i, j] / total:.0f}%)" for j in range(2)] for i in range(2)])
    acc = accuracy_score(y_true, y_pred)

    plt.figure(figsize=(5, 4.3))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=True,
                xticklabels=CLASS_TICKS, yticklabels=CLASS_TICKS)
    plt.xlabel("Предсказанный класс")
    plt.ylabel("Истинный класс")
    plt.title(f"{title}, accuracy={acc:.3f}")
    savefig(filename)


X, y = data_load()
X_train, X_test, y_train, y_test, X_sub, y_sub = split_and_scale(X, y)

counts = [int(np.sum(y == -1)), int(np.sum(y == 1))]
plt.figure(figsize=(5, 4))
bars = plt.bar(CLASS_TICKS, counts, color=[COLOR_NEG, COLOR_POS])
for b, c in zip(bars, counts):
    plt.text(b.get_x() + b.get_width() / 2, c + 10, str(c), ha="center")
plt.ylabel("количество объектов")
plt.title("Баланс классов")
savefig("00_class_distribution.png")

plt.figure(figsize=(6, 5))
sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
            cbar_kws={"label": "коэффициент корреляции"})
plt.title("Корреляция признаков")
savefig("01_corr_heatmap.png")

print(f"классы: поддельная {counts[0]}, подлинная {counts[1]}")
print(f"train={len(y_train)}, test={len(y_test)}, train_sub={len(y_sub)}")

linear_grid = [{"C": c} for c in np.logspace(-2, 2, 7)]
linear_results = grid_search(linear, linear_grid, X_sub, y_sub)
best_linear = max(linear_results, key=lambda r: r["accuracy"])

Cs = [r["C"] for r in linear_results]
plt.figure(figsize=(7, 5))
plt.plot(Cs, [r["accuracy"] for r in linear_results], marker="o", color=TAB10[0], label="accuracy")
plt.plot(Cs, [r["f1"] for r in linear_results], marker="o", linestyle="--", color=TAB10[0], label="F1")
plt.axvline(best_linear["C"], color="gray", linestyle="--", label=f"выбранное C={best_linear['C']:.3g}")
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("значение метрики")
plt.title("CV: линейное ядро")
plt.grid(alpha=0.3)
plt.legend()
savefig("02_cv_linear.png")

# coef0 зафиксирован в 1.0, по сетке ищутся только C и degree
poly_grid = [{"C": c, "degree": d, "coef0": 1.0} for c in [0.1, 1, 10, 30, 100] for d in [2, 3]]
poly_results = grid_search(polynomial, poly_grid, X_sub, y_sub)
best_poly = max(poly_results, key=lambda r: r["accuracy"])

plt.figure(figsize=(7, 5))
for i, deg in enumerate([2, 3]):
    rows = sorted([r for r in poly_results if r["degree"] == deg], key=lambda r: r["C"])
    plt.plot([r["C"] for r in rows], [r["accuracy"] for r in rows], marker="o",
             color=TAB10[i], label=f"degree={deg}, coef0=1")
plt.scatter([best_poly["C"]], [best_poly["accuracy"]], s=160, facecolor="none",
            edgecolor="black", linewidth=1.5, zorder=5, label="лучшая комбинация")
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("CV accuracy")
plt.title("CV: полиномиальное ядро")
plt.grid(alpha=0.3)
plt.legend()
savefig("03_cv_poly.png")

# сетка gamma вокруг 1/n_features (эвристика sklearn 'scale' при std-признаках с var=1)
gamma_scale = 1.0 / X_sub.shape[1]
rbf_grid = [{"C": c, "gamma": g} for c in [0.1, 1, 10, 100]
            for g in [gamma_scale / 4, gamma_scale / 2, gamma_scale, gamma_scale * 2, gamma_scale * 4, gamma_scale * 8]]
rbf_results = grid_search(rbf, rbf_grid, X_sub, y_sub)
best_rbf = max(rbf_results, key=lambda r: r["accuracy"])

rbf_Cs = sorted(set(r["C"] for r in rbf_grid))
rbf_gammas = sorted(set(r["gamma"] for r in rbf_grid))
acc_grid = np.zeros((len(rbf_Cs), len(rbf_gammas)))
for r in rbf_results:
    acc_grid[rbf_Cs.index(r["C"]), rbf_gammas.index(r["gamma"])] = r["accuracy"]

plt.figure(figsize=(7.5, 5))
sns.heatmap(acc_grid, cmap="viridis", cbar_kws={"label": "CV accuracy"},
            xticklabels=[f"{g:.3g}" for g in rbf_gammas], yticklabels=[f"{c:.3g}" for c in rbf_Cs])
bi, bj = rbf_Cs.index(best_rbf["C"]), rbf_gammas.index(best_rbf["gamma"])
plt.scatter(bj + 0.5, bi + 0.5, marker="*", color="red", s=220)
plt.text(bj + 0.5, bi + 0.25, f"{best_rbf['accuracy']:.3f}", color="white", ha="center", fontsize=8)
plt.xlabel("gamma")
plt.ylabel("C")
plt.title("CV: RBF-ядро")
savefig("04_cv_rbf.png")

print("лучшие гиперпараметры:")
print("linear:", best_linear)
print("poly:", best_poly)
print("rbf:", best_rbf)

model_linear, time_linear = fit_timed(linear, best_linear["C"], X_sub, y_sub)
model_poly, time_poly = fit_timed(polynomial, best_poly["C"], X_sub, y_sub,
                                   degree=best_poly["degree"], coef0=best_poly["coef0"])
model_rbf, time_rbf = fit_timed(rbf, best_rbf["C"], X_sub, y_sub, gamma=best_rbf["gamma"])
models = {"linear": model_linear, "poly": model_poly, "rbf": model_rbf}
times = {"linear": time_linear, "poly": time_poly, "rbf": time_rbf}

print("KKT-проверка:")
for name, m in models.items():
    print(name, kkt_report(m))

w, w0 = to_primal_weights(model_linear)
a_primal = X_test @ w - w0
a_dual = model_linear.decision_function(X_test)
sanity_max_diff = float(np.max(np.abs(a_primal - a_dual)))
sanity_sign_match = float(np.mean(np.sign(a_primal) == np.sign(a_dual))) * 100
print("sanity-check прямая/двойственная: max diff", sanity_max_diff, "sign match %", sanity_sign_match)

print("веса линейного классификатора:")
for name, wj in zip(FEATURE_NAMES, w):
    print(name, wj, abs(wj))

pca = PCA(n_components=2)
X_sub_2d = pca.fit_transform(X_sub)

model_linear_2d, _ = fit_timed(linear, best_linear["C"], X_sub_2d, y_sub)
model_poly_2d, _ = fit_timed(polynomial, best_poly["C"], X_sub_2d, y_sub,
                              degree=best_poly["degree"], coef0=best_poly["coef0"])
model_rbf_2d, _ = fit_timed(rbf, best_rbf["C"], X_sub_2d, y_sub, gamma=best_rbf["gamma"])
models_2d = {"linear": model_linear_2d, "poly": model_poly_2d, "rbf": model_rbf_2d}

titles = {
    "linear": f"Линейное ядро, C={best_linear['C']:.3g}",
    "poly": f"Полиномиальное ядро, C={best_poly['C']:.3g}, degree={best_poly['degree']}",
    "rbf": f"RBF-ядро, C={best_rbf['C']:.3g}, gamma={best_rbf['gamma']:.3g}",
}
filenames_2d = {"linear": "05_decision_boundary_linear_pca.png",
                "poly": "06_decision_boundary_poly_pca.png",
                "rbf": "07_decision_boundary_rbf_pca.png"}

for name, m2d in models_2d.items():
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    plot_decision_regions(ax, m2d, X_sub_2d, y_sub, titles[name])
    savefig(filenames_2d[name])

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharex=True, sharey=True)
fig.suptitle("Сравнение границ по ядрам")
for ax, name in zip(axes, ["linear", "poly", "rbf"]):
    plot_decision_regions(ax, models_2d[name], X_sub_2d, y_sub, titles[name])
savefig("08_decision_boundaries_compare.png")

y_pred_linear = model_linear.predict(X_test)
y_pred_poly = model_poly.predict(X_test)
y_pred_rbf = model_rbf.predict(X_test)

plot_confusion(y_test, y_pred_linear, titles["linear"], "09_confusion_matrix_linear.png")
plot_confusion(y_test, y_pred_poly, titles["poly"], "10_confusion_matrix_poly.png")
plot_confusion(y_test, y_pred_rbf, titles["rbf"], "11_confusion_matrix_rbf.png")

sklearn_params = {
    "linear": dict(kernel="linear", C=best_linear["C"]),
    "poly": dict(kernel="poly", C=best_poly["C"], degree=best_poly["degree"], coef0=best_poly["coef0"]),
    "rbf": dict(kernel="rbf", C=best_rbf["C"], gamma=best_rbf["gamma"]),
}

comparison_rows = []
sklearn_models = {}
own_preds = {"linear": y_pred_linear, "poly": y_pred_poly, "rbf": y_pred_rbf}
for name, own_model in models.items():
    y_pred_own = own_preds[name]
    comparison_rows.append({
        "kernel": name, "impl": "своя", "accuracy": accuracy_score(y_test, y_pred_own),
        "precision": precision_score(y_test, y_pred_own), "recall": recall_score(y_test, y_pred_own),
        "f1": f1_score(y_test, y_pred_own), "n_sv": len(own_model.sv_idx), "time": times[name],
    })

    t0 = time.perf_counter()
    sk = SVC(**sklearn_params[name]).fit(X_sub, y_sub)
    dt_sk = time.perf_counter() - t0
    sklearn_models[name] = sk
    y_pred_sk = sk.predict(X_test)
    comparison_rows.append({
        "kernel": name, "impl": "sklearn", "accuracy": accuracy_score(y_test, y_pred_sk),
        "precision": precision_score(y_test, y_pred_sk), "recall": recall_score(y_test, y_pred_sk),
        "f1": f1_score(y_test, y_pred_sk), "n_sv": int(sk.n_support_.sum()), "time": dt_sk,
    })

print("сравнение со sklearn:")
for row in comparison_rows:
    print(row)

best_sklearn_name = max(sklearn_models, key=lambda n: accuracy_score(y_test, sklearn_models[n].predict(X_test)))
y_pred_best_sklearn = sklearn_models[best_sklearn_name].predict(X_test)
plot_confusion(y_test, y_pred_best_sklearn, f"sklearn SVC, {best_sklearn_name}", "12_confusion_matrix_sklearn.png")

t0 = time.perf_counter()
sk_full = SVC(**sklearn_params[best_sklearn_name]).fit(X_train, y_train)
dt_sk_full = time.perf_counter() - t0
acc_sk_full = accuracy_score(y_test, sk_full.predict(X_test))
print(f"sklearn {best_sklearn_name} на полном train ({len(y_train)}): accuracy={acc_sk_full:.3f}, время={dt_sk_full:.2f}с")

dummy = DummyClassifier(strategy="most_frequent").fit(X_sub, y_sub)
y_pred_dummy = dummy.predict(X_test)

baseline_rows = [
    {"model": "DummyClassifier", "accuracy": accuracy_score(y_test, y_pred_dummy), "f1": f1_score(y_test, y_pred_dummy)},
    {"model": "линейный SVM", "accuracy": accuracy_score(y_test, y_pred_linear), "f1": f1_score(y_test, y_pred_linear)},
    {"model": "poly SVM", "accuracy": accuracy_score(y_test, y_pred_poly), "f1": f1_score(y_test, y_pred_poly)},
    {"model": "RBF SVM", "accuracy": accuracy_score(y_test, y_pred_rbf), "f1": f1_score(y_test, y_pred_rbf)},
]
print("baseline:")
for row in baseline_rows:
    print(row)

print(f"графики сохранены в {PLOTS_DIR}")
