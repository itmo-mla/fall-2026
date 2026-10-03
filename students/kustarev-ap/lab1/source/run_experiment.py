import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay,
)

from data import load_dataset, load_raw, TARGET_COL, HIGH_RISK_CATEGORIES
from linear_classifier import LinearClassifier, margin, loss, loss_grad
from metrics import evaluate_classifier, results_table, pm1_to_01

np.random.seed(42)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def savefig(name):
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / name)
    plt.close()


# Этап 0. Данные

df_raw = load_raw()
print("Размер датасета:", df_raw.shape)

counts = df_raw[TARGET_COL].value_counts()
fig, ax = plt.subplots(figsize=(6, 3.5))
colors = ["tab:red" if c in HIGH_RISK_CATEGORIES else "tab:blue" for c in counts.index]
ax.bar(counts.index, counts.values, color=colors)
ax.set_ylabel("количество клиентов")
ax.set_title("churn_risk_category (красный = повышенный риск -> класс 1)")
plt.xticks(rotation=20)
savefig("00_target_distribution.png")

X_train, X_test, y_train, y_test, preprocessor = load_dataset()
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Доля класса 1 (высокий риск) train: {(y_train == 1).mean():.3f}, test: {(y_test == 1).mean():.3f}")
print(f"Число признаков после OneHot/StandardScaler: {X_train.shape[1]}")

# PCA-проекция классов

pca = PCA(n_components=2, random_state=42)
X_train_2d = pca.fit_transform(X_train)

fig, ax = plt.subplots(figsize=(7, 5))
for cls, color, label in [(-1.0, "tab:blue", "низкий риск"), (1.0, "tab:red", "повышенный риск")]:
    mask = y_train == cls
    ax.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1], s=5, alpha=0.3, color=color, label=label)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Проекция train-выборки на первые 2 главные компоненты")
ax.legend()
savefig("01_pca.png")

print(f"Объяснённая дисперсия PCA: PC1={pca.explained_variance_ratio_[0]:.3f}, "
      f"PC2={pca.explained_variance_ratio_[1]:.3f}, сумма={pca.explained_variance_ratio_.sum():.3f}")

# Этап 1. Отступ объекта (п.2)

d = X_train.shape[1]
rng = np.random.default_rng(0)
w_random = rng.normal(scale=0.01, size=d + 1)
M_before = margin(w_random, X_train, y_train)

clf_demo = LinearClassifier(
    init="correlation", step_strategy="fixed", sampling_strategy="uniform",
    eta=0.005, gamma=0.9, l2_lambda=1e-2, max_iter=20000, random_state=42,
)
clf_demo.fit(X_train, y_train)
M_after = clf_demo.margin(X_train, y_train)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(M_before, bins=60, color="tab:gray")
axes[0].axvline(0, color="red", ls="--")
axes[0].set_title("Отступы ДО обучения (случайные веса)")
axes[0].set_xlabel("M_i"); axes[0].set_ylabel("число объектов")

axes[1].hist(M_after, bins=60, color="tab:blue")
axes[1].axvline(0, color="red", ls="--")
axes[1].set_title("Отступы ПОСЛЕ обучения")
axes[1].set_xlabel("M_i")
savefig("02_margin_before_after.png")

share_neg_before = (M_before < 0).mean()
share_neg_after = (M_after < 0).mean()
share_border_after = (np.abs(M_after) < 0.5).mean()
print(f"Доля объектов с M<0 ДО обучения:    {share_neg_before:.3f}")
print(f"Доля объектов с M<0 ПОСЛЕ обучения:  {share_neg_after:.3f}")
print(f"Доля пограничных объектов |M|<0.5:   {share_border_after:.3f}")

# Этап 4-5. SGD с инерцией и L2-регуляризация (п.5, п.6)

clf_l2_0 = LinearClassifier(init="correlation", step_strategy="fixed", sampling_strategy="uniform",
                             eta=0.005, gamma=0.9, l2_lambda=0.0, max_iter=20000, random_state=42)
clf_l2_0.fit(X_train, y_train)

clf_l2_big = LinearClassifier(init="correlation", step_strategy="fixed", sampling_strategy="uniform",
                               eta=0.005, gamma=0.9, l2_lambda=5.0, max_iter=20000, random_state=42)
clf_l2_big.fit(X_train, y_train)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(clf_l2_0.history_.w_norm, label="l2_lambda=0")
axes[0].plot(clf_l2_big.history_.w_norm, label="l2_lambda=5.0")
axes[0].set_title("Норма весов по итерациям")
axes[0].set_xlabel("итерация SGD"); axes[0].legend()

axes[1].plot(clf_l2_0.history_.Q, label="l2_lambda=0")
axes[1].plot(clf_l2_big.history_.Q, label="l2_lambda=5.0")
axes[1].set_yscale("log")
axes[1].set_title("Сходимость Q")
axes[1].set_xlabel("итерация SGD"); axes[1].legend()
savefig("03_l2_effect.png")

print(f"{'l2_lambda':>10} | {'norm(w)':>8} | {'train loss':>11} | {'test loss':>10}")
for name, clf in [("0.0", clf_l2_0), ("5.0", clf_l2_big)]:
    M_tr = clf.margin(X_train, y_train)
    M_te = clf.margin(X_test, y_test)
    print(f"{name:>10} | {np.linalg.norm(clf.w_):8.3f} | {loss(M_tr).mean():11.4f} | {loss(M_te).mean():10.4f}")

# Этап 6. Скорейший градиентный спуск (п.7)

common = dict(init="correlation", sampling_strategy="uniform", gamma=0.9, l2_lambda=1e-2,
              max_iter=20000, random_state=42)
# eta* выводится из полного градиента (loss + L2): eta*=||g||^2/(g^T H g),
# поэтому корректно работает вместе с регуляризацией

clf_fixed = LinearClassifier(step_strategy="fixed", eta=0.005, **common)
clf_fixed.fit(X_train, y_train)

clf_steep = LinearClassifier(step_strategy="steepest", **common)
clf_steep.fit(X_train, y_train)

plt.figure(figsize=(7, 4))
plt.plot(clf_fixed.history_.Q, label="fixed, eta=0.005")
plt.plot(clf_steep.history_.Q, label="steepest, eta*=||g||^2/(g^T H g)")
plt.yscale("log")
plt.xlabel("итерация SGD"); plt.ylabel("Q")
plt.title("Сходимость: fixed vs steepest")
plt.legend()
savefig("04_fixed_vs_steepest.png")

res_step = {
    "fixed": evaluate_classifier(clf_fixed, X_test, y_test),
    "steepest": evaluate_classifier(clf_steep, X_test, y_test),
}
print(results_table(res_step))

n = X_train.shape[0]
X_aug = np.hstack([X_train, np.ones((n, 1))])
x_i = X_aug[0]
w0 = np.zeros(x_i.shape[0])
grad0 = loss_grad(w0, x_i, y_train[0])
eta_star = LinearClassifier.steepest_eta(x_i, grad0, l2_lambda=0.0)
w1 = w0 - eta_star * grad0
M1 = y_train[0] * (x_i @ w1)
print(f"eta* (l2=0) = {eta_star:.5f}, отступ после шага M1 = {M1:.6f} (должен быть ~1.0)")

# Этап 7. Предъявление объектов по модулю отступа (п.8)

common2 = dict(init="correlation", step_strategy="fixed", eta=0.005, gamma=0.9, l2_lambda=1e-2,
               max_iter=20000, margin_recompute_every=100, random_state=42)

clf_unif = LinearClassifier(sampling_strategy="uniform", **common2)
clf_unif.fit(X_train, y_train)

clf_marg = LinearClassifier(sampling_strategy="margin", **common2)
clf_marg.fit(X_train, y_train)

plt.figure(figsize=(7, 4))
plt.plot(clf_unif.history_.Q, label="uniform sampling")
plt.plot(clf_marg.history_.Q, label="margin sampling")
plt.yscale("log")
plt.xlabel("итерация SGD"); plt.ylabel("Q")
plt.title("Сходимость: uniform vs margin sampling")
plt.legend()
savefig("05_uniform_vs_margin.png")

res_sampling = {
    "uniform": evaluate_classifier(clf_unif, X_test, y_test),
    "margin": evaluate_classifier(clf_marg, X_test, y_test),
}
print(results_table(res_sampling))

# Этап 8. Три режима обучения (п.9)

base = dict(gamma=0.9, l2_lambda=1e-2, eta=0.005, max_iter=20000,
            margin_recompute_every=100, random_state=42)

t0 = time.time()
clf_91 = LinearClassifier(init="correlation", step_strategy="fixed", sampling_strategy="uniform", **base)
clf_91.fit(X_train, y_train)
t1 = time.time()

clf_92 = LinearClassifier(init="random", n_starts=10, step_strategy="fixed", sampling_strategy="uniform", **base)
clf_92.fit(X_train, y_train)
t2 = time.time()

clf_93 = LinearClassifier(init="random", step_strategy="fixed", sampling_strategy="margin", **base)
clf_93.fit(X_train, y_train)
t3 = time.time()

modes = {
    "9.1 corr-init + uniform": clf_91,
    "9.2 random + multistart(10)": clf_92,
    "9.3 random + margin-sampling": clf_93,
}

print(f"Время обучения: 9.1={t1 - t0:.2f}s, 9.2={t2 - t1:.2f}s (10 стартов), 9.3={t3 - t2:.2f}s")
for name, clf in modes.items():
    print(f"{name:32s} n_iter={clf.history_.n_iter:6d}  Q_final={clf.history_.Q[-1]:.4f}")

# Этап 9. Оценка качества (п.10)

train_res = {name: evaluate_classifier(clf, X_train, y_train) for name, clf in modes.items()}
test_res = {name: evaluate_classifier(clf, X_test, y_test) for name, clf in modes.items()}

print("=== TRAIN ===")
print(results_table(train_res))
print("=== TEST ===")
print(results_table(test_res))

fig, ax = plt.subplots(figsize=(8, 5))
for name, clf in modes.items():
    ax.plot(clf.history_.Q, label=name)
ax.set_yscale("log")
ax.set_xlabel("итерация SGD")
ax.set_ylabel("Q")
ax.set_title("Сходимость Q: сравнение трёх режимов обучения (п.9)")
ax.legend()
savefig("06_three_modes_convergence.png")

# Этап 10. Сравнение с эталоном (п.11)

y01_train = pm1_to_01(y_train)
y01_test = pm1_to_01(y_test)

dummy = DummyClassifier(strategy="most_frequent", random_state=42)
dummy.fit(X_train, y01_train)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y01_train)


def eval_sklearn(model, X, y01):
    pred = model.predict(X)
    scores = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else pred
    return {
        "accuracy": accuracy_score(y01, pred),
        "precision": precision_score(y01, pred, zero_division=0),
        "recall": recall_score(y01, pred, zero_division=0),
        "f1": f1_score(y01, pred, zero_division=0),
        "roc_auc": roc_auc_score(y01, scores),
    }


best_mode_name = max(test_res, key=lambda k: test_res[k]["roc_auc"])

final_results = {
    "DummyClassifier": eval_sklearn(dummy, X_test, y01_test),
    "LogisticRegression (эталон)": eval_sklearn(logreg, X_test, y01_test),
    **{f"LinearClassifier: {name}": res for name, res in test_res.items()},
}

print(f"Лучший собственный режим: {best_mode_name}")
print(results_table(final_results))

best_clf = modes[best_mode_name]
best_scores = best_clf.decision_function(X_test)
best_pred01 = pm1_to_01(best_clf.predict(X_test))
log_scores = logreg.predict_proba(X_test)[:, 1]

fpr_best, tpr_best, _ = roc_curve(y01_test, best_scores)
fpr_log, tpr_log, _ = roc_curve(y01_test, log_scores)

plt.figure(figsize=(6, 6))
plt.plot(fpr_best, tpr_best, label=f"{best_mode_name}, AUC={roc_auc_score(y01_test, best_scores):.3f}")
plt.plot(fpr_log, tpr_log, label=f"LogisticRegression, AUC={roc_auc_score(y01_test, log_scores):.3f}")
plt.plot([0, 1], [0, 1], ls="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривая: лучший режим против эталона")
plt.legend()
savefig("07_roc_curve.png")

fig, ax = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay.from_predictions(
    y01_test, best_pred01, display_labels=["низкий риск", "повышенный риск"],
    cmap="Blues", colorbar=False, ax=ax,
)
ax.set_title(f"Матрица ошибок: режим {best_mode_name.split()[0]}")
savefig("08_confusion_matrix.png")

print(f"Графики сохранены в {PLOTS_DIR}")
