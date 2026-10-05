from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

from data import data_load
from knn import ParzenKNN, epanechnikov_kernel
from loo import loo_errors
from stolp import greedy_add, greedy_remove, margins

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def savefig(name):
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / name)
    plt.close()


X, y = data_load()

plt.figure(figsize=(6, 4))
sns.countplot(x=y, order=np.unique(y))
savefig("00_class_distribution.png")

plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm")
savefig("01_corr_heatmap.png")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

ks = np.arange(1, 31)
errors = loo_errors(X_train, y_train, ks)
best_k = ks[np.argmin(errors)]

plt.figure(figsize=(8, 5))
plt.plot(ks, errors, marker="o")
plt.axvline(best_k, color="red", linestyle="--")
plt.xlabel("k")
plt.ylabel("LOO error")
savefig("02_loo_errors.png")

errors_epanechnikov = loo_errors(X_train, y_train, ks, kernel=epanechnikov_kernel)
# k=1 вырожденный случай для ядра Епанечникова: единственный сосед лежит ровно на
# границе окна (расстояние = h), вес там точно 0, веса по всем классам нулевые
best_k_epanechnikov = ks[1:][np.argmin(errors_epanechnikov[1:])]

plt.figure(figsize=(8, 5))
plt.plot(ks[1:], errors[1:], marker="o", label="гауссово")
plt.plot(ks[1:], errors_epanechnikov[1:], marker="o", label="Епанечникова")
plt.axvline(best_k_epanechnikov, color="red", linestyle="--")
plt.xlabel("k")
plt.ylabel("LOO error")
plt.legend()
savefig("02b_loo_errors_kernels.png")

clf = ParzenKNN(k=best_k)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc_parzen = accuracy_score(y_test, y_pred)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
savefig("03_confusion_matrix.png")

pca = PCA(n_components=2)
X_2d_train = pca.fit_transform(X_train)
X_2d_test = pca.transform(X_test)

clf_2d = ParzenKNN(k=best_k)
clf_2d.fit(X_2d_train, y_train)

sk_clf_2d = KNeighborsClassifier(n_neighbors=best_k)
sk_clf_2d.fit(X_2d_train, y_train)

x_min, x_max = X_2d_train[:, 0].min() - 1, X_2d_train[:, 0].max() + 1
y_min, y_max = X_2d_train[:, 1].min() - 1, X_2d_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

classes = np.unique(y_train)
grid_codes = np.searchsorted(classes, clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
grid_codes_sk = np.searchsorted(classes, sk_clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
test_codes = np.searchsorted(classes, y_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].contourf(xx, yy, grid_codes, alpha=0.3, cmap="viridis")
axes[0].scatter(X_2d_test[:, 0], X_2d_test[:, 1], c=test_codes, cmap="viridis", edgecolor="k")
axes[0].set_title("ParzenKNN")
axes[0].set_xlabel("PCA 1")
axes[0].set_ylabel("PCA 2")

axes[1].contourf(xx, yy, grid_codes_sk, alpha=0.3, cmap="viridis")
axes[1].scatter(X_2d_test[:, 0], X_2d_test[:, 1], c=test_codes, cmap="viridis", edgecolor="k")
axes[1].set_title("sklearn KNeighborsClassifier")
axes[1].set_xlabel("PCA 1")
axes[1].set_ylabel("PCA 2")
savefig("04_decision_boundary_pca.png")

sk_clf = KNeighborsClassifier(n_neighbors=best_k)
sk_clf.fit(X_train, y_train)
y_pred_sk = sk_clf.predict(X_test)
acc_sklearn = accuracy_score(y_test, y_pred_sk)

add_idx, add_history = greedy_add(X_train, y_train, best_k)
remove_idx, remove_history = greedy_remove(X_train, y_train, best_k)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(add_history, marker="o")
axes[0].set_xlabel("число добавленных эталонов")
axes[0].set_ylabel("CCV (число ошибок)")
axes[0].set_title("жадное добавление")
axes[1].plot(remove_history, marker="o")
axes[1].set_xlabel("число удаленных объектов")
axes[1].set_ylabel("CCV (число ошибок)")
axes[1].set_title("жадное удаление")
savefig("05_ccv_history.png")

train_codes = np.searchsorted(classes, y_train)
noise_mask = margins(X_train, y_train, best_k) < 0


def plot_selection(ax, ref_idx, title):
    ref_mask = np.zeros(len(X_train), dtype=bool)
    ref_mask[ref_idx] = True
    uninformative_mask = ~ref_mask & ~noise_mask

    ax.scatter(X_2d_train[uninformative_mask, 0], X_2d_train[uninformative_mask, 1],
               c="lightgray", s=10, alpha=0.4, label="неинформативные")
    ax.scatter(X_2d_train[noise_mask, 0], X_2d_train[noise_mask, 1],
               c=train_codes[noise_mask], cmap="viridis", vmin=0, vmax=len(classes) - 1,
               marker="^", s=50, edgecolor="k", label="шумовые")
    ax.scatter(X_2d_train[ref_mask, 0], X_2d_train[ref_mask, 1],
               c=train_codes[ref_mask], cmap="viridis", vmin=0, vmax=len(classes) - 1,
               marker="o", s=90, edgecolor="k", label="эталоны")
    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend()


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_selection(axes[0], add_idx, "жадное добавление")
plot_selection(axes[1], remove_idx, "жадное удаление")
savefig("06_stolp_pca.png")

k_add = min(best_k, len(add_idx))
k_remove = min(best_k, len(remove_idx))
acc_add = accuracy_score(y_test, ParzenKNN(k=k_add).fit(X_train[add_idx], y_train[add_idx]).predict(X_test))
acc_remove = accuracy_score(y_test, ParzenKNN(k=k_remove).fit(X_train[remove_idx], y_train[remove_idx]).predict(X_test))

print("best_k:", best_k, "LOO error:", errors.min())
print("best_k (Епанечникова, без k=1):", best_k_epanechnikov, "LOO error:", errors_epanechnikov[1:].min())
print("ParzenKNN accuracy:", acc_parzen)
print("sklearn KNN accuracy:", acc_sklearn)
print(f"жадное добавление: {len(add_idx)} из {len(X_train)}, accuracy {acc_add}")
print(f"жадное удаление: {len(remove_idx)} из {len(X_train)}, accuracy {acc_remove}")
print(f"Графики сохранены в {PLOTS_DIR}")
