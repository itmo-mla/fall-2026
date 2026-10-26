from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve

from students.gopienko_ad.lab5.source.logistic_regression import LogisticRegressionIRLS


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def plot_loss(models: dict, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for name in ("newton", "irls"):
        model = models[name]
        plt.plot(range(len(model.loss_history_)), model.loss_history_, marker="o", label=f"{name} ({model.n_iter_} it.)")
    plt.xlabel("Iteration")
    plt.ylabel("Mean training log-loss")
    plt.title("Convergence: Newton–Raphson vs IRLS")
    plt.legend()
    _save(output_path)


def plot_roc(models: dict, X: np.ndarray, y: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(7, 6))
    for name, model in models.items():
        fpr, tpr, _ = roc_curve(y, model.predict_proba(X)[:, 1])
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves on test data")
    plt.legend()
    _save(path)


def plot_confusion(models: dict, X: np.ndarray, y: np.ndarray, path: Path) -> None:
    names = list(models)
    fig, axs = plt.subplots(1, len(names), figsize=(12, 4))
    for ax, name in zip(axs, names):
        pred = models[name].predict(X)
        cm = np.array([[np.sum((y == t) & (pred == p)) for p in (0, 1)] for t in (0, 1)])
        ax.imshow(cm, cmap="Blues")
        for i in (0, 1):
            for j in (0, 1):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13)
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="True", title=name)
    _save(path)


def plot_coefficients(models: dict, feature_names: list[str], path: Path) -> None:
    x = np.arange(len(feature_names))
    plt.figure(figsize=(10, 5))
    width = .26
    for i, (name, model) in enumerate(models.items()):
        plt.bar(x + (i-1)*width, model.coef_.ravel(), width=width, label=name)
    plt.axhline(0.0, color="black", linewidth=.6)
    plt.xticks(x, feature_names)
    plt.xlabel("Standardized feature")
    plt.ylabel("Coefficient")
    plt.title("Logistic regression coefficients")
    plt.legend()
    _save(path)


def plot_probability_hist(models: dict, X: np.ndarray, y: np.ndarray, path: Path) -> None:
    p = models["irls"].predict_proba(X)[:, 1]
    plt.figure(figsize=(8, 5))
    plt.hist(p[y == 0], bins=16, alpha=.6, label="True class 0")
    plt.hist(p[y == 1], bins=16, alpha=.6, label="True class 1")
    plt.axvline(.5, linestyle="--", color="black", label="Threshold 0.5")
    plt.xlabel("P(y=1 | x)")
    plt.ylabel("Number of objects")
    plt.title("Test predicted probabilities (IRLS)")
    plt.legend()
    _save(path)


def plot_pca_2d_boundary(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    Z = PCA(n_components=2).fit_transform(X)
    clf = LogisticRegressionIRLS(tol=1e-8).fit(Z, y)
    x1 = np.linspace(Z[:, 0].min()-.8, Z[:, 0].max()+.8, 220)
    x2 = np.linspace(Z[:, 1].min()-.8, Z[:, 1].max()+.8, 220)
    xx, yy = np.meshgrid(x1, x2)
    mesh = np.column_stack([xx.ravel(), yy.ravel()])
    prob = clf.predict_proba(mesh)[:, 1].reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, prob, levels=np.linspace(0, 1, 11), alpha=.35, cmap="RdBu_r")
    plt.contour(xx, yy, prob, levels=[.5], colors="black")
    plt.scatter(Z[:, 0], Z[:, 1], c=y, s=16, edgecolors="none", cmap="RdBu_r")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("IRLS decision boundary in PCA 2D (illustration only)")
    _save(path)