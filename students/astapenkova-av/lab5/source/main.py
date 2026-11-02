import os
import time

from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, log_loss as sk_log_loss, roc_auc_score,
                             roc_curve, confusion_matrix, classification_report)

from logreg import NewtonRaphsonLogReg, IRLSLogReg, log_loss, gradient

RANDOM_STATE = 42
LAM = 1.0  
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")

def load_data():
    data = load_breast_cancer()
    X, y01 = data.data, data.target          
    y = 2 * y01 - 1                           

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    F_train = np.hstack([np.ones((len(X_train), 1)), X_train])
    F_test = np.hstack([np.ones((len(X_test), 1)), X_test])
    feature_names = ["bias"] + list(data.feature_names)
    return F_train, F_test, y_train, y_test, feature_names


def fit_timed(model, F, y):
    start = time.perf_counter()
    model.fit(F, y)
    return model, time.perf_counter() - start


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    F_train, F_test, y_train, y_test, feature_names = load_data()
    print(f"Обучение: {F_train.shape}, тест: {F_test.shape}")
    print(f"Доля класса +1 (benign): {np.mean(y_train == 1):.3f}\n")

    newton, t_newton = fit_timed(NewtonRaphsonLogReg(lam=LAM), F_train, y_train)
    irls, t_irls = fit_timed(IRLSLogReg(lam=LAM), F_train, y_train)

    ref = LogisticRegression(C=1.0 / LAM, fit_intercept=False, solver="lbfgs",
                             tol=1e-12, max_iter=10_000)
    ref, t_ref = fit_timed(ref, F_train, y_train)
    w_ref = ref.coef_.ravel()

    models = {
        "Newton-Raphson": (newton.coef_, newton.n_iter_, t_newton),
        "IRLS": (irls.coef_, irls.n_iter_, t_irls),
        "sklearn (lbfgs)": (w_ref, int(ref.n_iter_[0]), t_ref),
    }
    rows = []
    for name, (w, n_iter, t) in models.items():
        p_test = 1.0 / (1.0 + np.exp(-(F_test @ w)))     # P(y=+1|x)
        pred = np.where(F_test @ w >= 0, 1, -1)
        rows.append({
            "Метод": name,
            "Итераций": n_iter,
            "Время, мс": 1000 * t,
            "Q(w) train": log_loss(w, F_train, y_train, LAM),
            "||Q'(w)||": np.linalg.norm(gradient(w, F_train, y_train, LAM)),
            "max|w - w_ref|": np.max(np.abs(w - w_ref)),
            "Accuracy test": accuracy_score(y_test, pred),
            "ROC-AUC test": roc_auc_score(y_test, p_test),
            "LogLoss test": sk_log_loss(y_test, p_test, labels=[-1, 1]),
        })
    table = pd.DataFrame(rows).set_index("Метод")
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.6g}")
    print(table.to_string(), "\n")

    p_ref = ref.predict_proba(F_test)[:, 1]
    print("max |P_newton - P_sklearn| на тесте:",
          f"{np.max(np.abs(newton.predict_proba(F_test) - p_ref)):.2e}")
    print("max |P_irls   - P_sklearn| на тесте:",
          f"{np.max(np.abs(irls.predict_proba(F_test) - p_ref)):.2e}")
    print("Совпадение предсказанных классов (Newton / IRLS vs sklearn):",
          np.mean(newton.predict(F_test) == ref.predict(F_test)),
          np.mean(irls.predict(F_test) == ref.predict(F_test)))

    plot_convergence(newton, irls, log_loss(w_ref, F_train, y_train, LAM))
    plot_coefficients(newton.coef_, irls.coef_, w_ref, feature_names)
    plot_probabilities(newton.predict_proba(F_test), irls.predict_proba(F_test), p_ref)

    report_dataset_results(newton, F_test, y_test, feature_names)
    print(f"\nГрафики сохранены в {os.path.normpath(IMAGES_DIR)}")


def report_dataset_results(model, F_test, y_test, names):

    print("\nРезультаты на датасете Breast Cancer (тест, Newton-Raphson)")
    pred = model.predict(F_test)
    proba = model.predict_proba(F_test)
    labels = ["malignant (-1)", "benign (+1)"]
    print(classification_report(y_test, pred, labels=[-1, 1], target_names=labels, digits=3))

    cm = confusion_matrix(y_test, pred, labels=[-1, 1])
    print("Матрица ошибок (строки — истина, столбцы — прогноз):\n", cm)

    wrong = np.where(pred != y_test)[0]
    print("\nОшибочно классифицированные объекты теста:")
    for i in wrong:
        print(f"  #{i}: истинный класс {y_test[i]:+d}, P(benign) = {proba[i]:.3f}")

    w = model.coef_[1:]
    order = np.argsort(np.abs(w))[::-1][:10]
    top = pd.DataFrame({"признак": np.array(names[1:])[order],
                        "w_j": w[order],
                        "exp(w_j), отношение шансов": np.exp(w[order])})
    print("\nТоп-10 признаков по |w_j|:\n", top.to_string(index=False))

    plot_classification(cm, y_test, proba, labels)
    plot_feature_importance(w[order], np.array(names[1:])[order])


def plot_classification(cm, y_test, proba, labels):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # 1) матрица ошибок
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Прогноз")
    ax.set_ylabel("Истинный класс")
    ax.set_title("Матрица ошибок (тест)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Число объектов")

    # 2) ROC-кривая
    ax = axes[1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot(fpr, tpr, linewidth=2, label=f"Логистическая регрессия, AUC = {roc_auc_score(y_test, proba):.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Случайный классификатор")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривая (тест)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # 3) распределение вероятностей по истинным классам
    ax = axes[2]
    bins = np.linspace(0, 1, 21)
    ax.hist(proba[y_test == -1], bins=bins, alpha=0.7, label="истинно malignant (-1)")
    ax.hist(proba[y_test == 1], bins=bins, alpha=0.7, label="истинно benign (+1)")
    ax.axvline(0.5, color="black", linestyle="--", label="порог 0.5")
    ax.set_xlabel("P(y = +1 | x) — вероятность benign")
    ax.set_ylabel("Число объектов")
    ax.set_title("Распределение предсказанных вероятностей")
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "classification.png"), dpi=120)
    plt.close(fig)


def plot_feature_importance(w, names):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = np.where(w > 0, "tab:green", "tab:red")
    ax.barh(np.arange(len(w))[::-1], w, color=colors)
    ax.set_yticks(np.arange(len(w))[::-1])
    ax.set_yticklabels(names)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Вес w_j (признаки стандартизованы)")
    ax.set_title("Топ-10 признаков по |w_j|")
    handles = [Patch(color="tab:green", label="w_j > 0: рост признака → чаще benign"),
               Patch(color="tab:red", label="w_j < 0: рост признака → чаще malignant")]
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "feature_importance.png"), dpi=120)
    plt.close(fig)


def plot_convergence(newton, irls, q_ref):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for model, name, marker in [(newton, "Newton-Raphson (w⁰ = 0)", "o"),
                                (irls, "IRLS (w⁰ = МНК)", "s")]:
        gap = np.maximum(np.array(model.history_) - q_ref, 1e-16)
        ax.semilogy(gap, marker=marker, label=name)
    ax.set_title("Сходимость функционала Q(w)")
    ax.set_xlabel("Итерация t")
    ax.set_ylabel("Q(wᵗ) − Q(w_sklearn)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    for model, name, marker in [(newton, "Newton-Raphson", "o"), (irls, "IRLS", "s")]:
        ax.semilogy(np.maximum(model.grad_norms_, 1e-16), marker=marker, label=name)
    ax.set_title("Норма градиента ||Q'(wᵗ)||")
    ax.set_xlabel("Итерация t")
    ax.set_ylabel("||Q'(w)||")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "convergence.png"), dpi=120)
    plt.close(fig)


def plot_coefficients(w_newton, w_irls, w_ref, names):
    idx = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(idx - 0.27, w_newton, width=0.27, label="Newton-Raphson")
    ax.bar(idx, w_irls, width=0.27, label="IRLS")
    ax.bar(idx + 0.27, w_ref, width=0.27, label="sklearn LogisticRegression")
    ax.set_xticks(idx)
    ax.set_xticklabels(names, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Значение коэффициента w_j")
    ax.set_title("Коэффициенты моделей: собственные реализации vs эталон sklearn")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "coefficients.png"), dpi=120)
    plt.close(fig)


def plot_probabilities(p_newton, p_irls, p_ref):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="y = x (полное совпадение)")
    ax.scatter(p_ref, p_newton, s=40, alpha=0.6, label="Newton-Raphson")
    ax.scatter(p_ref, p_irls, s=12, marker="x", label="IRLS")
    ax.set_xlabel("P(y = +1 | x), sklearn")
    ax.set_ylabel("P(y = +1 | x), собственная реализация")
    ax.set_title("Предсказанные вероятности на тестовой выборке")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGES_DIR, "probabilities.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
