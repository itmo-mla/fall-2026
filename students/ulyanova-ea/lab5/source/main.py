from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy.special import expit
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def Q(F, y, w):
    return np.logaddexp(0, -y * (F @ w)).sum()


def fit_irls(F, y, tol=1e-9, max_iter=50):
    w = np.linalg.solve(F.T @ F, F.T @ y)
    history = {"Q": [Q(F, y, w)], "sigma_change": []}

    for _ in range(max_iter):
        sigma_old = expit(y * (F @ w))
        sigma = np.clip(sigma_old, 1e-15, 1 - 1e-15)

        gamma = np.sqrt((1 - sigma) * sigma)
        F_tilde = gamma[:, None] * F
        y_tilde = y * np.sqrt((1 - sigma) / sigma)
        delta = np.linalg.solve(F_tilde.T @ F_tilde, F_tilde.T @ y_tilde)

        h = 1.0
        w = w + h * delta
        change = np.max(np.abs(expit(y * (F @ w)) - sigma_old))
        history["Q"].append(Q(F, y, w))
        history["sigma_change"].append(change)

        if change < tol:
            return w, history

    raise RuntimeError(f"IRLS did not converge in {max_iter} iterations")


def plot_convergence(history, tol):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["Q"], marker="o", label="Собственная IRLS")
    steps = np.arange(1, len(history["sigma_change"]) + 1)
    axes[1].semilogy(steps, np.maximum(history["sigma_change"], 1e-16), marker="o", label="Собственная IRLS")

    axes[0].set(xlabel="Итерация", ylabel="Значение Q(w)", title="Функционал качества Q(w)")
    axes[1].axhline(tol, color="black", linestyle="--", label="Порог остановки")
    axes[1].set(xlabel="Итерация", ylabel=r"Максимальное изменение $\sigma_i$", title="Критерий сходимости")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()

    img_dir = Path(__file__).resolve().parent.parent / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = img_dir / "convergence.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"\nГрафики: {path}")


def main():
    data = load_breast_cancer()
    X = data.data[:, :10]
    y = 2 * data.target - 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    F_train = np.column_stack((np.ones(len(X_train)), X_train))
    F_test = np.column_stack((np.ones(len(X_test)), X_test))
    tol = 1e-9

    start = perf_counter()
    w, history = fit_irls(F_train, y_train, tol=tol)
    elapsed = perf_counter() - start

    reference = sm.GLM((y_train + 1) / 2, F_train, family=sm.families.Binomial(link=sm.families.links.Logit()))
    start = perf_counter()
    result = reference.fit(method="IRLS", maxiter=100, tol=tol)
    elapsed_ref = perf_counter() - start
    p_ref = result.predict(F_test)
    print(f"{'Модель':<24} {'Accuracy':>9} {'Precision':>10} {'Recall':>9} {'F1':>9} {'Итераций':>9} {'Время, мс':>11}")
    pred = np.where(F_test @ w >= 0, 1, -1)
    metrics = [accuracy_score(y_test, pred), precision_score(y_test, pred), recall_score(y_test, pred), f1_score(y_test, pred)]
    print(f"{'Собственная IRLS':<24} " + " ".join(f"{value:9.4f}" for value in metrics) + f" {len(history['sigma_change']):9d} {elapsed * 1000:11.4f}")
    pred_ref = np.where(p_ref >= 0.5, 1, -1)
    metrics = [accuracy_score(y_test, pred_ref), precision_score(y_test, pred_ref), recall_score(y_test, pred_ref), f1_score(y_test, pred_ref)]
    print(f"{'Готовая (statsmodels)':<24} " + " ".join(f"{value:9.4f}" for value in metrics) + f" {result.fit_history['iteration']:9d} {elapsed_ref * 1000:11.4f}")

    if np.all(y_train * (F_train @ w) > 0):
        print("Обучающая выборка линейно разделима: конечного минимума Q без регуляризации нет.")

    p = expit(F_test @ w)
    eps = 1e-6
    differences = np.abs(p - p_ref)
    indices = np.flatnonzero(differences > eps)
    if len(indices):
        print(f"Расхождение выше допуска {eps:g} в {len(indices)} из {len(p)} объектов")
        print(f"{'p_IRLS':>14} {'p_reference':>14} {'Разница':>14}")
        for i in indices:
            print(f"{p[i]:14.10f} {p_ref[i]:14.10f} {differences[i]:14.3e}")
    else:
        print(f"Расхождений вероятностей выше допуска {eps:g} нет.")
    plot_convergence(history, tol)


if __name__ == "__main__":
    main()