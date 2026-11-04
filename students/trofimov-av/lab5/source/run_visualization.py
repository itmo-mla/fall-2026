from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from data import load_data
from logistic import LogisticRegression


def main():
    root = Path(__file__).resolve().parents[1]
    (root / "figures").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    x_train, x_test, y_train, y_test = load_data()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    histories, budget_rows = [], []
    for method, marker in (("newton", "o"), ("irls", "x")):
        model = LogisticRegression(method=method, l2=1).fit(x_train, y_train)
        axes[0].plot(model.loss_history_, marker=marker, markersize=9, label=method)
        axes[1].semilogy(np.maximum(model.gradient_history_, 1e-16), marker=marker, markersize=9, label=method)
        for iteration, (loss, gradient) in enumerate(zip(model.loss_history_, model.gradient_history_)):
            histories.append({"method": method, "iteration": iteration, "objective": loss, "gradient_norm": gradient})
        for budget in (1, 2, 3, 5, 10, 30):
            limited = LogisticRegression(method=method, l2=1, max_iter=budget).fit(x_train, y_train)
            budget_rows.append({
                "method": method, "max_iter": budget, "iterations": limited.n_iter_,
                "converged": limited.converged_, "gradient_norm": limited.gradient_history_[-1],
                "test_accuracy": accuracy_score(y_test, limited.predict(x_test)),
                "test_log_loss": log_loss(y_test, limited.predict_proba(x_test)[:, 1]),
            })
    axes[0].set(ylabel="Сумма log-loss + L2-штраф", title="Функция потерь на train")
    axes[1].set(ylabel="Максимальный модуль градиента", title="Критерий сходимости")
    axes[1].axhline(1e-8, linestyle="--", color="red", label="Допуск 10⁻⁸")
    for ax in axes:
        ax.set_xlabel("Число обновлений")
        ax.grid(alpha=0.25)
        ax.legend()
    figure.tight_layout()
    figure.savefig(root / "figures" / "convergence.png", dpi=160)
    plt.close(figure)
    pd.DataFrame(histories).to_csv(root / "results" / "convergence.tsv", sep="\t", index=False)
    budget_table = pd.DataFrame(budget_rows)
    budget_table.to_csv(root / "results" / "iteration_budget.tsv", sep="\t", index=False)

    x_train, x_test, y_train, y_test = load_data(features=[2, 3])
    combined = np.vstack((x_train, x_test))
    xx, yy = np.meshgrid(
        np.linspace(combined[:, 0].min() - 0.5, combined[:, 0].max() + 0.5, 200),
        np.linspace(combined[:, 1].min() - 0.5, combined[:, 1].max() + 0.5, 200),
    )
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    boundary_rows = []
    for ax, method in zip(axes, ("newton", "irls", "sklearn")):
        model = (SklearnLogisticRegression(C=1, tol=1e-12, max_iter=2000) if method == "sklearn"
                 else LogisticRegression(method=method, l2=1))
        model.fit(x_train, y_train)
        probability = model.predict_proba(grid)[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, probability, levels=np.linspace(0, 1, 11), cmap="RdBu", alpha=0.25)
        ax.contour(xx, yy, probability, levels=[0.5], colors="black", linewidths=1)
        ax.plot([], [], color="black", label="Граница p = 0.5")
        for label, name, color in ((0, "versicolor", "tab:red"), (1, "virginica", "tab:blue")):
            for data, target, marker, part in ((x_train, y_train, "o", "train"), (x_test, y_test, "^", "test")):
                mask = target == label
                ax.scatter(data[mask, 0], data[mask, 1], marker=marker, color=color,
                           edgecolors="black", s=45, label=f"{name}, {part}")
        accuracy = accuracy_score(y_test, model.predict(x_test))
        ax.set(title=f"{method}: test accuracy = {accuracy:.2f}",
               xlabel="Длина лепестка, стандартизованная", ylabel="Ширина лепестка, стандартизованная")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        boundary_rows.append({"method": method, "test_accuracy": accuracy,
                              "test_log_loss": log_loss(y_test, model.predict_proba(x_test)[:, 1])})
    figure.tight_layout()
    figure.savefig(root / "figures" / "boundaries.png", dpi=160)
    plt.close(figure)
    pd.DataFrame(boundary_rows).to_csv(root / "results" / "boundaries.tsv", sep="\t", index=False)
    print(budget_table.to_string(index=False))
    print(pd.DataFrame(boundary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
