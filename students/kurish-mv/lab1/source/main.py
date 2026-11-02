from pathlib import Path

import numpy as np

from data import load_pima, prepare_data
from experiments import (SGD_PARAMS, compare_results, evaluate, make_sgd,
                         run_experiment, run_multistart, test_table)
from explore import explore
from gradient import calculate_gradient, numerical_gradient
from initialization import correlation_initialization
from metrics import find_threshold
from optimizer import SteepestGradientDescent
from plots import (plot_confusion, plot_convergence, plot_margins,
                   plot_multistart, plot_recurrent)
from presentation import margin_presentation, random_presentation
from reference import analytic_ridge, make_model, sklearn_ridge

L2 = SGD_PARAMS["l2"]


def margin_stats(name, model, X, y):
    m = model.margin(X, y)
    print(f"{name}: M<0: {np.mean(m < 0):.1%}, |M|<0.2: {np.mean(np.abs(m) < 0.2):.1%}, mean M: {m.mean():.3f}")


def main():
    out = Path(__file__).parent / "res"
    out.mkdir(exist_ok=True)

    df = load_pima()
    explore(df, out)
    data = prepare_data(df)
    X_tr, y_tr = data["X_train"], data["y_train"]
    print(f"\ntrain/val/test: {len(X_tr)}/{len(data['X_val'])}/{len(data['X_test'])}")

    w, b = correlation_initialization(X_tr, y_tr)
    ga, gb = calculate_gradient(X_tr, y_tr, w, b, L2)
    gn, gbn = numerical_gradient(X_tr, y_tr, w, b, L2)
    print("Проверка градиента, max|diff| =", max(np.abs(ga - gn).max(), abs(gb - gbn)))

    results = [
        run_experiment("SGD corr + random", data, make_sgd(random_presentation), "correlation"),
        run_experiment("SGD corr + |M|", data, make_sgd(margin_presentation), "correlation"),
        run_experiment("Steepest corr", data, SteepestGradientDescent(l2=L2), "correlation"),
    ]
    runs_r, best_r = run_multistart("random", data, random_presentation)
    runs_m, best_m = run_multistart("|M|", data, margin_presentation)
    best_r["name"], best_m["name"] = "SGD multistart + random", "SGD multistart + |M|"
    results += [best_r, best_m]

    w_r, b_r = analytic_ridge(X_tr, y_tr, L2)
    ridge = make_model(w_r, b_r, L2)
    sk = make_model(*sklearn_ridge(X_tr, y_tr, L2), L2)
    steepest = results[2]["model"]

    print("\nValidation:")
    print(compare_results(results).round(4).to_string(index=False))
    compare_results(results).to_csv(out / "validation_metrics.csv", index=False)

    sgd_results = [r for r in results if r["name"].startswith("SGD")]
    best = max(sgd_results, key=lambda r: r["val"]["roc_auc"])
    print("\nЛучший SGD по val ROC-AUC:", best["name"])

    models = {best["name"]: best["model"], "Steepest": steepest,
              "Ridge analytic": ridge, "Ridge sklearn": sk}
    table = test_table(models, data)
    print("\nTest:")
    print(table.round(4).to_string(index=False))
    table.to_csv(out / "test_metrics.csv", index=False)

    print("\n||w_steepest - w_ridge|| =", np.linalg.norm(steepest.weights - w_r),
          "| ||w_ridge - w_sklearn|| =", np.linalg.norm(w_r - sk.weights))
    print("Q(train): steepest =", steepest.objective(X_tr, y_tr), "ridge =", ridge.objective(X_tr, y_tr))

    bm = best["model"]
    margin_stats("train", bm, X_tr, y_tr)
    margin_stats("test ", bm, data["X_test"], data["y_test"])
    plot_margins(bm.margin(X_tr, y_tr), "Отступы, train", out / "03_margins_train.png")
    plot_margins(bm.margin(data["X_test"], data["y_test"]), "Отступы, test", out / "04_margins_test.png")

    plot_convergence({r["name"]: r["model"].history for r in results}, ridge.objective(X_tr, y_tr),
                     out / "05_convergence.png")
    plot_recurrent(results[0]["model"].history, len(X_tr), out / "06_recurrent_q.png")
    plot_multistart({"random": [r["val_q"] for r in runs_r], "|M|": [r["val_q"] for r in runs_m]},
                    out / "07_multistart.png")

    thr = find_threshold(data["y_val"], bm.decision_function(data["X_val"]))
    cm = evaluate(bm, data["X_test"], data["y_test"], thr)["confusion_matrix"]
    plot_confusion(cm, f"Test, {best['name']}, thr={thr:.2f}", out / "08_confusion_test.png")
    print("\nРезультаты сохранены в", out)


if __name__ == "__main__":
    main()
