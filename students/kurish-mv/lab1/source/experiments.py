import pandas as pd

from classifier import LinearClassifier
from metrics import evaluate_classification, find_threshold
from optimizer import SGDWithMomentum

SGD_PARAMS = dict(lr=0.002, momentum=0.9, l2=1e-3, epochs=80)


def make_sgd(presentation, seed=0, **kwargs):
    return SGDWithMomentum(presentation=presentation, seed=seed, **{**SGD_PARAMS, **kwargs})


def evaluate(model, X, y, threshold=0.0):
    return evaluate_classification(y, model.predict(X, threshold), model.decision_function(X))


def run_experiment(name, data, optimizer, initialization, seed=0):
    model = LinearClassifier(optimizer, initialization, seed)
    model.fit(data["X_train"], data["y_train"])
    return {
        "name": name,
        "model": model,
        "val_q": model.objective(data["X_val"], data["y_val"]),
        "val": evaluate(model, data["X_val"], data["y_val"]),
    }


def run_multistart(name, data, presentation, n_starts=12):
    runs = [run_experiment(f"{name} #{s}", data, make_sgd(presentation, s), "random", s)
            for s in range(n_starts)]
    return runs, min(runs, key=lambda r: r["val_q"])


def compare_results(results):
    rows = [{"model": r["name"], "val_Q": r["val_q"],
             **{k: v for k, v in r["val"].items() if k != "confusion_matrix"}}
            for r in results]
    return pd.DataFrame(rows)


def test_table(models, data):
    rows = []
    for name, model in models.items():
        thr = find_threshold(data["y_val"], model.decision_function(data["X_val"]))
        for label, t in (("thr=0", 0.0), ("thr=val", thr)):
            m = evaluate(model, data["X_test"], data["y_test"], t)
            rows.append({"model": f"{name} ({label})",
                         **{k: v for k, v in m.items() if k != "confusion_matrix"}})
    return pd.DataFrame(rows)
