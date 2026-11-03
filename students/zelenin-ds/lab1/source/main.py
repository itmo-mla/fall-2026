import numpy as np
from lin_clf import LinearClassifier
from dataset import load_data, prepare_data, TARGET
from metrics import f1_score, accuracy
from sklearn.svm import LinearSVC

from plots import visualise_marg, visualise_corr

TAO = 0.8
N_MULTI = 10
FIT_KWARGS = dict(n_iter=1000, batch_size=264, momentum=0.8,
                  verbose=300, tao=TAO)


def fit_model(x, y, seed=0, sampling="margin", weight_init="correlation"):
    model = LinearClassifier(seed=seed)
    model.fit(x, y, sampling=sampling, weight_init=weight_init, **FIT_KWARGS)
    return model


if __name__ == "__main__":
    results = []

    data = load_data()
    data = prepare_data(data)
    visualise_corr(data)
    X = data.drop(columns=TARGET).to_numpy(dtype=float)
    y = data[TARGET].to_numpy(dtype=float).ravel()

    rng = np.random.default_rng(43)
    idx = rng.permutation(len(X))
    split = int(0.8 * len(X))
    tr, te = idx[:split], idx[split:]

    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]

    trained = {}

    marg_model = fit_model(X_tr, y_tr, seed=0, weight_init="correlation")
    trained["train"] = marg_model

    random_preview = LinearClassifier()
    random_preview.random_init(X_te)
    visualise_marg(random_preview, X_te, y_te, "random")
    corr_preview = LinearClassifier()
    corr_preview.corr_init(X_te, y_te)
    visualise_marg(corr_preview, X_te, y_te, "corr")
    visualise_marg(trained["train"], X_te, y_te, "train")

    for start, sampling in [("correlation", "margin"), ("random", "margin"),
                            ("correlation", "uniform")]:
        label = f"{start}" if sampling == "margin" else f"{start}_uniform"
        print(f"{label} (sampling={sampling})")
        model = fit_model(X_tr, y_tr, seed=0, sampling=sampling,
                          weight_init=start)
        trained[label] = model
        pred = model.predict(X_te)
        results.append({"method": label, "f1_train": None,
                        "f1_test": f1_score(y_te, pred),
                        "acc": accuracy(y_te, pred)})

    print("multi_start")
    best_score = -np.inf
    best_w = None
    for i in range(N_MULTI):
        model = LinearClassifier(seed=i)
        model.fit(X_tr, y_tr, sampling="margin", weight_init="random",
                  **FIT_KWARGS)
        score = f1_score(y_tr, model.predict(X_tr))
        print(f"    запуск {i}: F1(train) = {score:.4f}")
        if score > best_score:
            best_score = score
            best_w = model.weights.copy()

    model.weights = best_w
    trained["multi"] = model
    pred = model.predict(X_te)
    results.append({"method": "multi_start", "f1_train": best_score,
                    "f1_test": f1_score(y_te, pred),
                    "acc": accuracy(y_te, pred)})

    reference_model = LinearSVC()
    reference_model.fit(X_tr, y_tr)
    pred = reference_model.predict(X_te)
    results.append({"method": "reference_model", "f1_train": None,
                    "f1_test": f1_score(y_te, pred),
                    "acc": accuracy(y_te, pred)})

    print("\nИтоги (метрики на test, кроме F1(train) -- критерий выбора мультистарта):")
    for r in results:
        tr_str = "-" if r["f1_train"] is None else f"{r['f1_train']:.4f}"
        print(f"{r['method']:20s} F1(train)={tr_str:>7s}  "
              f"F1(test)={r['f1_test']:.4f}  Acc={r['acc']:.4f}")
