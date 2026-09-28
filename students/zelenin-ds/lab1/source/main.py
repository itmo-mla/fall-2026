import numpy as np
from lin_clf import  LinearClassifier
from  dataset import  *
from  metrics import *
from  sklearn.svm import LinearSVC

from plots import visualise_marg, visualise_corr

if __name__ ==  "__main__":
    tao = 0.8
    start_type = ["correlation","random","multi"]
    f1_scores = []
    f1_type = []

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
    for mode in ["train", "random", "corr"]:
        visualise_marg(X_te,y_te, mode)

"""
    for start in start_type:
        print(f"{start}")
        if start == "multi":
            max_score = 0
            for i in range(10):
                model.fit(X_tr, y[tr], n_iter=1000, batch_size=264, momentum=0.8, verbose=300, sampling= "margin", tao = tao)
                pred = model.predict(X_te)
                f1 = f1_score(y_te, pred)
                max_score = max(f1, max_score)
            f1_scores.append(max_score)
            f1_type.append("multi_start")
        else:
            model.fit(X_tr, y[tr], n_iter=1000, batch_size=264, momentum=0.8, verbose=300, sampling= "margin", weight_init=start, tao=tao)
            pred = model.predict(X_te)
            print(pred)
            f1 = f1_score(y_te, pred)
            f1_scores.append(f1)
            f1_type.append(f"{start}")

    reference_model = LinearSVC()
    reference_model.fit(X_tr, y_tr)
    pred = reference_model.predict(X_te)
    f1_scores.append(f1_score(y_te, pred))
    start_type.append("reference_model")
    print(f1_scores, start_type)
"""


