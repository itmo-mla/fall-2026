import os
import time

import numpy as np
from sklearn.svm import SVC

import plots
from data import load_cancer, load_moons, standardize, train_test_split
from svm import SVM, linear, polynomial, rbf

IMAGES = os.path.join(os.path.dirname(__file__), "..", "images")
C = 1.0


def kernels(n_features):
    """Our kernel and the same kernel in sklearn terms: sklearn's poly is (gamma <x, x'> + coef0)^degree."""
    gamma = 1.0 if n_features == 2 else 1.0 / n_features
    return {
        "линейное": (linear, {"kernel": "linear"}),
        "полиномиальное, d = 3": (polynomial(3), {"kernel": "poly", "degree": 3, "gamma": 1.0, "coef0": 1.0}),
        f"RBF, γ = {gamma:.3g}": (rbf(gamma), {"kernel": "rbf", "gamma": gamma}),
    }


def fit_timed(model, X, y):
    start = time.perf_counter()
    model.fit(X, y)
    return time.perf_counter() - start


def linear_classifier(model, reference, X_test, y_test):
    """a(x) = sign(<w, x> - w0) with w = sum_i lam_i y_i x_i; sklearn keeps w in coef_ and -w0 in intercept_."""
    w, w_ref = model.w, reference.coef_[0]
    predicted = np.where(X_test @ w - model.w0 >= 0, 1, -1)
    print(f"  {'':22s} a(x) = sign(<w, x> - w0): accuracy = {np.mean(predicted == y_test):.3f}")
    print(f"  {'':22s} w = sum λ_i y_i x_i: max|w - w_SVC| = {np.abs(w - w_ref).max():.4f}, "
          f"w0 = {model.w0:.3f} / {-reference.intercept_[0]:.3f}, "
          f"ширина полосы 2/||w|| = {2 / np.linalg.norm(w):.3f} / {2 / np.linalg.norm(w_ref):.3f}")
    if len(w) <= 3:
        print(f"  {'':22s} w = {np.round(w, 3)} / {np.round(w_ref, 3)}")


def compare(title, X, y):
    """Train our SVM and sklearn's SVC with every kernel, print the table, return both sets of models."""
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    Z_train, Z_test = standardize(X_train), standardize(X_test, X_train)
    print(f"\n{title}: обучение {Z_train.shape}, тест {Z_test.shape}, C = {C}")
    print(f"  {'ядро':22s} {'accuracy':>15s} {'опорных':>9s} {'совпадение':>11s} {'max|Δf|':>8s} {'время, с':>15s}")

    ours, references = {}, {}
    for name, (kernel, params) in kernels(X.shape[1]).items():
        model, reference = SVM(C=C, kernel=kernel), SVC(C=C, **params)
        t, t_ref = fit_timed(model, Z_train, y_train), fit_timed(reference, Z_train, y_train)

        predicted, predicted_ref = model.predict(Z_test), reference.predict(Z_test)
        accuracy, accuracy_ref = np.mean(predicted == y_test), np.mean(predicted_ref == y_test)
        agreement = np.mean(predicted == predicted_ref)
        gap = np.abs(model.decision_function(Z_test) - reference.decision_function(Z_test)).max()
        print(f"  {name:22s} {accuracy:6.3f} / {accuracy_ref:.3f} {len(model.lam):3d} / {reference.n_support_.sum():3d} "
              f"{agreement:11.3f} {gap:8.4f} {t:6.3f} / {t_ref:.3f}")

        if params["kernel"] == "linear":
            linear_classifier(model, reference, Z_test, y_test)

        ours[f"{name}, accuracy = {accuracy:.3f}"] = model
        references[f"{name}, accuracy = {accuracy:.3f}"] = reference
    print("  в парах: наш SVM / sklearn SVC")
    return Z_train, y_train, ours, references


def main():
    Z_train, y_train, ours, references = compare("Moons", *load_moons())
    plots.plot_decision(ours, Z_train, y_train, os.path.join(IMAGES, "moons_kernels.png"), references)

    compare("Breast Cancer", *load_cancer())


if __name__ == "__main__":
    main()
