import numpy as np
from sklearn.linear_model import LogisticRegression

from data import Dataset
from optimization import FitResult, fit_logistic_regression


def run_experiments(training_data: Dataset) -> dict[str, FitResult]:
    fits = {}
    for name, method in [("Ньютон–Рафсон", "newton"), ("IRLS", "irls")]:
        fits[name] = fit_logistic_regression(
            training_data.features, training_data.labels, method,
        )
    return fits


def train_reference(training_data: Dataset) -> LogisticRegression:
    reference = LogisticRegression(C=np.inf, solver="lbfgs", tol=1e-12, max_iter=1000)
    reference.fit(training_data.features[:, 1:], training_data.labels)
    return reference
