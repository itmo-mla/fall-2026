from sklearn.linear_model import SGDClassifier


def run_sklearn_baseline(X_train, y_train, l2_strength):
    model = SGDClassifier(
        loss='squared_error',
        penalty='l2',
        alpha=l2_strength,
        fit_intercept=False,     # fit_intercept=False и bias у нас уже в X
        random_state=None,
        learning_rate='adaptive',
        eta0=0.001
    )
    model.fit(X_train, y_train)

    return {"w": model.coef_[0], "Q": None, "iters": model.n_iter_, "model": model}
