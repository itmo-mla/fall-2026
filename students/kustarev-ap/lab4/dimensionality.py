import numpy as np


def reconstruction_mse(pca, X, m):
    Z = pca.transform(X, m)
    X_approx = pca.inverse_transform(Z, m)
    return np.mean((X - X_approx) ** 2)


def dimension_table(pca, X_train, X_test):
    d = pca.components_.shape[0]
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    rows = []
    for m in range(1, d + 1):
        rows.append({
            "m": m,
            "cum_var": cum_var[m - 1],
            "mse_train": reconstruction_mse(pca, X_train, m),
            "mse_test": reconstruction_mse(pca, X_test, m),
        })
    return rows


def choose_m_star(cum_var, threshold=0.95):
    for m, cv in enumerate(cum_var, start=1):
        if cv >= threshold:
            return m
    return len(cum_var)


def regression_r2(Z_train, y_train, Z_test, y_test):
    A_train = np.column_stack([Z_train, np.ones(len(Z_train))])
    coef, *_ = np.linalg.lstsq(A_train, y_train, rcond=None)
    A_test = np.column_stack([Z_test, np.ones(len(Z_test))])
    y_pred = A_test @ coef
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    return 1 - ss_res / ss_tot


def bonus_regression_table(pca, X_train, y_train, X_test, y_test, ms):
    rows = []
    for m in ms:
        Z_train = pca.transform(X_train, m)
        Z_test = pca.transform(X_test, m)
        r2 = regression_r2(Z_train, y_train, Z_test, y_test)
        rows.append({"m": m, "r2_test": r2})
    return rows
