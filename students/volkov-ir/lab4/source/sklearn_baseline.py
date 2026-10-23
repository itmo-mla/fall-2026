import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import model


def calculate_sklearn_pca(X, m, X_test):
    pca = PCA(n_components=m).fit(X)
    # sklearn делит на n - 1, умножаем обратно, чтобы сравнить с нашими lam
    lam = pca.explained_variance_ * (len(X) - 1)
    u = pca.components_.T
    g = pca.transform(X)
    f_hat = pca.inverse_transform(g)
    g_test = pca.transform(X_test)
    f_hat_test = pca.inverse_transform(g_test)
    return lam, u, g, f_hat, g_test, f_hat_test


def calculate_signs(u, u_sk):
    # знак столбца в разложении определен неоднозначно, смотрим, у каких столбцов он разошелся
    return np.sign(np.sum(u * u_sk, axis=0))


def calculate_sklearn_regression(X_train, y_train, X_test):
    model = LinearRegression().fit(X_train, y_train)
    return model.predict(X_test), model.coef_


# строки сравнения – (название, наши первые значения, значения sklearn, совпало ли по всем элементам)
def calculate_comparison(X_train, X_test, lam, u, m):
    n_features = X_train.shape[1]
    lam_sk, u_sk, g_sk, f_hat_sk, g_test_sk, f_hat_test_sk = calculate_sklearn_pca(X_train, n_features, X_test)
    # знаки направлений у нас и у sklearn могут отличаться, перед сравнением выравниваем
    signs = calculate_signs(u, u_sk)
    g = model.calculate_g(X_train, u, n_features) * signs
    g_test = model.calculate_g(X_test, u, n_features) * signs

    # восстановление при m направлениях, знак там сокращается
    f_hat_m_sk = calculate_sklearn_pca(X_train, m, X_test)[3]
    f_hat_test_m_sk = calculate_sklearn_pca(X_train, m, X_test)[5]
    f_hat = model.calculate_f_hat(model.calculate_g(X_train, u, m), u, m)
    f_hat_test = model.calculate_f_hat(model.calculate_g(X_test, u, m), u, m)

    # совпало ли по всем элементам, и на train, и на test
    rows = [
        ("lambda", lam, lam_sk, np.allclose(lam, lam_sk)),
        ("U, первое направление", u[:, 0] * signs[0], u_sk[:, 0], np.allclose(u * signs, u_sk)),
        ("G, первая машина", g[0], g_sk[0], np.allclose(g, g_sk) and np.allclose(g_test, g_test_sk)),
        ("F_hat при m = " + str(m) + ", первая машина", f_hat[0], f_hat_m_sk[0], np.allclose(f_hat, f_hat_m_sk) and np.allclose(f_hat_test, f_hat_test_m_sk)),
    ]
    return rows
