import numpy as np
from sklearn.decomposition import PCA


def compare_with_sklearn(pca_mine, X_train, svd_solver="auto"):
    d = pca_mine.components_.shape[0]
    skl = PCA(n_components=d, svd_solver=svd_solver).fit(X_train)

    cos = np.sum(pca_mine.components_ * skl.components_, axis=1)
    signs = np.sign(cos)
    signs[signs == 0] = 1

    rows = []
    for k in range(d):
        rows.append({
            "component": k + 1,
            "cos": abs(cos[k]),
            "var_ratio_diff": abs(pca_mine.explained_variance_ratio_[k] - skl.explained_variance_ratio_[k]),
        })

    Z_mine = pca_mine.transform(X_train)
    Z_skl = skl.transform(X_train) * signs
    max_z_diff = np.max(np.abs(Z_mine - Z_skl))

    X_approx_mine = pca_mine.inverse_transform(Z_mine)
    X_approx_skl = skl.inverse_transform(skl.transform(X_train))
    max_x_diff = np.max(np.abs(X_approx_mine - X_approx_skl))

    return skl, signs, rows, Z_mine, Z_skl, max_z_diff, max_x_diff
