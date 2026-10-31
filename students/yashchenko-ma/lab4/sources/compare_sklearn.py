"""Сравнение собственной PCA (через SVD) с sklearn.decomposition.PCA.

Компоненты определены с точностью до знака, поэтому знак выравнивается по скалярному произведению.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from common import get_regression_data, setup_style, save_fig, C1, C2, C3
from pca_svd import PCASVD


def align(ref, other):
    """Выровнять знаки строк other по ref."""
    s = np.sign(np.sum(ref * other, axis=1))
    s[s == 0] = 1
    return other * s[:, None], s


def compare(X_tr, X_te, m):
    own = PCASVD(m).fit(X_tr)
    sk = PCA(n_components=m, svd_solver="full").fit(X_tr)
    comp_sk, s = align(own.components_, sk.components_)
    res = {
        "components": np.abs(own.components_ - comp_sk).max(),
        "singular_values": np.abs(own.singular_values_ - sk.singular_values_).max(),
        "explained_variance": np.abs(own.explained_variance_ - sk.explained_variance_).max(),
        "explained_variance_ratio": np.abs(own.explained_variance_ratio_ - sk.explained_variance_ratio_).max(),
        "transform_train": np.abs(own.transform(X_tr) - sk.transform(X_tr) * s).max(),
        "transform_test": np.abs(own.transform(X_te) - sk.transform(X_te) * s).max(),
        "inverse_transform": np.abs(own.inverse_transform(own.transform(X_te))
                                    - sk.inverse_transform(sk.transform(X_te))).max(),
    }
    n = X_tr.shape[1]
    # слайд 21, п.4: ||G U^T - F||^2 = sum_{j>m} lambda_j
    res["recon_err_vs_theorem"] = abs(own.reconstruction_error(X_tr) - own.eigvals_[m:].sum())
    return own, sk, s, res


def main():
    setup_style()
    X_tr, X_te, _, _, names = get_regression_data()
    n = X_tr.shape[1]
    print("\nmax |own - sklearn| (допуск 1e-9):")
    ok = True
    for m in (n, 3, 2):
        _, _, _, res = compare(X_tr, X_te, m)
        print(f"\nm = {m}")
        for k, v in res.items():
            print(f"  {k:<26} {v:.3e}")
            ok &= v < 1e-9
    print("\nИТОГ: реализации эквивалентны" if ok else "\nИТОГ: ЕСТЬ РАСХОЖДЕНИЯ")

    # --- график: проекции own vs sklearn ---
    own, sk, s, _ = compare(X_tr, X_te, 3)
    G_own, G_sk = own.transform(X_te), sk.transform(X_te) * s
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X_te), 800, replace=False)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for j, (col, mk) in enumerate(zip((C1, C2, C3), ("o", "s", "^"))):
        ax.scatter(G_sk[idx, j], G_own[idx, j], s=16, color=col, marker=mk,
                   alpha=0.6, label=f"ГК{j + 1}")
    lim = np.abs(G_own[idx, :3]).max() * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="#52514e", lw=1, ls="--")
    ax.set_xlabel("sklearn PCA: проекция (test)")
    ax.set_ylabel("PCASVD: проекция (test)")
    ax.set_title("Проекции на 3 ГК: собственная vs sklearn")
    ax.legend()
    save_fig(fig, "pca_vs_sklearn.png")


if __name__ == "__main__":
    main()
