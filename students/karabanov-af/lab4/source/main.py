import numpy as np
from sklearn.decomposition import PCA

import os

import pca
import plots
import regression
from data import load_data, standardize, train_test_split


def main():
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"объектов: {len(y)}, признаков: {X.shape[1]}, признаки: {', '.join(feature_names)}")
    print(f"обучение: {X_train.shape}, тест: {X_test.shape}")
    print(f"целевая переменная: от {y.min():.0f} до {y.max():.0f}, среднее {y.mean():.1f}")

    Z = standardize(X_train)
    correlation = np.corrcoef(Z, rowvar=False)
    print("\nсамые скоррелированные пары признаков:")
    pairs = [(abs(correlation[i, j]), feature_names[i], feature_names[j], correlation[i, j])
             for i in range(len(feature_names)) for j in range(i + 1, len(feature_names))]
    for _, first, second, value in sorted(pairs, reverse=True)[:4]:
        print(f"  {first:4s} и {second:4s} r = {value:+.3f}")
    print(f"\nчисло обусловленности матрицы признаков: {np.linalg.cond(Z):.1f}")

    decomposition(Z)
    dimension(Z)
    Z_test = standardize(X_test, X_train)
    compare_with_reference(Z, Z_test)
    regression_experiment(X_train, Z, y_train, Z_test, y_test)


IMAGES = os.path.join(os.path.dirname(__file__), "..", "images")


def decomposition(Z):
    """Checks that the SVD decomposition really has the properties PCA is built on."""
    mean, components, singular = pca.fit(Z)
    variance = pca.explained_variance(singular, len(Z))
    print("\nсингулярное разложение:")
    print("  сингулярные числа   :", singular.round(2))
    print("  дисперсии компонент :", variance.round(3))
    print("  доли дисперсии      :", pca.explained_variance_ratio(singular).round(3))

    P = pca.transform(Z, mean, components)
    covariance = np.cov(P, rowvar=False)
    print("\nпроверки:")
    print(f"  ортонормированность осей, max|V Vt - I| = "
          f"{np.abs(components @ components.T - np.eye(len(components))).max():.2e}")
    print(f"  некоррелированность проекций, max вне диагонали = "
          f"{np.abs(covariance - np.diag(np.diag(covariance))).max():.2e}")
    print(f"  дисперсия проекций равна s^2/(n-1): {np.allclose(P.var(axis=0, ddof=1), variance)}")
    print(f"  восстановление по всем осям, max|Z - Z'| = "
          f"{np.abs(pca.inverse_transform(P, mean, components) - Z).max():.2e}")

    print("\nпотеря информации при отбрасывании осей:")
    for k in [2, 5, 8, 9]:
        approximation = pca.inverse_transform(pca.transform(Z, mean, components, k), mean, components)
        error = ((approximation - Z) ** 2).sum() / (len(Z) - 1)
        print(f"  k = {k}: ошибка восстановления {error:.4f}, сумма отброшенных дисперсий {variance[k:].sum():.4f}")


def dimension(Z):
    """Effective dimension: how many axes are worth keeping, by the criteria from the lecture."""
    _, _, singular = pca.fit(Z)
    variance = pca.explained_variance(singular, len(Z))
    ratio = pca.explained_variance_ratio(singular)
    residual = pca.residual_share(ratio)
    slope = pca.steep_slope(residual)

    print("\nэффективная размерность:")
    print(f"{'m':>3} {'дисперсия':>11} {'доля':>8} {'накопленно':>12} {'E(m)':>9} {'E(m-1)/E(m)':>13}")
    for m, (v, r, c, e) in enumerate(zip(variance, ratio, np.cumsum(ratio), residual), start=1):
        jump = f"{slope[m - 1]:>13.2f}" if m <= len(slope) else f"{'-':>13}"
        print(f"{m:>3} {v:>11.3f} {r:>8.3f} {c:>12.3f} {e:>9.4f}{jump}")

    print("\nкритерии:")
    for eps in [0.15, 0.05, 0.01]:
        print(f"  порог E(m) <= {eps:.0%}: {pca.effective_dimension(ratio, eps)} компонент")
    print(f"  критерий крутого склона: склон ломается на m = {int(np.argmax(slope > 2)) + 1} "
          f"(потеря впервые падает больше чем вдвое), резче всего на m = {int(np.argmax(slope)) + 1} "
          f"(в {slope.max():.1f} раза)")
    print(f"  критерий Кайзера (дисперсия > 1): {int((variance > 1).sum())} компонент")
    print(f"  обусловленность: s_max / s_min = {singular[0] / singular[-1]:.1f}, "
          f"при 8 компонентах {singular[0] / singular[7]:.1f}")

    plots.plot_scree(variance, residual, slope, "Эффективная размерность выборки diabetes",
                     os.path.join(IMAGES, "scree.png"))


def compare_with_reference(Z, Z_test):
    """Same decomposition by sklearn: everything must match up to the sign of the axes."""
    mean, components, singular = pca.fit(Z)
    reference = PCA(n_components=None, svd_solver="full").fit(Z)

    signs = np.sign(np.sum(components * reference.components_, axis=1))
    print("\nсравнение с sklearn.decomposition.PCA:")
    print(f"  знаки осей совпали у {int((signs > 0).sum())} из {len(signs)} компонент")
    print(f"  max|среднее - mean_|            = {np.abs(mean - reference.mean_).max():.2e}")
    print(f"  max|сингулярные - singular_values_| = {np.abs(singular - reference.singular_values_).max():.2e}")
    print(f"  max|дисперсии - explained_variance_| = "
          f"{np.abs(pca.explained_variance(singular, len(Z)) - reference.explained_variance_).max():.2e}")
    print(f"  max|доли - explained_variance_ratio_| = "
          f"{np.abs(pca.explained_variance_ratio(singular) - reference.explained_variance_ratio_).max():.2e}")
    print(f"  max|оси - components_| с учётом знака = "
          f"{np.abs(signs[:, None] * components - reference.components_).max():.2e}")

    own = pca.transform(Z_test, mean, components)
    print(f"  max|проекции теста - transform()| с учётом знака = "
          f"{np.abs(own * signs - reference.transform(Z_test)).max():.2e}")
    for k in [2, 5, 8]:
        restored = pca.inverse_transform(pca.transform(Z_test, mean, components, k), mean, components)
        model = PCA(n_components=k, svd_solver="full").fit(Z)
        difference = np.abs(restored - model.inverse_transform(model.transform(Z_test))).max()
        print(f"  k = {k}: max|восстановление - inverse_transform()| = {difference:.2e}")


def least_squares_checks(Z, y_train):
    """The lecture derives the same solution three ways: normal equations, pseudoinverse, SVD."""
    weights, intercept = regression.fit(Z, y_train)
    centered = y_train - intercept
    U, singular, _ = np.linalg.svd(Z, full_matrices=False)

    print("\nпроверки МНК-решения:")
    print(f"  max|w - pinv(Z) y|      = {np.abs(weights - np.linalg.pinv(Z) @ centered).max():.2e}")
    print(f"  max|w - (Zt Z)^-1 Zt y| = "
          f"{np.abs(weights - np.linalg.solve(Z.T @ Z, Z.T @ centered)).max():.2e}")
    print(f"  ||w||^2 = {np.linalg.norm(weights) ** 2:.4f}, "
          f"сумма (v y)^2 / lambda = {np.sum((U.T @ centered) ** 2 / singular ** 2):.4f}")


def regression_experiment(X_train, Z, y_train, Z_test, y_test):
    """Three methods differing only by the SVD filter: least squares, ridge, regression on components.

    The number of components and the regularization coefficient are chosen by cross validation
    on the training data: the test set is touched once, to report the final quality.
    """
    least_squares_checks(Z, y_train)
    _, singular, _ = np.linalg.svd(Z, full_matrices=False)

    print("\nрегрессия на главных компонентах:")
    print(f"{'k':>3} {'R2 CV':>9} {'R2 train':>10} {'R2 test':>10} {'RMSE test':>11} {'||w||':>9}")
    components = np.arange(1, Z.shape[1] + 1)
    pcr_cv, pcr_test = [], []
    for k in components:
        weights, intercept = regression.fit(Z, y_train, n_components=k)
        prediction = regression.predict(Z_test, weights, intercept)
        pcr_cv.append(regression.cross_val_r2(X_train, y_train, n_components=k))
        pcr_test.append(regression.r2(y_test, prediction))
        print(f"{k:>3} {pcr_cv[-1]:>9.4f} {regression.r2(y_train, regression.predict(Z, weights, intercept)):>10.4f} "
              f"{pcr_test[-1]:>10.4f} {regression.rmse(y_test, prediction):>11.2f} "
              f"{np.linalg.norm(weights):>9.2f}")

    alphas = np.logspace(-2, 4, 25)
    ridge_cv = [regression.cross_val_r2(X_train, y_train, alpha=alpha) for alpha in alphas]
    ridge_test = []
    for alpha in alphas:
        weights, intercept = regression.fit(Z, y_train, alpha=alpha)
        ridge_test.append(regression.r2(y_test, regression.predict(Z_test, weights, intercept)))

    best_k = components[int(np.argmax(pcr_cv))]
    best_alpha = alphas[int(np.argmax(ridge_cv))]
    print("\nвыбор гиперпараметров по кросс-валидации на обучении:")
    print(f"  число компонент k = {best_k} (R2 на кросс-валидации {max(pcr_cv):.4f})")
    print(f"  коэффициент alpha = {best_alpha:.3g} (R2 на кросс-валидации {max(ridge_cv):.4f})")
    print(f"  гребневая при этом alpha использует {regression.effective_dimension(singular, best_alpha):.2f} "
          f"направлений из {Z.shape[1]} (след проекционной матрицы)")

    print("\nитог на тесте:")
    for name, kwargs in [("МНК по всем признакам", {}),
                         (f"PCA-регрессия, k = {best_k}", {"n_components": int(best_k)}),
                         (f"гребневая, alpha = {best_alpha:.3g}", {"alpha": best_alpha})]:
        weights, intercept = regression.fit(Z, y_train, **kwargs)
        print(f"  {name:28s}: R2 = {regression.r2(y_test, regression.predict(Z_test, weights, intercept)):.4f}, "
              f"||w|| = {np.linalg.norm(weights):.2f}")

    plots.plot_regression(components, pcr_cv, pcr_test, alphas, ridge_cv, ridge_test,
                          regression.r2(y_test, regression.predict(Z_test, *regression.fit(Z, y_train))),
                          "Снижение размерности и регуляризация в задаче регрессии",
                          os.path.join(IMAGES, "regression.png"))


if __name__ == "__main__":
    main()
