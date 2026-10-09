import numpy as np

from sklearn.linear_model import LogisticRegression

import logistic
import plots
import os

from data import add_bias, load_data, standardize, train_test_split

IMAGES = os.path.join(os.path.dirname(__file__), "..", "images")


def main():
    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"объектов: {len(y)}, признаков: {X.shape[1]}")
    print(f"классы: злокачественная {int((y == 0).sum())}, доброкачественная {int((y == 1).sum())}")
    print(f"обучение: {X_train.shape}, тест: {X_test.shape}, "
          f"доля класса 1: {y_train.mean():.3f} и {y_test.mean():.3f}")

    print("\nразмах признаков до стандартизации:")
    for j in [0, 3, 23]:
        print(f"  {feature_names[j]:24s} от {X[:, j].min():8.3f} до {X[:, j].max():8.3f}")

    F = add_bias(standardize(X_train))
    F_test = add_bias(standardize(X_test, X_train))
    labels, labels_test = logistic.signed(y_train), logistic.signed(y_test)
    print(f"\nматрица обучения со свободным членом: {F.shape}, "
          f"число обусловленности {np.linalg.cond(F):.1f}")

    newton(F, labels, F_test, labels_test)
    separability(F, labels, F_test, labels_test)
    equivalence(F, labels, y_train)
    reweighting(F, labels)
    compare_with_reference(standardize(X_train), y_train, F, labels,
                           standardize(X_test, X_train), F_test)
    convergence(F, labels, y_train)
    regularization(X_train, labels, F_test, labels_test)
    probabilities(F, labels, F_test, labels_test)
    odds(F, labels, feature_names)


def newton(F, y, F_test, y_test, tau=1.0):
    """Newton-Raphson on the log-loss: the Hessian is rebuilt and inverted at every step."""
    trajectory = logistic.newton_raphson(F, y, tau=tau)
    weights = trajectory[-1]

    print(f"\nметод Ньютона-Рафсона, tau = {tau}:")
    print(f"{'шаг':>4} {'Q(w)':>12} {'изменение':>12} {'||w||':>9} {'cond гессиана':>15}")
    previous = None
    for step, w in enumerate(trajectory):
        value = logistic.log_loss(F, y, w, tau)
        change = "-" if previous is None else f"{previous - value:12.3e}"
        print(f"{step:>4} {value:>12.6f} {change:>12} {np.linalg.norm(w):>9.3f} "
              f"{np.linalg.cond(logistic.hessian(F, y, w, tau)):>15.1f}")
        previous = value

    print(f"  сошёлся за {len(trajectory) - 1} итераций, ||градиента|| = "
          f"{np.linalg.norm(logistic.gradient(F, y, weights, tau)):.2e}")
    print(f"  точность: обучение {logistic.accuracy(y, logistic.predict(F, weights)):.4f}, "
          f"тест {logistic.accuracy(y_test, logistic.predict(F_test, weights)):.4f}")


def separability(F, y, F_test, y_test):
    """Without regularization the maximum of the likelihood on separable data is at infinity."""
    print("\nчто делает регуляризация (та же задача при разных tau):")
    print(f"{'tau':>8} {'Q(w)':>10} {'||w||':>10} {'cond гессиана':>15} {'точность теста':>16}")
    for tau in [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]:
        weights = logistic.newton_raphson(F, y, tau=tau)[-1]
        print(f"{tau:>8.2f} {logistic.log_loss(F, y, weights):>10.4f} {np.linalg.norm(weights):>10.3f} "
              f"{np.linalg.cond(logistic.hessian(F, y, weights, tau)):>15.3e} "
              f"{logistic.accuracy(y_test, logistic.predict(F_test, weights)):>16.4f}")


def equivalence(F, y, y01, tau=1.0):
    """Newton-Raphson, IRLS and IRLS in GLM notation are the same step written three ways."""
    newton_path = logistic.newton_raphson(F, y, tau=tau)
    irls_path = logistic.irls(F, y, tau=tau)
    glm_path = logistic.irls_glm(F, y01, tau=tau)
    shared = min(len(newton_path), len(irls_path), len(glm_path))

    print("\nтри записи одного и того же шага:")
    print(f"{'шаг':>4} {'||w|| Ньютон':>14} {'|Ньютон - IRLS|':>17} {'|Ньютон - GLM|':>16}")
    for step in range(shared):
        print(f"{step:>4} {np.linalg.norm(newton_path[step]):>14.6f} "
              f"{np.abs(newton_path[step] - irls_path[step]).max():>17.2e} "
              f"{np.abs(newton_path[step] - glm_path[step]).max():>16.2e}")
    print(f"  итераций: Ньютон {len(newton_path) - 1}, IRLS {len(irls_path) - 1}, "
          f"GLM {len(glm_path) - 1} (критерии остановки разные: по Q и по sigma)")

    from_least_squares = logistic.irls(F, y, tau=tau, start=logistic.least_squares(F, y))
    print(f"  IRLS из МНК-приближения, как в лекции: {len(from_least_squares) - 1} итераций, "
          f"ответ отличается на {np.abs(from_least_squares[-1] - newton_path[-1]).max():.2e}")


def reweighting(F, y, tau=1.0):
    """What IRLS actually does: objects near the border get the biggest weight."""
    weights = logistic.newton_raphson(F, y, tau=tau)[-1]
    sigma = logistic.sigmoid(logistic.margins(F, y, weights))
    importance = (1 - sigma) * sigma
    order = np.argsort(-importance)

    print("\nвеса объектов на последней итерации IRLS:")
    print(f"{'объект':>8} {'маржа M':>10} {'sigma':>9} {'вес (1-s)s':>12} {'1/sigma':>10}")
    for i in np.concatenate([order[:3], order[-3:]]):
        print(f"{i:>8} {logistic.margins(F, y, weights)[i]:>10.3f} {sigma[i]:>9.6f} "
              f"{importance[i]:>12.3e} {1 / sigma[i]:>10.3f}")
    print(f"  суммарный вес {importance.sum():.2f} на {len(y)} объектов: "
          f"{int((importance > 0.01 * importance.max()).sum())} объектов несут почти всю информацию")


def compare_with_reference(X, y01, F, y, X_test, F_test, tau=1.0):
    """sklearn minimizes C * sum log(1 + exp(-M)) + ||w||^2 / 2, so its C is our 1 / tau."""
    weights = logistic.newton_raphson(F, y, tau=tau)[-1]

    print(f"\nсравнение с sklearn.linear_model.LogisticRegression, C = 1 / tau = {1 / tau:g}:")
    print(f"{'solver':>18} {'итераций':>9} {'max|свободный член|':>21} {'max|веса|':>11} {'max|вероятности|':>18}")
    for solver in ["newton-cholesky", "lbfgs", "liblinear"]:
        reference = LogisticRegression(C=1 / tau, solver=solver, tol=1e-12, max_iter=10000).fit(X, y01)
        probabilities = logistic.probability(F_test, weights)
        print(f"{solver:>18} {int(np.ravel(reference.n_iter_)[0]):>9} "
              f"{abs(weights[0] - reference.intercept_[0]):>21.2e} "
              f"{np.abs(weights[1:] - reference.coef_[0]).max():>11.2e} "
              f"{np.abs(probabilities - reference.predict_proba(X_test)[:, 1]).max():>18.2e}")

    print("  liblinear штрафует ещё и свободный член, поэтому решает другую задачу и отвечает иначе")

    reference = LogisticRegression(C=1 / tau, solver="newton-cholesky", tol=1e-12).fit(X, y01)
    print(f"  значение Q в нашей точке {logistic.log_loss(F, y, weights, tau):.10f}, "
          f"в точке sklearn {logistic.log_loss(F, y, np.r_[reference.intercept_, reference.coef_[0]], tau):.10f}")
    print(f"  предсказания на тесте совпали у {int((logistic.predict(F_test, weights) == logistic.signed(reference.predict(X_test))).sum())} "
          f"из {len(F_test)} объектов")


def convergence(F, y, y01, tau=1.0):
    """How fast each method gets to the optimum, and how far the three Newton steps drift apart."""
    paths = {"метод Ньютона-Рафсона": logistic.newton_raphson(F, y, tau=tau),
             "IRLS": logistic.irls(F, y, tau=tau),
             "IRLS в записи GLM": logistic.irls_glm(F, y01, tau=tau),
             "градиентный спуск": logistic.gradient_descent(F, y, tau=tau)}
    losses = {name: np.array([logistic.log_loss(F, y, w, tau) for w in path]) for name, path in paths.items()}
    newton_path = paths["метод Ньютона-Рафсона"]
    divergences = {name: np.array([np.abs(w - v).max() for w, v in zip(newton_path, path)])
                   for name, path in paths.items() if name != "метод Ньютона-Рафсона"}

    print("\nсколько итераций нужно, чтобы подойти к оптимуму ближе чем на 1e-6:")
    best = min(values.min() for values in losses.values())
    for name, values in losses.items():
        close = np.flatnonzero(values - best < 1e-6)
        reached = f"{close[0]}" if len(close) else f"не дошёл за {len(values) - 1}"
        print(f"  {name:22s}: {reached:>20}, в конце Q - Q* = {values[-1] - best:.2e}")

    plots.plot_convergence(losses, divergences, "Сходимость методов обучения логистической регрессии",
                           os.path.join(IMAGES, "convergence.png"))


def regularization(X_train, y, F_test, y_test):
    """Choose tau by cross validation on the training data, then look at the test once."""
    taus = np.logspace(-3, 3, 19)
    cross_validated = np.array([logistic.cross_val(X_train, y, tau) for tau in taus])
    test_losses, norms, conditions = [], [], []
    for tau in taus:
        weights = logistic.newton_raphson(add_bias(standardize(X_train)), y, tau=tau)[-1]
        test_losses.append(logistic.log_loss(F_test, y_test, weights) / len(y_test))
        norms.append(np.linalg.norm(weights))
        conditions.append(np.linalg.cond(logistic.hessian(add_bias(standardize(X_train)), y, weights, tau)))

    best = int(np.argmin(cross_validated[:, 1]))
    print(f"\nподбор tau по кросс-валидации: tau = {taus[best]:.3g}, "
          f"log-loss {cross_validated[best, 1]:.4f}, точность {cross_validated[best, 0]:.4f}")
    print(f"  на тесте при этом tau: log-loss {test_losses[best]:.4f}, ||w|| = {norms[best]:.2f}")

    plots.plot_regularization(taus, cross_validated[:, 1], test_losses, norms, conditions,
                              "Регуляризация логистической регрессии",
                              os.path.join(IMAGES, "regularization.png"))


def probabilities(F, y, F_test, y_test, tau=1.0):
    """The model predicts posterior probabilities, so they should match the observed frequencies."""
    weights = logistic.newton_raphson(F, y, tau=tau)[-1]
    predicted_probabilities = logistic.probability(F_test, weights)
    predicted, observed, sizes = logistic.calibration(predicted_probabilities, y_test)

    print("\nкалибровка вероятностей на тесте:")
    print(f"{'группа':>8} {'размер':>8} {'предсказано':>13} {'наблюдается':>13}")
    for number, (p, o, size) in enumerate(zip(predicted, observed, sizes), start=1):
        print(f"{number:>8} {size:>8} {p:>13.4f} {o:>13.4f}")
    print(f"  средняя предсказанная вероятность {predicted_probabilities.mean():.4f}, "
          f"доля класса 1 на тесте {(y_test > 0).mean():.4f}")

    plots.plot_calibration(predicted, observed, sizes, predicted_probabilities, y_test,
                           "Вероятностная интерпретация логистической регрессии",
                           os.path.join(IMAGES, "calibration.png"))


def odds(F, y, feature_names, tau=1.0):
    """The linear part is the log odds, so exp(w_j) is how the odds change per one std of a feature."""
    weights = logistic.newton_raphson(F, y, tau=tau)[-1]
    order = np.argsort(-np.abs(weights[1:]))

    print("\nсамые влиятельные признаки (шансы на доброкачественную опухоль):")
    print(f"{'признак':>24} {'вес':>9} {'exp(вес)':>10}")
    for j in np.concatenate([order[:5], order[-2:]]):
        print(f"{feature_names[j]:>24} {weights[j + 1]:>9.3f} {np.exp(weights[j + 1]):>10.3f}")
    print(f"  свободный член {weights[0]:.3f}: шансы {np.exp(weights[0]):.3f} к одному "
          f"для объекта со средними признаками")


if __name__ == "__main__":
    main()
