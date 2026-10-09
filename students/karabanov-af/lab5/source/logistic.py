import numpy as np

from data import add_bias, standardize


def sigmoid(z):
    """Logistic function 1 / (1 + exp(-z)), written through tanh so that exp() never overflows."""
    return 0.5 * (1 + np.tanh(0.5 * z))


def signed(y):
    """Labels {0, 1} as the margins {-1, +1} the lecture uses."""
    return 2 * y - 1


def margins(F, y, w):
    """M_i = y_i <w, x_i>: positive when the object is on its own side of the plane."""
    return y * (F @ w)


def without_bias(w):
    """Zero out the free term: it is the one weight that stays unregularized."""
    return np.concatenate([[0.0], w[1:]])


def log_loss(F, y, w, tau=0.0):
    """Q(w) = sum log(1 + exp(-M_i)) + tau/2 ||w||^2, computed without overflow on large margins."""
    return float(np.logaddexp(0, -margins(F, y, w)).sum() + 0.5 * tau * w[1:] @ w[1:])


def gradient(F, y, w, tau=0.0):
    """dQ/dw_j = -sum (1 - sigma_i) y_i f_j(x_i), where sigma_i = sigma(M_i)."""
    return -F.T @ ((1 - sigmoid(margins(F, y, w))) * y) + tau * without_bias(w)


def hessian(F, y, w, tau=0.0):
    """d2Q/dw_j dw_k = sum (1 - sigma_i) sigma_i f_j(x_i) f_k(x_i) = F.T D F."""
    sigma = sigmoid(margins(F, y, w))
    return F.T @ (((1 - sigma) * sigma)[:, None] * F) + tau * np.diag(without_bias(np.ones(len(w))))


def newton_raphson(F, y, tau=1.0, step=1.0, n_iterations=50, tolerance=1e-10):
    """w := w - h * inv(Q'') Q', the step the lecture writes out for the log-loss.

    Returns the whole trajectory of weights, starting from zeros: the caller reads
    the answer off its last row and the convergence off the rest.
    """
    trajectory = [np.zeros(F.shape[1])]
    for _ in range(n_iterations):
        w = trajectory[-1]
        trajectory.append(w - step * np.linalg.solve(hessian(F, y, w, tau), gradient(F, y, w, tau)))
        if abs(log_loss(F, y, trajectory[-2], tau) - log_loss(F, y, trajectory[-1], tau)) < tolerance:
            break
    return np.array(trajectory)


def ridge(A, b, tau):
    """Solve the normal system of a weighted least squares problem with an L2 penalty."""
    return np.linalg.solve(A + tau * np.diag(without_bias(np.ones(len(A)))), b)


def irls(F, y, tau=1.0, step=1.0, n_iterations=50, tolerance=1e-10, start=None):
    """The same Newton step written as a weighted least squares problem, as on slide 9.

    gamma_i = sqrt((1 - sigma_i) sigma_i) are the object weights, y~_i = y_i sqrt((1 - sigma_i) / sigma_i)
    the modified answers, and w += h (F~.T F~)^-1 F~.T y~ is an ordinary least squares step.
    """
    trajectory = [np.zeros(F.shape[1]) if start is None else start]
    for _ in range(n_iterations):
        w = trajectory[-1]
        sigma = sigmoid(margins(F, y, w))
        gamma = np.sqrt((1 - sigma) * sigma)
        weighted = gamma[:, None] * F
        answers = y * np.sqrt((1 - sigma) / sigma)
        trajectory.append(w + step * ridge(weighted.T @ weighted,
                                           weighted.T @ answers - tau * without_bias(w), tau))
        if np.abs(sigmoid(margins(F, y, trajectory[-1])) - sigma).max() < tolerance:
            break
    return np.array(trajectory)


def irls_glm(F, y, tau=1.0, step=1.0, n_iterations=50, tolerance=1e-10):
    """The same loop in the general GLM notation of slide 19, with labels {0, 1}.

    For the Bernoulli distribution c'(theta) = sigma(theta) is the mean, c''(theta) = sigma (1 - sigma)
    the variance and phi = 1, so gamma_i = sqrt(c''(theta_i)) and y~_i = (y_i - c'(theta_i)) / gamma_i.

    The variance has a floor: a saturated object gets gamma_i = 0 and turns y~_i into 0/0.
    """
    trajectory = [np.zeros(F.shape[1])]
    for _ in range(n_iterations):
        w = trajectory[-1]
        mean = sigmoid(F @ w)
        gamma = np.sqrt(np.maximum(mean * (1 - mean), 1e-30))
        weighted = gamma[:, None] * F
        answers = (y - mean) / gamma
        trajectory.append(w + step * ridge(weighted.T @ weighted,
                                           weighted.T @ answers - tau * without_bias(w), tau))
        if np.abs(sigmoid(F @ trajectory[-1]) - mean).max() < tolerance:
            break
    return np.array(trajectory)


def least_squares(F, y):
    """Plain least squares, the zero approximation the lecture starts IRLS from."""
    return np.linalg.lstsq(F, y, rcond=None)[0]


def gradient_descent(F, y, tau=1.0, step=None, n_iterations=300):
    """Plain gradient descent, for contrast: without the Hessian the convergence is only linear.

    The default step is 1 / L, where L is the largest eigenvalue of the Hessian at the starting point.
    """
    trajectory = [np.zeros(F.shape[1])]
    if step is None:
        step = 1 / np.linalg.eigvalsh(hessian(F, y, trajectory[-1], tau)).max()
    for _ in range(n_iterations):
        trajectory.append(trajectory[-1] - step * gradient(F, y, trajectory[-1], tau))
    return np.array(trajectory)


def cross_val(X, y, tau, folds=5, seed=0):
    """Accuracy and log-loss averaged over `folds` parts, with the standardization refitted inside."""
    index = np.random.default_rng(seed).permutation(len(y))
    scores = []
    for fold in np.array_split(index, folds):
        train = np.setdiff1d(index, fold)
        w = newton_raphson(add_bias(standardize(X[train])), y[train], tau=tau)[-1]
        held_out = add_bias(standardize(X[fold], X[train]))
        scores.append((accuracy(y[fold], predict(held_out, w)),
                       log_loss(held_out, y[fold], w) / len(fold)))
    return np.mean(scores, axis=0)


def calibration(probabilities, y, n_bins=8):
    """Reliability of the probabilities: predicted against observed frequency, in bins of equal size."""
    order = np.argsort(probabilities)
    predicted, observed, sizes = [], [], []
    for part in np.array_split(order, n_bins):
        predicted.append(probabilities[part].mean())
        observed.append((y[part] > 0).mean())
        sizes.append(len(part))
    return np.array(predicted), np.array(observed), np.array(sizes)


def probability(F, w):
    """Posterior probability of the positive class: P(y = +1 | x) = sigma(<w, x>)."""
    return sigmoid(F @ w)


def predict(F, w):
    return np.where(F @ w >= 0, 1, -1)


def accuracy(y_true, y_predicted):
    return float(np.mean(y_true == y_predicted))
