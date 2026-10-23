import unittest

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

from data import load_data
from logistic import LogisticRegression


class LogisticTest(unittest.TestCase):
    def test_reference_and_convergence(self):
        x_train, x_test, y_train, _ = load_data()
        for l2 in (0.01, 0.1, 1, 10):
            reference = SklearnLogisticRegression(C=1 / l2, tol=1e-12, max_iter=2000).fit(x_train, y_train)
            for method in ("newton", "irls"):
                with self.subTest(l2=l2, method=method):
                    model = LogisticRegression(method=method, l2=l2).fit(x_train, y_train)
                    self.assertTrue(model.converged_)
                    self.assertLessEqual(model.gradient_history_[-1], model.tol)
                    self.assertTrue(np.all(np.diff(model.loss_history_) <= 1e-10))
                    np.testing.assert_allclose(model.predict_proba(x_test), reference.predict_proba(x_test), atol=1e-6, rtol=0)
                    np.testing.assert_array_equal(model.predict(x_test), reference.predict(x_test))

    def test_newton_irls_same_iterations(self):
        x_train, _, y_train, _ = load_data()
        models = [LogisticRegression(method=method).fit(x_train, y_train) for method in ("newton", "irls")]
        np.testing.assert_allclose(models[0].loss_history_, models[1].loss_history_, atol=1e-10, rtol=0)
        np.testing.assert_allclose(models[0].coef_, models[1].coef_, atol=1e-10, rtol=0)
        for method in ("newton", "irls"):
            limited = LogisticRegression(method=method, max_iter=1).fit(x_train, y_train)
            self.assertFalse(limited.converged_)
            self.assertEqual(limited.n_iter_, 1)

    def test_derivatives(self):
        rng = np.random.default_rng(3)
        x = np.column_stack((np.ones(12), rng.normal(size=(12, 3))))
        y = rng.integers(0, 2, size=12)
        beta = rng.normal(size=4)
        penalty = np.diag([0, 0.7, 0.7, 0.7])

        def loss(b):
            eta = x @ b
            return np.sum(np.logaddexp(0, eta) - y * eta) + 0.5 * b @ penalty @ b

        def gradient(b):
            return x.T @ (expit(x @ b) - y) + penalty @ b

        epsilon = 1e-5
        directions = np.eye(4) * epsilon
        numerical_gradient = np.array([(loss(beta + d) - loss(beta - d)) / (2 * epsilon) for d in directions])
        numerical_hessian = np.column_stack([(gradient(beta + d) - gradient(beta - d)) / (2 * epsilon) for d in directions])
        probability = expit(x @ beta)
        hessian = x.T @ ((probability * (1 - probability))[:, None] * x) + penalty
        np.testing.assert_allclose(gradient(beta), numerical_gradient, atol=1e-8)
        np.testing.assert_allclose(hessian, numerical_hessian, atol=1e-8)

    def test_separable_data_and_large_logits(self):
        x = np.array([[-3.], [-2.], [-1.], [1.], [2.], [3.]])
        y = np.array([0, 0, 0, 1, 1, 1])
        for method in ("newton", "irls"):
            model = LogisticRegression(method=method, l2=0.1).fit(x, y)
            self.assertTrue(model.converged_)
            np.testing.assert_array_equal(model.predict(x), y)
            probabilities = model.predict_proba([[-1e6], [1e6]])
            self.assertTrue(np.isfinite(probabilities).all())
            np.testing.assert_allclose(probabilities.sum(axis=1), 1)

    def test_duplicate_features_without_penalty(self):
        x = np.array([[-2.], [-1.], [0.], [0.], [1.], [2.]])
        x = np.column_stack((x, x))
        y = np.array([0, 1, 0, 1, 0, 1])
        for method in ("newton", "irls"):
            model = LogisticRegression(method=method, l2=0).fit(x, y)
            self.assertTrue(model.converged_)
            self.assertTrue(np.isfinite(model.coef_).all())


if __name__ == "__main__":
    unittest.main()
