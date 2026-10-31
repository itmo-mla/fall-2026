import unittest

import numpy as np
from sklearn.svm import SVC

from data import load_iris_split
from svm import KernelSVM, LinearSVM


class SVMTests(unittest.TestCase):
    def test_both_kernels_against_reference_and_kkt(self):
        x_train, x_test, y_train, _ = load_iris_split()
        for kernel in ("linear", "rbf"):
            with self.subTest(kernel=kernel):
                model = KernelSVM(kernel=kernel, gamma=0.25).fit(x_train, y_train)
                reference = SVC(C=1, kernel=kernel, gamma=0.25, tol=1e-7).fit(x_train, y_train)
                np.testing.assert_array_equal(model.predict(x_test), reference.predict(x_test))
                np.testing.assert_allclose(
                    model.decision_function(x_test), reference.decision_function(x_test), atol=1e-3
                )
                self.assertTrue(np.all(model.alpha >= -1e-7))
                self.assertTrue(np.all(model.alpha <= model.c + 1e-7))
                self.assertAlmostEqual(model.alpha @ y_train, 0, places=6)
                margins = y_train * model.decision_function(x_train)
                free = (model.alpha > 1e-6) & (model.alpha < model.c - 1e-6)
                np.testing.assert_allclose(margins[free], 1, atol=1e-3)
                self.assertTrue(np.all(margins[model.alpha <= 1e-6] >= 1 - 1e-3))
                self.assertTrue(np.all(margins[model.alpha >= model.c - 1e-6] <= 1 + 1e-3))

    def test_rbf_separates_xor(self):
        x = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        y = np.array([-1, 1, 1, -1])
        model = KernelSVM(c=10, kernel="rbf", gamma=1).fit(x, y)
        np.testing.assert_array_equal(model.predict(x), y)
        self.assertTrue(np.all(y * model.decision_function(x) >= 1 - 1e-3))

    def test_bias_without_free_support_vectors(self):
        x = np.array([[-1.0], [1.0]])
        y = np.array([-1, 1])
        for kernel in ("linear", "rbf"):
            with self.subTest(kernel=kernel):
                model = KernelSVM(c=0.01, kernel=kernel).fit(x, y)
                np.testing.assert_allclose(model.alpha, model.c, atol=1e-7)
                self.assertAlmostEqual(model.bias, 0, places=6)
                np.testing.assert_array_equal(model.predict(x), y)

    def test_linear_interface_is_preserved(self):
        x_train, x_test, y_train, _ = load_iris_split()
        linear = LinearSVM().fit(x_train, y_train)
        kernel = KernelSVM(kernel="linear").fit(x_train, y_train)
        np.testing.assert_allclose(
            x_test @ linear.weights + linear.bias, kernel.decision_function(x_test), atol=1e-8
        )


if __name__ == "__main__":
    unittest.main()
