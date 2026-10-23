import  numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import  load_iris

from dataset import load_data, TARGET, prepare_data


class LinearClassifier:

    def __init__(self,seed = 432):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.weights = None
        self.velocity = None
        self.q_ema = None
        self.loss_history = []


    def _add_one(self, x: np.ndarray) -> np.ndarray:
        ones = np.ones((x.shape[0], 1))
        return np.hstack((ones, x))
    def _sample_batch(self, x: np.ndarray, y: np.ndarray, batch_size: int = 16, sampling = "uniform", random_prob= 0.2):
        n = x.shape[0]
        eps = 0.01
        if sampling == "uniform":
            idx = self.rng.choice(n, size=batch_size, replace=False)
        elif sampling == "margin":
            m = np.abs(self.margin(x, y))
            p_margin = 1.0 / (m+eps)
            p_margin = p_margin/p_margin.sum()
            p = random_prob / n+ (1.0 - random_prob) * p_margin
            idx = self.rng.choice(n, size= batch_size, replace = False, p = p)
        else:
            raise  ValueError(f"Неизвестный семплинг {sampling}")
        return  x[idx], y[idx]


    def random_init(self, x: np.ndarray):
        n = x.shape[1] + 1
        self.weights = self.rng.uniform(-1/(2*n), 1/(2*n), size=n)


    def corr_init(self, x: np.ndarray, y: np.ndarray, center_data: bool = False):
        x_proc = x.copy()
        y_proc = y.copy()
        if center_data:
            x_proc = x_proc - np.mean(x_proc, axis = 0)
            y_proc = y_proc - np.mean(y_proc)
        numerator = x_proc.T @ y_proc
        denominators = np.sum(x_proc**2, axis=0)
        denominators = np.where(denominators == 0, 1e-9, denominators)
        self.weights = numerator / denominators
        self.weights = np.insert(self.weights, 0,0)
        return  self.weights




    def margin(self, x: np.ndarray, y:  np.ndarray):
        x_ext = self._add_one(x)
        return y * (x_ext @ self.weights)

    def mse_grad(self, x: np.ndarray, y:np.ndarray, tao:float = 0.0) -> np.ndarray:
        x_ect = self._add_one(x)

        margin = self.margin(x,y)
        grad_val = 2 * -(x_ect.T @ ((1- margin) * y)) / x.shape[0]
        if tao > 0:
            l2_grad = tao * self.weights.copy()
            l2_grad[0] = 0.0
            grad_val = grad_val + l2_grad

        return  grad_val

    def loss(self, x: np.ndarray, y: np.ndarray, tao: float = 0.0):
        margins = self.margin(x, y)
        mse_loss = np.mean((1- margins)**2)
        l2_loss = (tao/2) * np.sum(self.weights[1:]**2)
        return  mse_loss + l2_loss
    def update_q_ema(self, current_loss: float, alpha: float = 0.05)-> float:
        if self.q_ema is None:
            self.q_ema = current_loss
        else:
            self.q_ema = (1- alpha) * self.q_ema + alpha * current_loss
        return  self.q_ema


    def steepest_step(self,x,y,grad,tao):
        eps = 1e-12
        x_ext = self._add_one(x)
        d = y * (x_ext @ grad)
        denom = 2.0 * np.mean(d**2) + tao * np.sum(grad[1:] ** 2)
        return (grad @ grad) / (denom + eps)

    def sgd_momentum(self, x_batch, y_batch, lr=0.001, momentum=0.9,
                     tao=0.0, step="fixed"):
        grad = self.mse_grad(x_batch, y_batch, tao)

        if step == "steepest":
            eta = self.steepest_step(x_batch, y_batch, grad, tao)
            self.weights -= eta * grad

            return

        if self.velocity is None:
            self.velocity = np.zeros_like(self.weights)
        self.velocity = momentum * self.velocity + (1 - momentum) * grad
        self.weights -= lr * self.velocity



    def fit(self, x: np.ndarray, y:np.ndarray, n_iter: int = 200,batch_size:int = 16, verbose = False, weight_init = "random", tao:float = 0.0, alpha:float = 0.05, momentum: float = 0.9, sampling = "uniform", step = "fixed" ):
        if weight_init == "random":
            self.random_init(x)
        elif weight_init == "correlation":
            self.corr_init(x,y)
        else:
            raise ValueError("Укажите random или correlation ")
        self.q_ema = None
        self.loss_history = []
        self.velocity = None


        for i in range(n_iter):
            x_b, y_b = self._sample_batch(x,y, batch_size, sampling = sampling)
            current_loss = self.loss(x_b, y_b, tao)
            self.update_q_ema(current_loss, alpha)
            self.loss_history.append(current_loss)

            self.sgd_momentum(x_b, y_b, step = step ,momentum = momentum, tao=tao )

            if verbose and i % verbose == 0:
                print(f"{i:5d}  ema={self.q_ema:.4f}  full={self.loss(x, y, tao):.4f}")
        return  self

    def predict(self, x: np.ndarray):
        x_ect = self._add_one(x)
        return np.sign(x_ect @ self.weights)


