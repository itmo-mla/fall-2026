from lin_clf.model import Model
from lin_clf.optimizer import Optimizer


class SGD(Optimizer):
    mom_v = None

    def __init__(
        self, model: Model, tao: float, momentum_k: float, h: float | None = None
    ):
        super().__init__(model)
        self.tao = tao
        self.momentum_k = momentum_k
        self.h = h

    def count_h_star(self, dL, c1, c2):
        return (dL.T @ c1 + dL.T @ c2 @ self.model.w) / (dL.T @ c2 @ dL)

    def step(self, mini_batch_x, mini_batch_y):
        dL, c1, c2 = self.model.diff(mini_batch_x, mini_batch_y, self.tao)

        if self.mom_v is None:
            mom_v = dL
        mom_v = (1 - self.momentum_k) * mom_v + self.momentum_k * dL

        train_speed = self.h
        if self.h is None:
            train_speed = self.count_h_star(dL, c1, c2)
        # lin_model.w = lin_model.w - alpha * g
        self.model.w = self.model.w - train_speed * mom_v
