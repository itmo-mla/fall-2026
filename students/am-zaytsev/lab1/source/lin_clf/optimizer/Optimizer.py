from lin_clf.model import Model


class Optimizer:
    def __init__(self, model: Model):
        self.model = model

    def step(self):
        return NotImplementedError()
