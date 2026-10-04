from lin_clf.model import Model


class Optimizer:
    """Base class for optimizers that update a model's weights.

    Subclasses implement ``step`` to modify ``model.w``.
    """

    def __init__(self, model: Model):
        self.model = model

    def step(self):
        return NotImplementedError()
