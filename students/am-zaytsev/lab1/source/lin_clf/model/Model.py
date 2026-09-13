class Model:
    """Abstract interface for a linear model.

    Defines the contract used by trainers and optimizers: computing the
    margin, the loss, class predictions and the loss gradient.
    """

    def __init__(self):
        self.w = NotImplemented
        pass

    def predict_margin(self, feat_data, target_data, need_pad=False):
        raise NotImplementedError()

    def loss(self, feat_data, target_data):
        raise NotImplementedError()

    def predict(self, feat_data):
        raise NotImplementedError()

    def diff(self, feat_data, target_data):
        raise NotImplementedError()
