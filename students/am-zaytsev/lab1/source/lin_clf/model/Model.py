class Model:
    def __init__(self):
        self.w = NotImplemented
        pass

    def predict_margin(self, feat_data, target_data, need_pad=False):
        raise NotImplementedError()

    def loss(self, feat_data, target_data):
        raise NotImplementedError()

    def predict(self, feat_data):
        raise NotImplementedError()

    def diff(self, feat_data, target_data, tao):
        raise NotImplementedError()
