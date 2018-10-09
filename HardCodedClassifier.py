from HardCodedModel import HardCodedModel


class HardCodedClassifier:
    def __init__(self, k):
        self.k = k

    def fit(self, data, targets):
        fit_model = HardCodedModel(self.k, data, targets)
        return fit_model
