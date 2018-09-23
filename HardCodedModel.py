class HardCodedModel:
    def __init__(self):
        pass

    def predict(self, testData):
        prediction = []
        for i in range(len(testData)):
            prediction.append(1)

        return prediction
