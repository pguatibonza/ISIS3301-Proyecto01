from joblib import load

class Model:

    def __init__(self,columns):
        self.model = load("assets/logistic_regression.joblib")

    def make_predictions(self, data):
        result = self.model.predict(data)
        return result
    def retrain_model(self,data):
        return None
