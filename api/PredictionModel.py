from joblib import load
import pandas as pd


class Model:

    def __init__(self, columns):
        self.model = load("assets/logistic_regression.joblib")

    def make_predictions(self, data):
        result = self.model.predict(data)
        df = pd.DataFrame(result, columns=['predict'])

    def retrain_model(self, data):
        return None
