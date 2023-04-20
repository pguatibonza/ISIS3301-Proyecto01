import os
from joblib import load, dump
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Model:
    def __init__(self, columns):
        model_path = os.path.join(os.getcwd(), "assets", "logistic_regression.joblib")
        if not os.path.exists(model_path):
            raise Exception(f"Model file not found in {model_path}")
        self.model = load(model_path)
        if self.model is None:
            raise Exception("Failed to load model")
        self.columns = columns

    def make_predictions(self, data):
        df = pd.DataFrame(data, columns=self.columns)
        result = self.model.predict(df)
        return {"prediction": result}

    def retrain_model(self, data):
        X = data[["review_es"]]
        Y = data[["sentimiento"]]
        self.model.fit(X, Y)
        y_pred = self.model.predict(X)
        dump(self.model, "assets/logistic_regression.joblib")

        # Compare Y and self.model.predict(X) to see how good the model is
        rmse = mean_squared_error(Y, y_pred, squared=False)
        mae = mean_absolute_error(Y, y_pred)
        r2 = r2_score(Y, y_pred)

        return {"rmse": rmse, "mae": mae, "r2": r2}