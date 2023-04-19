from joblib import load, dump
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



class Model:

    def __init__(self, columns):
        self.model = load("assets/logistic_regression.joblib")

    def make_predictions(self, data):
        result = self.model.predict(data)
        df = pd.DataFrame(result, columns=['predict'])
        return df

    def retrain_model(self, data):
        X = data[['review_es']]
        Y = data[['sentimiento']]
        self.model.fit(X, Y)
        y_pred = self.model.predict(X)
        dump(self.model, 'assets/logistic_regression.joblib')
        
        #Compare Y and self.model.predict(X) to see how good the model is
        rmse = mean_squared_error(y, y_pred, squared=False)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        return {"rmse": rmse, "mae": mae, "r2": r2}
        







